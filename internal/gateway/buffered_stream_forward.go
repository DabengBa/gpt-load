//go:build !windows

package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"gpt-load/internal/execution"
)

func (forwarder *ExecutionForwarder) forwardBufferedStream(
	ctx context.Context,
	input ForwardInput,
	downstream http.ResponseWriter,
) UpstreamResult {
	result := UpstreamResult{BufferedStream: true, DispatchState: execution.DispatchNotSent}
	if downstream == nil {
		result.Err = errors.New("buffered stream downstream writer is required")
		return result
	}
	spool, err := newBufferedStreamSpool(bufferedStreamMemoryLimit, bufferedStreamResponseLimit)
	if err != nil {
		result.Err = err
		return result
	}
	defer func() { _ = spool.Close() }()

	output := BufferedStreamOutput{}
	session := bufferedStreamSessionFromContext(ctx)
	output.HTTPCommitted = session.HTTPCommitted
	output.ClientVisibleBytes = session.VisibleBytes
	output.PayloadReleased = session.PayloadReleased
	writeTimeout := forwarder.writeTimeout
	if writeTimeout <= 0 {
		writeTimeout = downstreamWriteTimeout
	}
	controller := newStreamWriteController(downstream, writeTimeout)
	defer func() { _ = controller.clear() }()
	heartbeatHeaders := normalizeStreamResponseHeaders(http.Header{
		"Content-Type": {"text/event-stream"},
	})
	if !session.HTTPCommitted {
		for name, values := range heartbeatHeaders {
			for _, value := range values {
				downstream.Header().Set(name, value)
			}
		}
		downstream.Header().Del("Content-Length")
		downstream.Header().Del("Content-Encoding")
		if err := controller.writeHeader(http.StatusOK); err != nil {
			result.Err = fmt.Errorf("write buffered stream headers: %w", err)
			return result
		}
		output.markHTTPCommitted()
		session.HTTPCommitted = true
		result.Committed = true
		written, err := controller.write([]byte(bufferedStreamHeartbeat))
		if err != nil || written != len(bufferedStreamHeartbeat) {
			if err == nil {
				err = io.ErrShortWrite
			}
			result.Err = &streamFailure{kind: streamFailureDownstreamWrite, err: fmt.Errorf("write buffered stream heartbeat: %w", err)}
			result.HTTPCommitted = output.HTTPCommitted
			result.ClientVisibleBytes = int64(written)
			return result
		}
		output.markVisibleBytes(int64(written))
		session.VisibleBytes += int64(written)
		if err := controller.flush(); err != nil {
			result.Err = &streamFailure{kind: streamFailureDownstreamWrite, err: fmt.Errorf("flush buffered stream heartbeat: %w", err)}
			result.HTTPCommitted = output.HTTPCommitted
			result.ClientVisibleBytes = output.ClientVisibleBytes
			return result
		}
	}
	result.Committed = session.HTTPCommitted

	capture := newBufferedStreamCaptureWriter(spool)
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	controller.ctx = streamCtx
	done := make(chan UpstreamResult, 1)
	go func() {
		done <- forwarder.forwardStream(streamCtx, input, capture)
	}()
	heartbeatInterval := forwarder.heartbeatInterval
	if heartbeatInterval <= 0 {
		heartbeatInterval = 15 * time.Second
	}
	ticker := time.NewTicker(heartbeatInterval)
	defer ticker.Stop()
	var heartbeatErr error
	for {
		select {
		case result = <-done:
			goto streamDone
		case <-ticker.C:
			written, writeErr := controller.write([]byte(bufferedStreamHeartbeat))
			if writeErr == nil && written == len(bufferedStreamHeartbeat) {
				output.markVisibleBytes(int64(written))
				session.VisibleBytes += int64(written)
				writeErr = controller.flush()
			} else if writeErr == nil {
				writeErr = io.ErrShortWrite
			}
			if writeErr != nil {
				heartbeatErr = &streamFailure{kind: streamFailureDownstreamWrite, err: fmt.Errorf("write buffered stream heartbeat: %w", writeErr)}
				cancel()
				result = <-done
				goto streamDone
			}
		case <-ctx.Done():
			cancel()
			result = <-done
			goto streamDone
		}
	}

streamDone:
	responseID := result.Stream.ResponseID
	if heartbeatErr != nil {
		result.Err = heartbeatErr
		result.Stream = streamTerminalObservationWithResponseID(StreamEndDownstreamWriteFailure, responseID)
	}
	result.BufferedStream = true
	result.Committed = true // preserve the irreversible HTTP commitment.
	result.HTTPCommitted = output.HTTPCommitted
	result.ClientVisibleBytes = output.ClientVisibleBytes
	var releaseStartedAt time.Time
	if result.Err == nil && result.Stream.EndReason == StreamEndCleanEOF {
		if err := streamCtx.Err(); err != nil {
			result.Err = &streamFailure{kind: streamFailureDownstreamWrite, err: fmt.Errorf("buffered stream deadline expired before release: %w", err)}
			result.Stream = streamTerminalObservationWithResponseID(StreamEndDownstreamWriteFailure, responseID)
		} else {
			releaseStartedAt = time.Now()
			if err := output.markPayloadReleased(); err != nil {
				result.Err = err
				result.Stream = streamTerminalObservationWithResponseID(StreamEndDownstreamWriteFailure, responseID)
			} else {
				session.PayloadReleased = true
				writer := bufferedContextWriter{ctx: streamCtx, writer: &bufferedControllerWriter{controller: controller}}
				payloadBytes, replayErr := spool.ReplayTo(&writer)
				if replayErr == nil {
					replayErr = controller.flush()
				}
				output.markVisibleBytes(payloadBytes)
				if replayErr != nil {
					result.Err = &streamFailure{kind: streamFailureDownstreamWrite, err: fmt.Errorf("release buffered stream payload: %w", replayErr)}
					result.Stream = streamTerminalObservationWithResponseID(StreamEndDownstreamWriteFailure, responseID)
				}
			}
		}
	}
	result.HTTPCommitted = output.HTTPCommitted
	result.PayloadReleased = output.PayloadReleased
	session.HTTPCommitted = output.HTTPCommitted
	session.VisibleBytes = output.ClientVisibleBytes
	session.PayloadReleased = output.PayloadReleased
	result.ClientVisibleBytes = output.ClientVisibleBytes
	result.BufferedPeakBytes = spool.Size()
	result.BufferedSpilled = spool.Spilled()
	result.PayloadReleaseStartedAt = releaseStartedAt
	return result
}

type bufferedContextWriter struct {
	ctx    context.Context
	writer io.Writer
}

func (writer *bufferedContextWriter) Write(data []byte) (int, error) {
	if err := writer.ctx.Err(); err != nil {
		return 0, err
	}
	return writer.writer.Write(data)
}

type bufferedControllerWriter struct {
	controller *streamWriteController
}

func (writer *bufferedControllerWriter) Write(data []byte) (int, error) {
	if writer == nil || writer.controller == nil {
		return 0, errors.New("buffered stream writer is required")
	}
	written, err := writer.controller.write(data)
	if err != nil {
		return written, err
	}
	if written != len(data) {
		return written, io.ErrShortWrite
	}
	return written, nil
}
