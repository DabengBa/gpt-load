//go:build !windows

package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
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

	// The heartbeat must not precede provider dispatch: a local pre-dispatch
	// failure keeps its own HTTP status, so HTTP 200 is committed only once the
	// attempt either proves dispatch or terminates after dispatch.
	commitClient := func() (UpstreamResult, bool) {
		if output.HTTPCommitted {
			return UpstreamResult{}, true
		}
		for name, values := range heartbeatHeaders {
			for _, value := range values {
				downstream.Header().Set(name, value)
			}
		}
		downstream.Header().Del("Content-Length")
		downstream.Header().Del("Content-Encoding")
		if err := controller.writeHeader(http.StatusOK); err != nil {
			return UpstreamResult{BufferedStream: true, DispatchState: execution.DispatchNotSent,
				Err: fmt.Errorf("write buffered stream headers: %w", err)}, false
		}
		output.markHTTPCommitted()
		session.HTTPCommitted = true
		written, err := controller.write([]byte(bufferedStreamHeartbeat))
		if err != nil || written != len(bufferedStreamHeartbeat) {
			if err == nil {
				err = io.ErrShortWrite
			}
			return UpstreamResult{BufferedStream: true, DispatchState: execution.DispatchNotSent,
				Committed:          output.HTTPCommitted,
				HTTPCommitted:      output.HTTPCommitted,
				ClientVisibleBytes: int64(written),
				Err: &streamFailure{kind: streamFailureDownstreamWrite,
					err: fmt.Errorf("write buffered stream heartbeat: %w", err)}}, false
		}
		output.markVisibleBytes(int64(written))
		session.VisibleBytes += int64(written)
		if err := controller.flush(); err != nil {
			return UpstreamResult{BufferedStream: true, DispatchState: execution.DispatchNotSent,
				Committed:          output.HTTPCommitted,
				HTTPCommitted:      output.HTTPCommitted,
				ClientVisibleBytes: output.ClientVisibleBytes,
				Err: &streamFailure{kind: streamFailureDownstreamWrite,
					err: fmt.Errorf("flush buffered stream heartbeat: %w", err)}}, false
		}
		return UpstreamResult{}, true
	}

	capture := newBufferedStreamDispatchWriter(newBufferedStreamCaptureWriter(spool))
	dispatchProof := capture.dispatched
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	controller.ctx = streamCtx
	done := make(chan UpstreamResult, 1)
	go func() {
		done <- forwarder.forwardStream(streamCtx, input, capture)
	}()
	// abort stops the executor goroutine and drains it before any return path
	// that abandons the attempt, so the spool is never written after it closes.
	abort := func(failure UpstreamResult) UpstreamResult {
		cancel()
		<-done
		return failure
	}

	heartbeatInterval := forwarder.heartbeatInterval
	if heartbeatInterval <= 0 {
		heartbeatInterval = 15 * time.Second
	}
	ticker := time.NewTicker(heartbeatInterval)
	defer ticker.Stop()
	dispatchWatch := dispatchProof
	var heartbeatErr error
streamWait:
	for {
		select {
		case result = <-done:
			if !output.HTTPCommitted {
				if !bufferedStreamDispatched(dispatchProof) && result.DispatchState == execution.DispatchNotSent {
					// Local pre-dispatch failure: keep the original HTTP status.
					break streamWait
				}
				// A canceled execution context is already terminal: the heartbeat is
				// not written after the downstream deadline has passed.
				if streamCtx.Err() == nil {
					if failure, committed := commitClient(); !committed {
						return abort(failure)
					}
				}
			}
			break streamWait
		case <-dispatchWatch:
			dispatchWatch = nil
			if streamCtx.Err() == nil {
				if failure, committed := commitClient(); !committed {
					return abort(failure)
				}
			}
		case <-ticker.C:
			if !output.HTTPCommitted {
				// No dispatch proof yet; a local failure may still keep its own
				// HTTP status, so the client wait continues without a heartbeat.
				continue
			}
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
				break streamWait
			}
		case <-ctx.Done():
			cancel()
			result = <-done
			break streamWait
		}
	}

	responseID := result.Stream.ResponseID
	if heartbeatErr != nil {
		result.Err = heartbeatErr
		result.Stream = streamTerminalObservationWithResponseID(StreamEndDownstreamWriteFailure, responseID)
	}
	result.BufferedStream = true
	result.Committed = output.HTTPCommitted // irreversible HTTP commitment only.
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

// bufferedStreamDispatchWriter wraps the buffered capture spool and reports the
// first execution write through dispatched. The executor only reaches the
// capture spool after passing its local pre-dispatch validation, so that signal
// is the dispatch proof which authorizes the downstream HTTP 200 heartbeat.
type bufferedStreamDispatchWriter struct {
	writer     *bufferedStreamCaptureWriter
	dispatched chan struct{}
	once       sync.Once
}

func newBufferedStreamDispatchWriter(writer *bufferedStreamCaptureWriter) *bufferedStreamDispatchWriter {
	return &bufferedStreamDispatchWriter{writer: writer, dispatched: make(chan struct{})}
}

func (writer *bufferedStreamDispatchWriter) Header() http.Header { return writer.writer.Header() }

func (writer *bufferedStreamDispatchWriter) WriteHeader(status int) {
	writer.markDispatch()
	writer.writer.WriteHeader(status)
}

func (writer *bufferedStreamDispatchWriter) Write(data []byte) (int, error) {
	writer.markDispatch()
	return writer.writer.Write(data)
}

func (writer *bufferedStreamDispatchWriter) FlushError() error { return writer.writer.FlushError() }

func (writer *bufferedStreamDispatchWriter) markDispatch() {
	writer.once.Do(func() { close(writer.dispatched) })
}

// bufferedStreamDispatchSignaler is the dispatch proof channel of the buffered
// capture path. forwardStream signals it as soon as the attempt is known to
// have reached the provider.
func markDownstreamDispatched(downstream http.ResponseWriter) {
	if signaler, ok := downstream.(interface{ markDispatch() }); ok {
		signaler.markDispatch()
	}
}

// bufferedStreamDispatched reports whether the execution path already reached
// the buffered spool, i.e. the attempt passed local pre-dispatch validation.
func bufferedStreamDispatched(dispatched <-chan struct{}) bool {
	select {
	case <-dispatched:
		return true
	default:
		return false
	}
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
