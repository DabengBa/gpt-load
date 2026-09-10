package gateway

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"gpt-load/internal/execution"
	"gpt-load/internal/health"
)

func TestBufferedStreamFailedAttemptKeepsCommittedAndCanRetryBeforeRelease(t *testing.T) {
	executor := fakeExecutionExecutor{stream: func(
		_ context.Context,
		_ execution.AttemptSpec,
		sink execution.StreamSink,
	) execution.StreamResult {
		if err := sink(execution.StreamEvent{
			Sequence: 1, Kind: execution.StreamEventReady,
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"text/event-stream"}},
		}); err != nil {
			t.Fatal(err)
		}
		if err := sink(execution.StreamEvent{
			Sequence: 2, Kind: execution.StreamEventData,
			Data: []byte("data: {\"id\":\"chat_1\""),
		}); err != nil {
			t.Fatal(err)
		}
		return execution.StreamResult{
			DispatchState:   execution.DispatchMaybeSent,
			ResponseStarted: true, StatusCode: http.StatusOK,
			Header: http.Header{"Content-Type": {"text/event-stream"}},
		}
	}}
	input := executionForwardInput()
	input.BufferedStream = true
	recorder := httptest.NewRecorder()
	result := NewExecutionForwarder(executor).ForwardStream(context.Background(), input, recorder)
	if !result.Committed || !result.HTTPCommitted || result.PayloadReleased ||
		recorder.Body.String() != bufferedStreamHeartbeat {
		t.Fatalf("failed buffered attempt = %#v, body = %q", result, recorder.Body.String())
	}
	decision := judgeUpstreamResult(result, timeNowForBufferedTest(), health.DecisionContext{
		Method:                 http.MethodPost,
		Operation:              execution.OperationChatCompletion,
		BufferedReplayEligible: true,
	})
	if decision.Retry != health.RetryNextCandidate {
		t.Fatalf("failed buffered attempt decision = %#v, want retry", decision)
	}
}

func TestBufferedStreamReleaseStopsOnPartialDownstreamWrite(t *testing.T) {
	writer := &bufferedPartialWriteResponseWriter{header: make(http.Header)}
	executor := fakeExecutionExecutor{stream: func(_ context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			return execution.StreamResult{Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}, DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
		}
		for sequence, data := range [][]byte{
			[]byte(`data: {"choices":[{"index":0,"finish_reason":"stop"}]}` + "\n\n"),
			[]byte("data: [DONE]\n\n"),
		} {
			if err := sink(execution.StreamEvent{Sequence: uint64(sequence + 2), Kind: execution.StreamEventData, Data: data}); err != nil {
				return execution.StreamResult{Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}, DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
			}
		}
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	input := executionForwardInput()
	input.BufferedStream = true
	result := NewExecutionForwarder(executor).ForwardStream(context.Background(), input, writer)
	if result.Stream.EndReason != StreamEndDownstreamWriteFailure || !result.PayloadReleased {
		t.Fatalf("partial release result = %#v", result)
	}
	if got := bufferedStreamReservedBytes.Load(); got != 0 {
		t.Fatalf("reserved budget after partial release = %d", got)
	}
}

func TestBufferedStreamCancellationCleansSpool(t *testing.T) {
	started := make(chan struct{})
	executor := fakeExecutionExecutor{stream: func(ctx context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		close(started)
		if err := sink(execution.StreamEvent{
			Sequence: 1, Kind: execution.StreamEventReady,
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"text/event-stream"}},
		}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
		}
		if err := sink(execution.StreamEvent{
			Sequence: 2, Kind: execution.StreamEventData,
			Data: []byte("data: {\"id\":\"cancelled\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"buffered\"},\"finish_reason\":null}]}\n\n"),
		}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
		}
		<-ctx.Done()
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled, Summary: "canceled"}}
	}}
	ctx, cancel := context.WithCancel(context.Background())
	resultCh := make(chan UpstreamResult, 1)
	input := executionForwardInput()
	input.BufferedStream = true
	go func() { resultCh <- NewExecutionForwarder(executor).ForwardStream(ctx, input, httptest.NewRecorder()) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("buffered stream did not start")
	}
	cancel()
	select {
	case result := <-resultCh:
		if result.Stream.EndReason != StreamEndClientCanceled &&
			!errors.Is(result.Err, context.Canceled) && result.ErrorSummary != "canceled" {
			t.Fatalf("cancellation result = %#v", result)
		}
	case <-time.After(time.Second):
		t.Fatal("buffered stream did not stop after cancellation")
	}
	if got := bufferedStreamReservedBytes.Load(); got != 0 {
		t.Fatalf("reserved budget after cancellation = %d", got)
	}
}

func TestBufferedStreamSpoolBudgetIsReusableByNextAttempt(t *testing.T) {
	before := bufferedStreamReservedBytes.Load()
	first, err := newBufferedStreamSpool(1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Write([]byte("first-attempt-spill")); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if got := bufferedStreamReservedBytes.Load(); got != before {
		t.Fatalf("reserved budget after first attempt cleanup = %d, want %d", got, before)
	}
	second, err := newBufferedStreamSpool(32, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if _, err := second.Write(bytes.Repeat([]byte("x"), 32)); err != nil {
		t.Fatalf("next attempt could not reserve released budget: %v", err)
	}
}

type bufferedPartialWriteResponseWriter struct {
	header    http.Header
	committed bool
	writes    int
}

func (writer *bufferedPartialWriteResponseWriter) Header() http.Header              { return writer.header }
func (writer *bufferedPartialWriteResponseWriter) WriteHeader(int)                  { writer.committed = true }
func (writer *bufferedPartialWriteResponseWriter) SetWriteDeadline(time.Time) error { return nil }
func (writer *bufferedPartialWriteResponseWriter) FlushError() error                { return nil }
func (writer *bufferedPartialWriteResponseWriter) Write(data []byte) (int, error) {
	writer.writes++
	if writer.writes > 1 {
		return 1, io.ErrClosedPipe
	}
	return len(data), nil
}

func timeNowForBufferedTest() time.Time { return time.Unix(1_800_000_000, 0) }
