package gateway

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestBufferedStreamOutputKeepsHTTPCommitAndPayloadReleaseIndependent(t *testing.T) {
	var output BufferedStreamOutput
	output.markHTTPCommitted()
	output.markVisibleBytes(int64(len(bufferedStreamHeartbeat)))
	if !output.canRetry() || !output.HTTPCommitted || output.PayloadReleased {
		t.Fatalf("heartbeat output = %#v, want retryable committed state", output)
	}
	if err := output.markPayloadReleased(); err != nil {
		t.Fatal(err)
	}
	if output.canRetry() || !output.PayloadReleased {
		t.Fatalf("released output = %#v, must prohibit retry", output)
	}
	if err := output.markPayloadReleased(); err == nil {
		t.Fatal("second payload release unexpectedly succeeded")
	}
}

func TestBufferedStreamForwarderHidesPayloadUntilValidatedEOF(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
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
			t.Fatalf("ready sink: %v", err)
		}
		if err := sink(execution.StreamEvent{
			Sequence: 2, Kind: execution.StreamEventData,
			Data: []byte("data: {\"id\":\"chat_1\",\"choices\":[{\"delta\":{\"content\":\"secret\"},\"finish_reason\":\"stop\"}]}\n\n"),
		}); err != nil {
			t.Fatalf("data sink: %v", err)
		}
		close(started)
		<-release
		if err := sink(execution.StreamEvent{
			Sequence: 3, Kind: execution.StreamEventData, Data: []byte("data: [DONE]\n\n"),
		}); err != nil {
			t.Fatalf("done sink: %v", err)
		}
		return execution.StreamResult{
			DispatchState:   execution.DispatchMaybeSent,
			ResponseStarted: true, StatusCode: http.StatusOK,
			Header: http.Header{"Content-Type": {"text/event-stream"}},
		}
	}}
	input := executionForwardInput()
	input.BufferedStream = true
	input.ClientProtocol = protocol.OpenAICompletions
	recorder := httptest.NewRecorder()
	finished := make(chan UpstreamResult, 1)
	go func() {
		finished <- NewExecutionForwarder(executor).ForwardStream(context.Background(), input, recorder)
	}()
	<-started
	if got := recorder.Body.String(); got != bufferedStreamHeartbeat {
		t.Fatalf("visible body before validated EOF = %q, want heartbeat only", got)
	}
	close(release)
	result := <-finished
	if result.Err != nil || !result.HTTPCommitted || !result.PayloadReleased ||
		result.ClientVisibleBytes <= int64(len(bufferedStreamHeartbeat)) ||
		!bytes.Contains(recorder.Body.Bytes(), []byte("secret")) {
		t.Fatalf("buffered result = %#v, body = %q", result, recorder.Body.String())
	}
}
