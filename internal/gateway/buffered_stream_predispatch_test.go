//go:build !windows

package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

// bufferedStreamCommitWriter records the first downstream HTTP commit of one
// buffered forward so a test can assert that a local pre-dispatch failure never
// produced an HTTP 200 heartbeat.
type bufferedStreamCommitWriter struct {
	*httptest.ResponseRecorder
	committed chan struct{}
	once      sync.Once
	mu        sync.Mutex
	headers   int
	writes    int
}

func newBufferedStreamCommitWriter() *bufferedStreamCommitWriter {
	return &bufferedStreamCommitWriter{
		ResponseRecorder: httptest.NewRecorder(),
		committed:        make(chan struct{}),
	}
}

func (writer *bufferedStreamCommitWriter) markCommitted() {
	writer.once.Do(func() { close(writer.committed) })
}

func (writer *bufferedStreamCommitWriter) WriteHeader(status int) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	writer.headers++
	writer.ResponseRecorder.WriteHeader(status)
}

func (writer *bufferedStreamCommitWriter) Write(data []byte) (int, error) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	writer.writes++
	written, err := writer.ResponseRecorder.Write(data)
	writer.markCommitted()
	return written, err
}

func (writer *bufferedStreamCommitWriter) committedCounts() (int, int) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	return writer.headers, writer.writes
}

// body returns a synchronized snapshot; the forwarder keeps writing heartbeats
// while the test observes the visible bytes.
func (writer *bufferedStreamCommitWriter) body() string {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	return writer.ResponseRecorder.Body.String()
}

// TestBufferedStreamLocalPreDispatchFailureDoesNotCommitHTTP is the U005 red
// proof: a local pre-dispatch rejection must keep the original HTTP status and
// error code instead of committing an HTTP 200 heartbeat first.
func TestBufferedStreamLocalPreDispatchFailureDoesNotCommitHTTP(t *testing.T) {
	executor := fakeExecutionExecutor{stream: func(
		context.Context,
		execution.AttemptSpec,
		execution.StreamSink,
	) execution.StreamResult {
		evidence := execution.ErrorEvidence{
			Kind:       execution.ErrorKindConversionUnsupported,
			OriginHint: execution.ErrorOriginInternal,
			ScopeHint:  execution.ErrorScopeGroup,
			Code:       execution.ErrorCodeCriticalSemanticLoss,
			Summary:    "protocol conversion cannot preserve Anthropic zero-output semantics",
		}
		return execution.StreamResult{DispatchState: execution.DispatchNotSent, Error: &evidence}
	}}
	input := executionForwardInput()
	input.BufferedStream = true
	input.ClientProtocol = protocol.Anthropic
	writer := newBufferedStreamCommitWriter()

	result := NewExecutionForwarder(executor).ForwardStream(context.Background(), input, writer)

	headers, writes := writer.committedCounts()
	if headers != 0 || writes != 0 {
		t.Fatalf("local pre-dispatch failure committed HTTP: headers=%d writes=%d body=%q",
			headers, writes, writer.body())
	}
	if result.Committed || result.HTTPCommitted || result.ClientVisibleBytes != 0 {
		t.Fatalf("local pre-dispatch failure reported a commit: %#v", result)
	}
	if !result.BufferedStream {
		t.Fatal("buffered attempt lost its buffered contract marker")
	}
	if result.DispatchState != execution.DispatchNotSent {
		t.Fatalf("dispatch state = %q, want not_sent", result.DispatchState)
	}
	if !isConversionUnsupportedResult(result) {
		t.Fatalf("conversion evidence was lost: %#v", result.ExecutionError)
	}
	if got := transportReason(result); got != reasonProtocolConversionUnsupported {
		t.Fatalf("original HTTP reason = %#v, want %#v", got, reasonProtocolConversionUnsupported)
	}
}

// TestBufferedStreamDispatchedSlowStreamHeartbeatsBeforeRelease guards the
// opposite contract: provider dispatch alone must commit the heartbeat, before
// the provider produces its first payload byte and long before release.
func TestBufferedStreamDispatchedSlowStreamHeartbeatsBeforeRelease(t *testing.T) {
	release := make(chan struct{})
	dispatched := make(chan struct{})
	provideData := make(chan struct{})
	executor := fakeExecutionExecutor{stream: func(
		ctx context.Context,
		_ execution.AttemptSpec,
		sink execution.StreamSink,
	) execution.StreamResult {
		if err := sink(execution.StreamEvent{
			Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK,
			Header: http.Header{"Content-Type": {"text/event-stream"}},
		}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
		}
		// The provider acknowledged with response metadata but its first payload
		// byte is still pending.
		close(dispatched)
		select {
		case <-provideData:
		case <-ctx.Done():
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled, Summary: "canceled"}}
		}
		if err := sink(execution.StreamEvent{
			Sequence: 2, Kind: execution.StreamEventData,
			Data: []byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"},\"finish_reason\":null}]}\n\n"),
		}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
		}
		select {
		case <-release:
		case <-ctx.Done():
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled, Summary: "canceled"}}
		}
		for _, data := range [][]byte{
			[]byte("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"),
			[]byte("data: [DONE]\n\n"),
		} {
			if err := sink(execution.StreamEvent{Sequence: 3, Kind: execution.StreamEventData, Data: data}); err != nil {
				return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
			}
		}
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	input := executionForwardInput()
	input.BufferedStream = true
	input.ClientProtocol = protocol.OpenAICompletions
	forwarder := NewExecutionForwarder(executor)
	// No heartbeat ticker may be needed: dispatch alone must commit the
	// heartbeat, otherwise the buffered contract would depend on a slow timer.
	forwarder.heartbeatInterval = time.Hour
	writer := newBufferedStreamCommitWriter()
	resultCh := make(chan UpstreamResult, 1)
	go func() { resultCh <- forwarder.ForwardStream(context.Background(), input, writer) }()

	waitForStreamSignal(t, dispatched, "buffered execution dispatch")
	select {
	case <-writer.committed:
	case <-time.After(time.Second):
		t.Fatal("dispatched slow stream did not commit a heartbeat")
	}
	select {
	case result := <-resultCh:
		t.Fatalf("stream terminated before release: %#v", result)
	default:
	}
	if body := writer.body(); body != bufferedStreamHeartbeat {
		t.Fatalf("visible bytes before the first payload byte = %q, want a heartbeat only", body)
	}
	close(provideData)
	if body := writer.body(); body != bufferedStreamHeartbeat {
		t.Fatalf("provider payload was visible before release = %q", body)
	}
	close(release)
	result := <-resultCh
	if !result.PayloadReleased || !result.HTTPCommitted {
		t.Fatalf("released result = %#v", result)
	}
	if body := writer.body(); !strings.Contains(body, `"content":"first"`) || !strings.Contains(body, "[DONE]") {
		t.Fatalf("released body = %q", body)
	}
}
