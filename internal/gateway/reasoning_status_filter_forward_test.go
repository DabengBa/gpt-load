package gateway

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/execution"
)

func TestExecutionForwarderReasoningStatus(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("enabled=%t/stream=%t", enabled, stream), func(t *testing.T) {
				input := responsesExecutionForwardInput()
				input.Group.ResponsesReasoningStatusFilterEnabled = enabled
				input.Request.Body = fmt.Appendf(nil, `{ "model":"public", "stream":%t, "input":%s }`, stream, reasoningStatusInput)
				original := bytes.Clone(input.Request.Body)
				want := string(original)
				if enabled {
					want = fmt.Sprintf(`{"model":"public","stream":%t,"input":%s}`, stream, reasoningStatusFilteredInput)
				}
				var forwarded []byte
				response := []byte(`{"id":"resp_filter","object":"response","status":"completed","output":[]}`)
				executor := fakeExecutionExecutor{
					unary: func(_ context.Context, spec execution.AttemptSpec) execution.AttemptResult {
						forwarded = bytes.Clone(spec.Body)
						return execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
							StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: response}
					},
					stream: func(_ context.Context, spec execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
						forwarded = bytes.Clone(spec.Body)
						if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady,
							StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
							t.Fatal(err)
						}
						if err := sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData,
							Data: []byte("data: {\"type\":\"response.completed\",\"response\":" + string(response) + "}\n\n")}); err != nil {
							t.Fatal(err)
						}
						return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
					},
				}
				forwarder := NewExecutionForwarder(executor)
				if stream {
					result := forwarder.ForwardStream(t.Context(), input, httptest.NewRecorder())
					if result.Err != nil || !result.Committed || result.ExecutionError != nil {
						t.Fatalf("ForwardStream() = %#v", result)
					}
				} else {
					result := forwarder.Forward(t.Context(), input)
					if result.Err != nil || result.StatusCode != http.StatusOK {
						t.Fatalf("Forward() = %#v", result)
					}
				}
				assertJSONEqual(t, string(forwarded), want)
				if !enabled && !bytes.Equal(forwarded, original) {
					t.Fatalf("disabled filter rewrote bytes: %s", forwarded)
				}
				if !bytes.Equal(input.Request.Body, original) {
					t.Fatal("filter mutated original request")
				}
			})
		}
	}
}
