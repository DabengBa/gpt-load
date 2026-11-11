package bifrost

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestConvertedHTTP429PreservesReplayPolicy(t *testing.T) {
	t.Parallel()
	for _, body := range []string{
		`{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"slow down"}}`,
		`{"type":"response.failed","error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"HTTP body must not impersonate an SSE event"}}`,
		`{"error":{"type":"usage_limit_reached","code":"usage_limit_reached","message":"quota exhausted"}}`,
		`{"error":{"type":"rate_limit_error","code":"model_at_capacity","message":"capacity exhausted"}}`,
	} {
		t.Run(body, func(t *testing.T) {
			t.Parallel()
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.URL.Path != "/v1/responses" {
					t.Errorf("upstream path=%q, want /v1/responses", r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = io.WriteString(w, body)
			}))
			defer server.Close()
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
			spec := convertedSpec(channel.OpenAI, protocol.Anthropic, execution.OperationChatCompletion,
				"/v1/messages", []byte(`{"model":"client-model","max_tokens":16,"messages":[{"role":"user","content":"hello"}]}`))
			var events []execution.StreamEvent
			result := runtime.ExecuteStream(t.Context(), spec, func(event execution.StreamEvent) error {
				events = append(events, event.Clone())
				return nil
			})
			if err := result.Validate(); err != nil {
				t.Fatal(err)
			}
			if calls.Load() != 1 || len(events) != 0 || result.StatusCode != 429 || result.Error == nil {
				t.Fatalf("calls=%d events=%v result=%+v", calls.Load(), events, result)
			}
			if got, want := result.Error.ReplaySafety == execution.ReplaySafetyRejectedBeforeProcessing, strings.Contains(body, "rate_limit_exceeded"); got != want {
				t.Errorf("HTTP 429 replay marker=%t want=%t: %+v", got, want, result.Error)
			}
			if result.Usage != nil {
				t.Errorf("HTTP error without usage invented usage: %+v", result.Usage)
			}
		})
	}
}
