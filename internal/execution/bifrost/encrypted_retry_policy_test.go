package bifrost

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestEncryptedReasoningRejectionDoesNotRetryInsideAttempt(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "unary", true: "stream"}[stream], func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"The encrypted content for item rs_1 could not be verified. Reason: Encrypted content could not be decrypted or parsed."}}`)
			}))
			defer server.Close()
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, anthropicBaseURL: server.URL})
			spec := convertedSpec(channel.Anthropic, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", []byte(`{"model":"client-model","input":[{"role":"user","content":"hello"},{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"planning"}],"encrypted_content":"cipher"}]}`))
			spec.UpstreamModel = "claude-sonnet-4-6"
			if stream {
				result := runtime.ExecuteStream(t.Context(), spec, func(execution.StreamEvent) error { return nil })
				if err := result.Validate(); err != nil {
					t.Fatalf("invalid result: %v", err)
				}
				if calls.Load() != 1 || result.StatusCode != http.StatusBadRequest {
					t.Fatalf("calls=%d status=%d, want one provider request and original 400", calls.Load(), result.StatusCode)
				}
			} else {
				result := runtime.Execute(t.Context(), spec)
				if err := result.Validate(); err != nil {
					t.Fatalf("invalid result: %v", err)
				}
				if calls.Load() != 1 || result.StatusCode != http.StatusBadRequest {
					t.Fatalf("calls=%d status=%d, want one provider request and original 400", calls.Load(), result.StatusCode)
				}
			}
		})
	}
}
