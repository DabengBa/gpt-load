package bifrost

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/usage"
)

func TestBilledCacheWriteMapping(t *testing.T) {
	for _, unknown := range []bool{false, true} {
		details := &schemas.ChatPromptTokensDetails{CachedReadTokens: 3, CachedWriteTokens: 7}
		want := usage.Tokens{UncachedInput: 11, CacheRead: 3, CacheWrite5M: 5, CacheWrite1H: 2, Output: 2}
		if unknown {
			want.CacheWrite5M, want.CacheWrite1H, want.CacheWriteUnknown = 0, 0, 7
		} else {
			details.CachedWriteTokenDetails = &schemas.ChatCachedWriteTokenDetails{CachedWriteTokens5m: 5, CachedWriteTokens1h: 2}
		}
		err := &schemas.BifrostError{ExtraFields: schemas.BifrostErrorExtraFields{BilledUsage: &schemas.BifrostLLMUsage{PromptTokens: 21, CompletionTokens: 2, TotalTokens: 23, PromptTokensDetails: details}}}
		got := errorUsageEvidence(err, nil)
		if got == nil || got.Normalized.Tokens != want || got.Normalized.State != usage.StatePartial {
			t.Fatalf("unknown=%t usage=%+v want=%+v", unknown, got, want)
		}
	}
}

func TestRealProviderBilledCacheWrites(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 2 {
			w.WriteHeader(500)
			_, _ = io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"second turn failed"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"msg_first","type":"message","role":"assistant","model":"claude-test","content":[{"type":"tool_use","id":"tool_1","name":"bifrost_tool_search","input":{"query":"select:lookup"}}],"stop_reason":"tool_use","usage":{"input_tokens":11,"cache_read_input_tokens":3,"cache_creation_input_tokens":7,"cache_creation":{"ephemeral_5m_input_tokens":5,"ephemeral_1h_input_tokens":2},"output_tokens":2}}`)
	}))
	t.Cleanup(server.Close)
	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, anthropicBaseURL: server.URL})
	spec := convertedSpec(channel.Anthropic, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", []byte(`{"model":"client-model","input":"hello","store":false,"tools":[{"type":"function","name":"lookup","defer_loading":true,"parameters":{"type":"object","properties":{}}}]}`))
	r := runtime.Execute(t.Context(), spec)
	want := usage.Tokens{UncachedInput: 11, CacheRead: 3, CacheWrite5M: 5, CacheWrite1H: 2, Output: 2}
	if calls.Load() != 2 || r.Error == nil || r.Usage == nil || r.Usage.Normalized.Tokens != want || r.Usage.Normalized.State != usage.StatePartial {
		t.Fatalf("calls=%d result=%+v usage=%+v want=%+v", calls.Load(), r, r.Usage, want)
	}
}

// Real SDK + loopback provider, not a pre-populated executor result. Each case
// owns cancellation, server release and runtime cleanup; no sleeps or disk data.
func TestRealSDKCanceledBilledTurn(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, cancelAt := range []int32{1, 2} {
			t.Run(fmt.Sprintf("stream=%t/call=%d", stream, cancelAt), func(t *testing.T) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				second := make(chan struct{})
				release := make(chan struct{})
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if calls.Add(1) == cancelAt {
						close(second)
						select {
						case <-r.Context().Done():
						case <-release:
						}
						return
					}
					body := `{"responseId":"first","modelVersion":"gemini-test","candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"bifrost_tool_search","args":{"query":"select:lookup"}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":12}}`
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprintf(w, "data: %s\n\n", body)
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, body)
					}
				}))
				t.Cleanup(server.Close)
				t.Cleanup(func() { close(release) })
				runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, geminiBaseURL: server.URL + "/v1beta"})
				spec := convertedSpec(channel.Gemini, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", []byte(fmt.Sprintf(`{"model":"client-model","input":"hello","stream":%t,"store":false,"tools":[{"type":"function","name":"lookup","defer_loading":true,"parameters":{"type":"object","properties":{}}}]}`, stream)))
				type outcome struct {
					evidence *execution.UsageEvidence
					failure  *execution.ErrorEvidence
					dispatch execution.DispatchState
				}
				done := make(chan outcome, 1)
				go func() {
					if stream {
						r := runtime.ExecuteStream(ctx, spec, func(execution.StreamEvent) error { return nil })
						done <- outcome{r.Usage, r.Error, r.DispatchState}
					} else {
						r := runtime.Execute(ctx, spec)
						done <- outcome{r.Usage, r.Error, r.DispatchState}
					}
				}()
				select {
				case <-second:
				case <-time.After(3 * time.Second):
					cancel()
					t.Fatal("second SDK provider call not received")
				}
				cancel()
				select {
				case got := <-done:
					want := usage.Result{State: usage.StatePartial, Tokens: usage.Tokens{UncachedInput: 6, CacheRead: 4, Output: 2}}
					usageOK := got.evidence == nil
					if cancelAt == 2 {
						usageOK = got.evidence != nil && got.evidence.Normalized == want
					}
					if calls.Load() != cancelAt || got.failure == nil || got.failure.Kind != execution.ErrorKindCanceled || got.dispatch != execution.DispatchMaybeSent || !usageOK {
						t.Fatalf("calls=%d outcome=%+v usage=%+v want=%+v", calls.Load(), got, got.evidence, want)
					}
				case <-time.After(3 * time.Second):
					t.Fatal("canceled SDK call did not return within bound")
				}
			})
		}
	}
}

func TestCanceledReportedUsageImmediateReplacement(t *testing.T) {
	ctx := schemas.NewBifrostContext(t.Context(), schemas.NoDeadline)
	defer ctx.Cancel()
	if got := canceledReportedUsage(ctx, nil); got != nil {
		t.Fatalf("unreported usage invented: %+v", got)
	}
	for range 2 {
		schemas.SetReportedUsageSnapshot(ctx, &schemas.BifrostLLMUsage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12})
	}
	got := canceledReportedUsage(ctx, nil)
	if got.Normalized.Tokens.UncachedInput != 10 || got.Normalized.Tokens.Output != 2 {
		t.Fatalf("cumulative handoff = %+v, want replacement not sum", got)
	}
}
