package bifrost

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/usage"
)

type delayedBilledTerminalHook struct {
	anthropicUsageStreamHook
	release <-chan struct{}
	cancel  context.CancelFunc
}

func (h delayedBilledTerminalHook) GetName() string { return "test-billed-terminal" }
func (h delayedBilledTerminalHook) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if err != nil && err.ExtraFields.BilledUsage != nil {
		<-h.release
	}
	if h.cancel != nil && resp != nil && resp.NormalizedUsage() != nil {
		h.cancel()
	}
	return resp, err, nil
}

func TestReportedStreamUsageBeforeDownstreamDrop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+`{"responseId":"done","modelVersion":"gemini-test","candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":12}}`+"\n\n")
	}))
	t.Cleanup(server.Close)
	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, geminiBaseURL: server.URL + "/v1beta"})
	spec := convertedSpec(channel.Gemini, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", []byte(`{"model":"client-model","input":"hello","stream":true,"store":false}`))
	dropped := false
	r := runtime.ExecuteStream(t.Context(), spec, func(e execution.StreamEvent) error {
		if bytes.Contains(e.Data, []byte(`"type":"response.completed"`)) {
			dropped = true
			return errors.New("downstream closed after usage observed")
		}
		return nil
	})
	want := usage.Tokens{UncachedInput: 6, CacheRead: 4, Output: 2}
	if !dropped || r.Error == nil || r.Usage == nil || r.Usage.Normalized.Tokens != want {
		t.Fatalf("dropped=%t result=%+v usage=%+v", dropped, r, r.Usage)
	}
}

func TestReportedUsageBeforeDelayedTerminalHook(t *testing.T) {
	for _, stream := range []bool{false, true} {

		for _, successCancel := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%t/success_cancel=%t", stream, successCancel), func(t *testing.T) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				second := make(chan struct{})
				release := make(chan struct{})
				var once sync.Once
				unblock := func() { once.Do(func() { close(release) }) }
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					call := calls.Add(1)
					if call == 2 && !successCancel {
						close(second)
						select {
						case <-r.Context().Done():
						case <-release:
						}
						return
					}
					body := `{"responseId":"first","modelVersion":"gemini-test","candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"bifrost_tool_search","args":{"query":"select:lookup"}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":12}}`
					if call == 2 {
						body = `{"responseId":"second","modelVersion":"gemini-test","candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"cachedContentTokenCount":6,"candidatesTokenCount":3,"totalTokenCount":23}}`
					}
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprintf(w, "data: %s\n\n", body)
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, body)
					}
				}))
				t.Cleanup(server.Close)
				runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, geminiBaseURL: server.URL + "/v1beta"})
				t.Cleanup(unblock)
				hook := delayedBilledTerminalHook{release: release}
				if successCancel {
					hook.cancel = cancel
				}
				spec := convertedSpec(channel.Gemini, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", []byte(fmt.Sprintf(`{"model":"client-model","input":"hello","stream":%t,"store":false,"tools":[{"type":"function","name":"lookup","defer_loading":true,"parameters":{"type":"object","properties":{}}}]}`, stream)))
				if _, failure := runtime.prepare(spec, stream); failure != nil {
					t.Fatal(failure)
				}
				runtime.manager.pool.mu.Lock()
				for _, entry := range runtime.manager.pool.entries {
					core := entry.runtime.(*Runtime).core
					if err := core.ReloadPlugin(hook, []schemas.PluginType{schemas.PluginTypeLLM}); err != nil {
						t.Error(err)
					}
					core.ReorderPlugins([]string{"gpt-load-anthropic-usage", hook.GetName()})
				}
				runtime.manager.pool.mu.Unlock()
				done := make(chan *execution.UsageEvidence, 1)
				go func() {
					if stream {
						r := runtime.ExecuteStream(ctx, spec, func(execution.StreamEvent) error { return nil })
						done <- r.Usage
					} else {
						r := runtime.Execute(ctx, spec)
						done <- r.Usage
					}
				}()
				if !successCancel {
					select {
					case <-second:
					case <-time.After(3 * time.Second):
						t.Fatal("second call not received")
					}
					cancel()
				}
				select {
				case got := <-done:
					want := usage.Tokens{UncachedInput: 6, CacheRead: 4, Output: 2}
					if successCancel {
						want = usage.Tokens{UncachedInput: 20, CacheRead: 10, Output: 5}
					}
					if got == nil || got.Normalized.Tokens != want {
						t.Fatalf("usage=%+v want=%+v with terminal hook still blocked", got, want)
					}
				case <-time.After(3 * time.Second):
					t.Fatal("cancellation waited for terminal hook")
				}
				// Keep the terminal hook unavailable beyond the rejected grace period.
				if !successCancel {
					<-time.After(300 * time.Millisecond)
				}
			})
		}
	}
}
