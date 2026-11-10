package gateway

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"gpt-load/internal/dialect"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
)

// Exercise the actual HTTP forwarder, scheduler and request log together. Each
// server is a distinct group, and the unused third server detects budget leaks.
func TestHTTPRetryRotationSharesBudgetAndLogsAttempts(t *testing.T) {
	const empty = `{"id":"empty","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":12,"total_tokens":15}}`
	const success = `{"id":"success","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"recovered"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9}}`
	const hostError = `{"error":{"message":"upstream timed out"}}`
	tests := []struct {
		name       string
		firstHost  bool
		budget     int
		wantStatus int
		wantBody   string
	}{
		{"empty then 524 recovers", false, 3, http.StatusOK, success},
		{"524 then empty recovers", true, 3, http.StatusOK, success},
		{"empty then 524 exhausts shared budget", false, 2, 524, hostError},
		{"524 then empty exhausts shared budget", true, 2, http.StatusOK, empty},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var mu sync.Mutex
			var calls []string
			groups := make([]dialectGatewayGroup, 3)
			for i := range groups {
				index := i
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					mu.Lock()
					calls = append(calls, fmt.Sprintf("%d %s %s", index+1, r.URL.Path, r.Header.Get("Authorization")))
					mu.Unlock()
					w.Header().Set("Content-Type", "application/json")
					if index < 2 && (index == 0) == tt.firstHost {
						w.WriteHeader(524)
						_, _ = io.WriteString(w, hostError)
					} else if index < 2 {
						_, _ = io.WriteString(w, empty)
					} else {
						_, _ = io.WriteString(w, success)
					}
				}))
				t.Cleanup(server.Close)
				groups[i] = dialectGatewayGroup{id: uint(i + 1), name: fmt.Sprintf("group-%d", i+1), upstreamURL: server.URL, apiKeys: []string{fmt.Sprintf("sk-%d", i+1)}}
			}
			sink := &recordingRequestLogSink{}
			engine, _ := newDialectGatewayEngineWithRequestLog(t, protocol.OpenAICompletions, "gpt-4o", dialect.NewSet(dialect.NewOpenAI()), newTestExecutionForwarder(t), config.Settings{state.SettingRetryCount: tt.budget}, sink, groups...)
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"ping"}]}`))
			req.Header.Set("Authorization", "Bearer gl-client")
			rec := httptest.NewRecorder()
			engine.ServeHTTP(rec, req)
			if rec.Code != tt.wantStatus || rec.Body.String() != tt.wantBody {
				t.Fatalf("response = %d %q, want %d %q", rec.Code, rec.Body.String(), tt.wantStatus, tt.wantBody)
			}
			mu.Lock()
			gotCalls := append([]string(nil), calls...)
			mu.Unlock()
			if len(gotCalls) != tt.budget || rec.Header().Get(debugHeaderAttempts) != fmt.Sprint(tt.budget) {
				t.Fatalf("upstream calls = %v, attempts header = %q, want %d", gotCalls, rec.Header().Get(debugHeaderAttempts), tt.budget)
			}
			for i, call := range gotCalls {
				want := fmt.Sprintf("%d /v1/chat/completions Bearer sk-%d", i+1, i+1)
				if call != want {
					t.Fatalf("call %d = %q, want %q", i+1, call, want)
				}
			}
			events := sink.snapshot()
			if len(events) != 1 || len(events[0].Attempts) != tt.budget {
				t.Fatalf("request log events = %#v, want one event with %d attempts", events, tt.budget)
			}
			wantEventStatus, wantErrorCode := telemetry.RequestStatusSuccess, ""
			if tt.budget == 2 {
				wantEventStatus = telemetry.RequestStatusError
				if tt.firstHost {
					wantErrorCode = "upstream_empty_completion"
				} else {
					wantErrorCode = "upstream_host_error"
				}
			}
			if events[0].Status != wantEventStatus || events[0].ErrorCode != wantErrorCode || events[0].StatusCode != tt.wantStatus {
				t.Fatalf("request log = %#v, want status=%s code=%s HTTP=%d", events[0], wantEventStatus, wantErrorCode, tt.wantStatus)
			}
			for i, attempt := range events[0].Attempts {
				if attempt.Sequence != i+1 || attempt.GroupID != uint(i+1) || attempt.CredentialID != uint(i+1) || attempt.WillRetry != (i+1 < tt.budget) {
					t.Fatalf("attempt %d = %#v, unexpected order, identity or retry", i+1, attempt)
				}
				if i == 2 {
					if attempt.StatusCode != http.StatusOK || attempt.FailureCategory != telemetry.FailureCategoryOK {
						t.Fatalf("success attempt = %#v", attempt)
					}
				} else if (i == 0) == tt.firstHost {
					if attempt.StatusCode != 524 || attempt.FailureCategory != telemetry.FailureCategoryUpstreamHost || attempt.RetryDirective != telemetry.RetryNextCandidate || attempt.Effect != telemetry.EffectSkipGroup || attempt.RuleID != "upstream.host_error.retry" || attempt.ErrorCode != "upstream_host_error" {
						t.Fatalf("host attempt = %#v", attempt)
					}
				} else if attempt.StatusCode != http.StatusOK || attempt.FailureCategory != telemetry.FailureCategoryAmbiguous || attempt.RetryDirective != telemetry.RetryNextCandidate || attempt.Effect != telemetry.EffectNone || attempt.RuleID != "upstream.empty_completion" || attempt.ErrorCode != "upstream_empty_completion" {
					t.Fatalf("empty attempt = %#v", attempt)
				}
			}
		})
	}
}

func TestHTTP524WithNoBackupReturnsLastResponse(t *testing.T) {
	const body = `{"error":{"message":"upstream timed out"}}`
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(524)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(upstream.Close)
	sink := &recordingRequestLogSink{}
	engine, _ := newDialectGatewayEngineWithRequestLog(t, protocol.OpenAICompletions, "gpt-4o", dialect.NewSet(dialect.NewOpenAI()), newTestExecutionForwarder(t), config.Settings{state.SettingRetryCount: 3}, sink,
		dialectGatewayGroup{id: 1, name: "only", upstreamURL: upstream.URL, apiKeys: []string{"sk-only"}})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	req.Header.Set("Authorization", "Bearer gl-client")
	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, req)
	events := sink.snapshot()
	if calls.Load() != 1 || rec.Code != 524 || rec.Body.String() != body || len(events) != 1 || len(events[0].Attempts) != 1 || events[0].Attempts[0].WillRetry || events[0].Attempts[0].Effect != telemetry.EffectSkipGroup || events[0].Status != telemetry.RequestStatusError {
		t.Fatalf("response = %d %q, calls = %d, events = %#v; want last 524 without actual retry", rec.Code, rec.Body.String(), calls.Load(), events)
	}
}
