package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

// retry_count is the total attempt budget for one request: 1 means a single
// attempt, 2 means the request may still switch candidate once.
func TestRetryAttemptLimitTreatsRetryCountAsTotalAttemptBudget(t *testing.T) {
	tests := []struct {
		retryCount int
		want       int
	}{
		{retryCount: 0, want: 1},
		{retryCount: 1, want: 1},
		{retryCount: 2, want: 2},
		{retryCount: 3, want: 3},
		{retryCount: 100, want: 100},
	}

	for _, test := range tests {
		got := retryAttemptLimit(state.GroupView{RetryCount: test.retryCount})
		if got != test.want {
			t.Fatalf("retryAttemptLimit(retry_count=%d) = %d, want %d", test.retryCount, got, test.want)
		}
	}
}

type missingEvidenceStreamForwarder struct {
	calls  int
	groups []uint
}

func (*missingEvidenceStreamForwarder) Forward(context.Context, ForwardInput) UpstreamResult {
	return UpstreamResult{Err: errors.New("unexpected unary forward")}
}

// ForwardStream reproduces the recorded upstream answer of the incident: 503
// with no classifiable evidence while the buffered heartbeat already committed
// HTTP and no provider payload was released.
func (forwarder *missingEvidenceStreamForwarder) ForwardStream(_ context.Context, input ForwardInput, _ http.ResponseWriter) UpstreamResult {
	forwarder.calls++
	forwarder.groups = append(forwarder.groups, input.Group.ID)
	return UpstreamResult{
		StatusCode:         http.StatusServiceUnavailable,
		RequestWritten:     true,
		DispatchState:      execution.DispatchMaybeSent,
		Committed:          true,
		HTTPCommitted:      true,
		BufferedStream:     true,
		ClientVisibleBytes: int64(len(bufferedStreamHeartbeat)),
	}
}

func serveMissingEvidenceStreamRequest(t *testing.T, engine http.Handler) {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"claude-client","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"stream":true}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	engine.ServeHTTP(httptest.NewRecorder(), request)
}

func TestHandlerBufferedStreamRetriesMissingEvidenceWithinAttemptBudget(t *testing.T) {
	tests := []struct {
		name       string
		retryCount string
		wantCalls  int
	}{
		{name: "single attempt budget stops after the first attempt", retryCount: "1", wantCalls: 1},
		{name: "two attempt budget switches candidate", retryCount: "2", wantCalls: 2},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := &missingEvidenceStreamForwarder{}
			engine, _ := newConvertedFallbackHandlerTestRuntime(t, forwarder, config.Settings{
				state.SettingBufferedStream: true,
				state.SettingRetryCount:     json.Number(test.retryCount),
			})
			serveMissingEvidenceStreamRequest(t, engine)

			if forwarder.calls != test.wantCalls {
				t.Fatalf(
					"buffered attempts = %d, want %d with retry_count=%s",
					forwarder.calls, test.wantCalls, test.retryCount,
				)
			}
		})
	}
}

// A candidate that keeps answering with a transient status but no classifiable
// evidence must still be counted as a credential failure: without the count,
// blacklist_threshold never takes it out of rotation and every later request
// pays for the same broken candidate first.
func TestHandlerBlacklistsCandidateAfterMissingEvidenceFailures(t *testing.T) {
	forwarder := &missingEvidenceStreamForwarder{}
	engine, _ := newConvertedFallbackHandlerTestRuntime(t, forwarder, config.Settings{
		state.SettingBufferedStream:     true,
		state.SettingRetryCount:         json.Number("1"),
		state.SettingBlacklistThreshold: json.Number("1"),
	})

	serveMissingEvidenceStreamRequest(t, engine)
	if len(forwarder.groups) != 1 {
		t.Fatalf("first request attempts = %v, want one attempt", forwarder.groups)
	}
	repeated := forwarder.groups[0]

	serveMissingEvidenceStreamRequest(t, engine)
	if len(forwarder.groups) != 2 {
		t.Fatalf("attempts = %v, want one attempt per request", forwarder.groups)
	}
	if forwarder.groups[1] == repeated {
		t.Fatalf("group %d stayed in rotation after its credential reached the blacklist threshold", repeated)
	}
}
