package health

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

// An upstream 503 without classifiable evidence used to terminate the request:
// the judge fell back to missing evidence with RetryNone and the gateway never
// switched candidate. The operator asked for candidate switching on that path.
func TestJudgeExecutionSwitchesCandidateWithoutEvidenceOnRetryableStatus(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		dispatch   execution.DispatchState
		wantRetry  RetryDirective
		wantEffect Effect
		wantScope  execution.ErrorScope
		wantRule   RuleID
	}{
		{
			name:       "service unavailable",
			status:     http.StatusServiceUnavailable,
			dispatch:   execution.DispatchMaybeSent,
			wantRetry:  RetryNextCandidate,
			wantEffect: EffectRecordCredentialFailure,
			wantScope:  execution.ErrorScopeCredential,
			wantRule:   "fallback.missing_evidence_retry",
		},
		{
			name:       "gateway timeout",
			status:     http.StatusGatewayTimeout,
			dispatch:   execution.DispatchMaybeSent,
			wantRetry:  RetryNextCandidate,
			wantEffect: EffectRecordCredentialFailure,
			wantScope:  execution.ErrorScopeCredential,
			wantRule:   "fallback.missing_evidence_retry",
		},
		{
			name:       "rate limited",
			status:     http.StatusTooManyRequests,
			dispatch:   execution.DispatchMaybeSent,
			wantRetry:  RetryNextCandidate,
			wantEffect: EffectRecordCredentialFailure,
			wantScope:  execution.ErrorScopeCredential,
			wantRule:   "fallback.missing_evidence_retry",
		},
		{
			name:       "client error stays final",
			status:     http.StatusNotFound,
			dispatch:   execution.DispatchMaybeSent,
			wantRetry:  RetryNone,
			wantEffect: EffectNone,
			wantRule:   "fallback.missing_evidence",
		},
		{
			name:       "unclassified status stays final",
			status:     0,
			dispatch:   execution.DispatchMaybeSent,
			wantRetry:  RetryNone,
			wantEffect: EffectNone,
			wantRule:   "fallback.missing_evidence",
		},
		{
			name:       "no response and nothing dispatched stays final",
			status:     0,
			dispatch:   execution.DispatchNotSent,
			wantRetry:  RetryNone,
			wantEffect: EffectNone,
			wantRule:   "fallback.missing_evidence",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := JudgeExecution(ExecutionAttempt{
				DispatchState: test.dispatch,
				StatusCode:    test.status,
			}, DecisionContext{})
			if decision.Retry != test.wantRetry || decision.RuleID != test.wantRule {
				t.Fatalf("JudgeExecution() = %#v, want retry %v rule %q", decision, test.wantRetry, test.wantRule)
			}
			if decision.Effect != test.wantEffect || decision.Scope != test.wantScope {
				t.Fatalf(
					"JudgeExecution() = %#v, want effect %v scope %v",
					decision, test.wantEffect, test.wantScope,
				)
			}
		})
	}
}

// A buffered request whose heartbeat is committed but whose payload is not
// released is the one committed state where the retry may still travel.
func TestJudgeExecutionKeepsMissingEvidenceRetryForUnreleasedBufferedStream(t *testing.T) {
	buffered := JudgeExecution(ExecutionAttempt{
		DispatchState:      execution.DispatchMaybeSent,
		StatusCode:         http.StatusServiceUnavailable,
		BufferedStream:     true,
		HTTPCommitted:      true,
		ClientVisibleBytes: 14,
	}, DecisionContext{})
	if buffered.Retry != RetryNextCandidate || buffered.RuleID != "fallback.missing_evidence_retry" {
		t.Fatalf("buffered heartbeat attempt = %#v, want missing-evidence retry", buffered)
	}
	if buffered.Effect != EffectRecordCredentialFailure {
		t.Fatalf("buffered heartbeat attempt = %#v, want the credential failure counted", buffered)
	}

	tests := []struct {
		name    string
		attempt ExecutionAttempt
	}{
		{
			name: "released payload",
			attempt: ExecutionAttempt{
				DispatchState:       execution.DispatchMaybeSent,
				StatusCode:          http.StatusServiceUnavailable,
				BufferedStream:      true,
				HTTPCommitted:       true,
				PayloadReleased:     true,
				DownstreamCommitted: true,
				ClientVisibleBytes:  4096,
			},
		},
		{
			name: "committed live stream",
			attempt: ExecutionAttempt{
				DispatchState:       execution.DispatchMaybeSent,
				StatusCode:          http.StatusServiceUnavailable,
				DownstreamCommitted: true,
				ClientVisibleBytes:  4096,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := JudgeExecution(test.attempt, DecisionContext{})
			if decision.Retry != RetryNone {
				t.Fatalf(
					"JudgeExecution() = %#v, want no retry and no credential failure once the client already saw output",
					decision,
				)
				if decision.Effect != EffectNone {
					t.Fatalf("JudgeExecution() = %#v, want effect %v", decision, EffectNone)
				}
			}
		})
	}
}
