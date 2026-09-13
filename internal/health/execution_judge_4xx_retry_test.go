package health

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

func TestJudgeExecutionProviderClientStatusRequiresReplayProof(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		evidence   *execution.ErrorEvidence
		wantRetry  RetryDirective
		wantEffect Effect
		wantRule   RuleID
	}{
		{
			name:   "unknown 400 does not replay",
			status: http.StatusBadRequest,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, StatusCode: http.StatusBadRequest,
				ReplaySafety: execution.ReplaySafetyUnknown, Summary: "invalid request",
			},
			wantRetry: RetryNone, wantEffect: EffectNone, wantRule: "fallback.http_client_error",
		},
		{
			name:   "explicit rejection retries",
			status: http.StatusBadRequest,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, StatusCode: http.StatusBadRequest,
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing, Summary: "rejected",
			},
			wantRetry: RetryNextCandidate, wantEffect: EffectNone,
			wantRule: "upstream.http_4xx_rejected_before_processing",
		},
		{
			name:   "unknown 401 keeps replay boundary",
			status: http.StatusUnauthorized,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, StatusCode: http.StatusUnauthorized,
				ReplaySafety: execution.ReplaySafetyUnknown, Summary: "authorization failed",
			},
			wantRetry: RetryNone, wantEffect: EffectNone, wantRule: "safety.replay_unknown",
		},
		{
			name:   "explicit invalid credential retries",
			status: http.StatusUnauthorized,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, Hint: execution.FailureHintInvalidCredential,
				ScopeHint: execution.ErrorScopeCredential, StatusCode: http.StatusUnauthorized,
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
			wantRetry: RetryNextCandidate, wantEffect: EffectRecordCredentialFailure,
			wantRule: "auth.invalid_credential",
		},
		{
			name:   "request scoped 429 does not switch candidate",
			status: http.StatusTooManyRequests,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, Hint: execution.FailureHintRateLimited,
				ScopeHint: execution.ErrorScopeRequest, StatusCode: http.StatusTooManyRequests,
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
			wantRetry: RetryNone, wantEffect: EffectNone, wantRule: "rate_limit.scoped",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := JudgeExecution(ExecutionAttempt{
				DispatchState: execution.DispatchMaybeSent,
				StatusCode:    test.status,
				Evidence:      test.evidence,
			}, DecisionContext{Method: http.MethodPost, Operation: execution.OperationImagesGenerate})
			if decision.Retry != test.wantRetry || decision.Effect != test.wantEffect || decision.RuleID != test.wantRule {
				t.Fatalf("JudgeExecution() = %#v, want retry=%q effect=%q rule=%q", decision, test.wantRetry, test.wantEffect, test.wantRule)
			}
			if err := decision.Validate(); err != nil {
				t.Fatalf("Decision.Validate() error = %v", err)
			}
		})
	}
}

func TestJudgeExecutionDoesNotRetryProviderClientStatusAfterCommit(t *testing.T) {
	for name, attempt := range map[string]ExecutionAttempt{
		"payload released": {
			DispatchState: execution.DispatchMaybeSent,
			StatusCode:    http.StatusBadRequest,
			Evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, StatusCode: http.StatusBadRequest,
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
			PayloadReleased: true,
		},
		"downstream committed": {
			DispatchState: execution.DispatchMaybeSent,
			StatusCode:    http.StatusBadRequest,
			Evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, StatusCode: http.StatusBadRequest,
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
			DownstreamCommitted: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			decision := JudgeExecution(attempt, DecisionContext{})
			if decision.Retry != RetryNone || decision.RuleID != "safety.committed" {
				t.Fatalf("decision = %#v", decision)
			}
		})
	}
}

func TestJudgeExecutionDoesNotRetryLocalClientError(t *testing.T) {
	decision := JudgeExecution(ExecutionAttempt{
		DispatchState: execution.DispatchNotSent,
		Evidence: &execution.ErrorEvidence{
			Kind: execution.ErrorKindInvalidRequest, StatusCode: http.StatusBadRequest,
		},
	}, DecisionContext{})
	if decision.Retry != RetryNone || decision.RuleID != "request.invalid_before_dispatch" {
		t.Fatalf("decision = %#v", decision)
	}
}
