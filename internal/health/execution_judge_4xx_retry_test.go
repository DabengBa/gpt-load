package health

import (
	"net/http"
	"testing"
	"time"

	"gpt-load/internal/execution"
)

func TestJudgeExecutionRetriesProviderClientStatus(t *testing.T) {
	decision := JudgeExecution(ExecutionAttempt{
		DispatchState: execution.DispatchMaybeSent,
		StatusCode:    http.StatusBadRequest,
		Evidence: &execution.ErrorEvidence{
			Kind: execution.ErrorKindHTTP, StatusCode: http.StatusBadRequest,
		},
	}, DecisionContext{})
	if decision.Category != FailureCategoryClientError || decision.Retry != RetryNextCandidate ||
		decision.Effect != EffectNone || decision.RuleID != "upstream.http_4xx_rejected_before_processing" {
		t.Fatalf("decision = %#v", decision)
	}
}

func TestJudgeExecutionProviderClientStatusReplayContract(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name       string
		status     int
		evidence   *execution.ErrorEvidence
		operation  execution.Operation
		wantScope  execution.ErrorScope
		wantEffect Effect
		wantRetry  RetryDirective
		wantRule   RuleID
		wantUntil  time.Time
	}{
		{
			name:       "bodyless replay-required operation advances",
			status:     http.StatusBadRequest,
			operation:  execution.OperationImagesGenerate,
			wantScope:  execution.ErrorScopeRequest,
			wantEffect: EffectNone,
			wantRetry:  RetryNextCandidate,
			wantRule:   "upstream.http_4xx_rejected_before_processing",
		},
		{
			name:       "bodyless 401 retains credential effect",
			status:     http.StatusUnauthorized,
			operation:  execution.OperationImagesGenerate,
			wantScope:  execution.ErrorScopeCredential,
			wantEffect: EffectRecordCredentialFailure,
			wantRetry:  RetryNextCandidate,
			wantRule:   "auth.invalid_credential",
		},
		{
			name:      "explicit unknown replay safety still advances",
			status:    http.StatusForbidden,
			operation: execution.OperationImagesGenerate,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, StatusCode: http.StatusForbidden,
				ReplaySafety: execution.ReplaySafetyUnknown, Summary: "permission denied",
			},
			wantScope:  execution.ErrorScopeRequest,
			wantEffect: EffectNone,
			wantRetry:  RetryNextCandidate,
			wantRule:   "upstream.http_4xx_rejected_before_processing",
		},
		{
			name:      "401 retains credential effect",
			status:    http.StatusUnauthorized,
			operation: execution.OperationImagesGenerate,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, Hint: execution.FailureHintInvalidCredential,
				ScopeHint: execution.ErrorScopeCredential, StatusCode: http.StatusUnauthorized,
				ReplaySafety: execution.ReplaySafetyUnknown, Summary: "credential rejected",
			},
			wantScope:  execution.ErrorScopeCredential,
			wantEffect: EffectRecordCredentialFailure,
			wantRetry:  RetryNextCandidate,
			wantRule:   "auth.invalid_credential",
		},
		{
			name:      "classified 403 retains credential effect",
			status:    http.StatusForbidden,
			operation: execution.OperationImagesGenerate,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, Hint: execution.FailureHintInvalidCredential,
				ScopeHint: execution.ErrorScopeCredential, StatusCode: http.StatusForbidden,
				ReplaySafety: execution.ReplaySafetyUnknown, Summary: "access denied",
			},
			wantScope:  execution.ErrorScopeCredential,
			wantEffect: EffectRecordCredentialFailure,
			wantRetry:  RetryNextCandidate,
			wantRule:   "auth.invalid_credential",
		},
		{
			name:      "classified model failure retains cooldown",
			status:    http.StatusNotFound,
			operation: execution.OperationChatCompletion,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, Hint: execution.FailureHintModelUnavailable,
				ScopeHint: execution.ErrorScopeModel, StatusCode: http.StatusNotFound,
				ReplaySafety: execution.ReplaySafetyUnknown, Summary: "model unavailable",
			},
			wantScope:  execution.ErrorScopeModel,
			wantEffect: EffectCooldownCredential,
			wantRetry:  RetryNextCandidate,
			wantRule:   "model.unavailable",
			wantUntil:  now.Add(time.Hour),
		},
		{
			name:      "request scoped 429 advances without cooldown",
			status:    http.StatusTooManyRequests,
			operation: execution.OperationImagesGenerate,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, Hint: execution.FailureHintRateLimited,
				ScopeHint: execution.ErrorScopeRequest, StatusCode: http.StatusTooManyRequests,
				ReplaySafety: execution.ReplaySafetyUnknown, Summary: "request rate limited",
			},
			wantScope:  execution.ErrorScopeRequest,
			wantEffect: EffectNone,
			wantRetry:  RetryNextCandidate,
			wantRule:   "rate_limit.scoped",
		},
		{
			name:      "credential scoped 429 preserves retry-after cooldown",
			status:    http.StatusTooManyRequests,
			operation: execution.OperationImagesGenerate,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, Hint: execution.FailureHintRateLimited,
				ScopeHint: execution.ErrorScopeCredential, StatusCode: http.StatusTooManyRequests,
				Code: "rate_limit_exceeded", ReplaySafety: execution.ReplaySafetyUnknown,
				RetryAfter: 45 * time.Second, Summary: "credential rate limited",
			},
			wantScope:  execution.ErrorScopeCredential,
			wantEffect: EffectCooldownCredential,
			wantRetry:  RetryNextCandidate,
			wantRule:   "rate_limit.retry_after",
			wantUntil:  now.Add(45 * time.Second),
		},
		{
			name:       "bodyless non-4xx unknown replay remains blocked",
			status:     http.StatusInternalServerError,
			operation:  execution.OperationImagesGenerate,
			wantScope:  execution.ErrorScopeCredential,
			wantEffect: EffectRecordCredentialFailure,
			wantRetry:  RetryNone,
			wantRule:   "safety.operation_replay_unsafe",
		},
		{
			name:      "non-4xx unknown replay remains blocked",
			status:    http.StatusInternalServerError,
			operation: execution.OperationImagesGenerate,
			evidence: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, StatusCode: http.StatusInternalServerError,
				ReplaySafety: execution.ReplaySafetyUnknown, Summary: "server failed",
			},
			wantScope:  execution.ErrorScopeGroup,
			wantEffect: EffectSkipGroup,
			wantRetry:  RetryNone,
			wantRule:   "upstream.host_error.replay_unsafe",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := JudgeExecution(ExecutionAttempt{
				DispatchState: execution.DispatchMaybeSent,
				StatusCode:    test.status,
				Evidence:      test.evidence,
				Now:           now,
			}, DecisionContext{
				Method:    http.MethodPost,
				Operation: test.operation,
			})
			if decision.Scope != test.wantScope || decision.Effect != test.wantEffect ||
				decision.Retry != test.wantRetry || decision.RuleID != test.wantRule ||
				!decision.CooldownUntil.Equal(test.wantUntil) {
				t.Fatalf("JudgeExecution() = %#v", decision)
			}
			if err := decision.Validate(); err != nil {
				t.Fatalf("Decision.Validate() error = %v", err)
			}
		})
	}
}
func TestJudgeExecutionDoesNotRetryBodylessProviderClientStatusAfterCommit(t *testing.T) {
	for name, attempt := range map[string]ExecutionAttempt{
		"payload released": {
			DispatchState:   execution.DispatchMaybeSent,
			StatusCode:      http.StatusBadRequest,
			PayloadReleased: true,
		},
		"downstream committed": {
			DispatchState:       execution.DispatchMaybeSent,
			StatusCode:          http.StatusBadRequest,
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
