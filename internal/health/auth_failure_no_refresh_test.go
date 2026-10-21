package health

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

// Ordinary API credentials cannot be refreshed by GPT-Load, so a provider auth
// failure must never claim a credential refresh: it either retries another
// candidate or records the credential failure.
func TestOrdinaryAPIAuthFailureNeverClaimsCredentialRefresh(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name     string
		hint     execution.FailureHint
		category FailureCategory
		retry    RetryDirective
		effect   Effect
		rule     RuleID
	}{
		{name: "refresh required", hint: execution.FailureHintRefreshRequired,
			category: FailureCategoryAuthenticationRequired, retry: RetryNextCandidate,
			effect: EffectNone, rule: "auth.refresh_unavailable"},
		{name: "reauthorization required", hint: execution.FailureHintReauthorizationRequired,
			category: FailureCategoryAuthenticationRequired, retry: RetryNextCandidate,
			effect: EffectNone, rule: "auth.reauthorization_required"},
		{name: "invalid credential", hint: execution.FailureHintInvalidCredential,
			category: FailureCategoryInvalidKey, retry: RetryNextCandidate,
			effect: EffectRecordCredentialFailure, rule: "auth.invalid_credential"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			replaySafety := execution.ReplaySafetyRejectedBeforeProcessing

			decision := JudgeExecution(ExecutionAttempt{
				DispatchState: execution.DispatchMaybeSent,
				StatusCode:    http.StatusUnauthorized,
				Evidence: &execution.ErrorEvidence{
					Kind:         execution.ErrorKindHTTP,
					Hint:         test.hint,
					ScopeHint:    execution.ErrorScopeCredential,
					StatusCode:   http.StatusUnauthorized,
					ReplaySafety: replaySafety,
					Summary:      "upstream rejected the credential",
				},
			}, DecisionContext{Method: http.MethodPost})
			if decision.Category != test.category || decision.Retry != test.retry ||
				decision.Effect != test.effect || decision.RuleID != test.rule {
				t.Fatalf("decision = %#v, want category=%s retry=%s effect=%s rule=%s",
					decision, test.category, test.retry, test.effect, test.rule)
			}

		})
	}
}
