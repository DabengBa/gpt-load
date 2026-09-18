package health

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

func TestJudgeExecutionEmptyCompletionEvidence(t *testing.T) {
	decision := JudgeExecution(ExecutionAttempt{
		DispatchState: execution.DispatchMaybeSent,
		StatusCode:    http.StatusOK,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindProvider,
			OriginHint: execution.ErrorOriginUpstream,
			ScopeHint:  execution.ErrorScopeRequest,
			StatusCode: http.StatusOK,
			Code:       "upstream_empty_completion",
		},
	}, DecisionContext{})

	if decision.Category != FailureCategoryAmbiguous ||
		decision.Origin != execution.ErrorOriginUpstream ||
		decision.Scope != execution.ErrorScopeRequest ||
		decision.Retry != RetryNone ||
		decision.Effect != EffectNone ||
		decision.RuleID != RuleID("upstream.empty_completion") {
		t.Fatalf("JudgeExecution() = %#v, want ambiguous/upstream/request/no-retry/no-effect/upstream.empty_completion", decision)
	}
	if err := decision.Validate(); err != nil {
		t.Fatalf("JudgeExecution().Validate() = %v", err)
	}
}
