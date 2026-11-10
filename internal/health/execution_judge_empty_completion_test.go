package health

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

func TestJudgeExecutionEmptyCompletionEvidence(t *testing.T) {
	decision := JudgeExecution(ExecutionAttempt{
		DispatchState:   execution.DispatchMaybeSent,
		StatusCode:      http.StatusOK,
		EmptyCompletion: true,
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
		decision.Retry != RetryNextCandidate ||
		decision.Effect != EffectNone ||
		decision.RuleID != RuleID("upstream.empty_completion") {
		t.Fatalf("JudgeExecution() = %#v, want ambiguous/upstream/request/next-candidate/no-effect/upstream.empty_completion", decision)
	}
	if err := decision.Validate(); err != nil {
		t.Fatalf("JudgeExecution().Validate() = %v", err)
	}
}

func TestJudgeExecutionEmptyCompletionKeepsStrictOperationReplayBoundary(t *testing.T) {
	decision := JudgeExecution(ExecutionAttempt{
		DispatchState:   execution.DispatchMaybeSent,
		StatusCode:      http.StatusOK,
		EmptyCompletion: true,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindProvider,
			OriginHint: execution.ErrorOriginUpstream,
			ScopeHint:  execution.ErrorScopeRequest,
			StatusCode: http.StatusOK,
			Code:       "upstream_empty_completion",
		},
	}, DecisionContext{
		Method:    http.MethodPost,
		Operation: execution.OperationEmbeddingsCreate,
	})

	if decision.Retry != RetryNone || decision.RuleID != RuleID("safety.operation_replay_unsafe") {
		t.Fatalf("JudgeExecution() = %#v, want strict operation replay to remain blocked", decision)
	}
}
