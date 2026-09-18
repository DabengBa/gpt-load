package health

import (
	"net/http"
	"testing"
	"time"

	"gpt-load/internal/execution"
)

// TestJudgeExecutionContentFilterEvidence proves R1's decision mapping: the
// gateway's "upstream_content_filter" evidence is a client error with no retry
// and no runtime effect, so a content-filtered response is recorded as a failed
// request instead of a successful one.
func TestJudgeExecutionContentFilterEvidence(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, time.September, 18, 7, 0, 0, 0, time.UTC)
	tests := []struct {
		name                string
		downstreamCommitted bool
	}{
		{name: "before commit"},
		{name: "after commit", downstreamCommitted: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			decision := JudgeExecution(ExecutionAttempt{
				DispatchState: execution.DispatchMaybeSent,
				StatusCode:    http.StatusOK,
				Evidence: &execution.ErrorEvidence{
					Kind:       execution.ErrorKindProvider,
					OriginHint: execution.ErrorOriginUpstream,
					ScopeHint:  execution.ErrorScopeRequest,
					StatusCode: http.StatusOK,
					Code:       "upstream_content_filter",
					Summary:    "upstream response was blocked by content filter and contained no assistant content",
				},
				DownstreamCommitted: test.downstreamCommitted,
				Now:                 now,
			}, DecisionContext{DefaultRateLimitCooldown: time.Minute})

			if decision.Category != FailureCategoryClientError ||
				decision.Origin != execution.ErrorOriginUpstream ||
				decision.Scope != execution.ErrorScopeRequest ||
				decision.Retry != RetryNone ||
				decision.Effect != EffectNone ||
				decision.RuleID != RuleID("content_filter.no_content") {
				t.Fatalf("JudgeExecution() = %#v, want client_error with rule content_filter.no_content", decision)
			}
			if decision.Category == FailureCategoryOK || decision.LegacyAction() != ActionTerminate {
				t.Fatalf("JudgeExecution() recorded the filtered response as a success: %#v", decision)
			}
			if err := decision.Validate(); err != nil {
				t.Fatalf("JudgeExecution().Validate() = %v", err)
			}
		})
	}
}
