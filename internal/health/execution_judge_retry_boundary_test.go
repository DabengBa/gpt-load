package health

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

// Candidate rotation must retain operation replay proof and payload boundaries,
// even when buffered heartbeats have already committed HTTP headers.
func TestJudgeExecutionRetryPreservesOperationAndPayloadBoundaries(t *testing.T) {
	for _, operation := range []execution.Operation{
		execution.OperationChatCompletion,
		execution.OperationResponsesCreate,
		execution.OperationImagesGenerate,
		execution.OperationImagesEdit,
		execution.OperationEmbeddingsCreate,
		execution.OperationRerank,
	} {
		t.Run(string(operation), func(t *testing.T) {
			for _, failure := range []struct {
				name     string
				status   int
				code     string
				category FailureCategory
				scope    execution.ErrorScope
				effect   Effect
				rule     RuleID
			}{
				{"empty completion", http.StatusOK, "upstream_empty_completion", FailureCategoryAmbiguous, execution.ErrorScopeRequest, EffectNone, "upstream.empty_completion"},
				{"host unavailable", http.StatusServiceUnavailable, "upstream_error", FailureCategoryUpstreamHostError, execution.ErrorScopeGroup, EffectSkipGroup, "upstream.host_error.retry"},
				{"host timeout", http.StatusGatewayTimeout, "upstream_error", FailureCategoryUpstreamHostError, execution.ErrorScopeGroup, EffectSkipGroup, "upstream.host_error.retry"},
			} {
				t.Run(failure.name, func(t *testing.T) {
					for _, safety := range []execution.ReplaySafety{"", execution.ReplaySafetyUnknown, execution.ReplaySafetyRejectedBeforeProcessing} {
						t.Run("safety="+string(safety), func(t *testing.T) {
							for _, window := range []string{"uncommitted", "buffered heartbeat", "released payload", "committed live stream"} {
								t.Run(window, func(t *testing.T) {
									attempt := ExecutionAttempt{
										DispatchState:   execution.DispatchMaybeSent,
										StatusCode:      failure.status,
										EmptyCompletion: failure.code == "upstream_empty_completion",
										Evidence: &execution.ErrorEvidence{
											Kind: execution.ErrorKindProvider, OriginHint: execution.ErrorOriginUpstream,
											ScopeHint: failure.scope, StatusCode: failure.status, Code: failure.code,
											ReplaySafety: safety,
										},
									}
									switch window {
									case "buffered heartbeat":
										attempt.BufferedStream, attempt.HTTPCommitted = true, true
										attempt.ClientVisibleBytes = 14
									case "released payload":
										attempt = withReleasedPayload(attempt)
									case "committed live stream":
										attempt.HTTPCommitted, attempt.DownstreamCommitted = true, true
									}
									wantRetry, wantRule, wantEffect := RetryNextCandidate, failure.rule, failure.effect
									if failure.status != http.StatusOK && safety == execution.ReplaySafetyRejectedBeforeProcessing {
										wantRule = "upstream.host_error.rejected_before_processing"
									}
									strict := operation == execution.OperationImagesGenerate || operation == execution.OperationImagesEdit ||
										operation == execution.OperationEmbeddingsCreate || operation == execution.OperationRerank
									if strict && safety != execution.ReplaySafetyRejectedBeforeProcessing {
										wantRetry, wantRule = RetryNone, "safety.operation_replay_unsafe"
									}
									if failure.status != http.StatusOK && window == "buffered heartbeat" && (!strict || safety == execution.ReplaySafetyRejectedBeforeProcessing) {
										wantRule = "buffered_stream.retry_before_release_upstream_status"
									}
									if window == "released payload" || window == "committed live stream" {
										wantRetry, wantEffect = RetryNone, EffectNone
									}
									got := JudgeExecution(attempt, DecisionContext{
										Method: http.MethodPost, Operation: operation, BufferedReplayEligible: true,
									})
									if got.Retry != wantRetry || got.Effect != wantEffect || got.Category != failure.category || got.Origin != execution.ErrorOriginUpstream {
										t.Fatalf("JudgeExecution() = %#v, want category %v retry %v effect %v", got, failure.category, wantRetry, wantEffect)
									}
									if window == "uncommitted" || window == "buffered heartbeat" {
										if got.RuleID != wantRule || got.Scope != failure.scope {
											t.Fatalf("JudgeExecution() = %#v, want rule %q scope %q", got, wantRule, failure.scope)
										}
									}
									if err := got.Validate(); err != nil {
										t.Fatalf("Decision.Validate() = %v", err)
									}
								})
							}
						})
					}
				})
			}
		})
	}
}
