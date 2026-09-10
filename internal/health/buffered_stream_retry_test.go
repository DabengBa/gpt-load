package health

import (
	"net/http"
	"testing"
	"time"

	"gpt-load/internal/execution"
)

func TestBufferedStreamRetryUsesPayloadReleaseGate(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	base := ExecutionAttempt{
		DispatchState:       execution.DispatchMaybeSent,
		StatusCode:          http.StatusOK,
		DownstreamCommitted: true,
		HTTPCommitted:       true,
		ClientVisibleBytes:  16, // gateway heartbeat only
		BufferedStream:      true,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindTimeout,
			OriginHint: execution.ErrorOriginUpstream,
			ScopeHint:  execution.ErrorScopeRequest,
			Code:       "upstream_stream_idle_timeout",
			Summary:    "upstream stream idle timeout",
		},
		Now: now,
	}

	decision := JudgeExecution(base, DecisionContext{
		Method:                   http.MethodPost,
		Operation:                execution.OperationChatCompletion,
		BufferedReplayEligible:   true,
		DefaultRateLimitCooldown: time.Minute,
	})
	if decision.Retry != RetryNextCandidate || decision.Effect != EffectSkipGroup {
		t.Fatalf("heartbeat-only buffered attempt decision = %#v, want controlled retry", decision)
	}

	base.PayloadReleased = true
	decision = JudgeExecution(base, DecisionContext{
		Method:                 http.MethodPost,
		Operation:              execution.OperationChatCompletion,
		BufferedReplayEligible: true,
	})
	if decision.Retry != RetryNone {
		t.Fatalf("released buffered attempt decision = %#v, must not retry", decision)
	}
}

func TestBufferedStreamHealthDoesNotTreatHTTPCommitAsPayloadRelease(t *testing.T) {
	attempt := ExecutionAttempt{
		DispatchState:       execution.DispatchMaybeSent,
		StatusCode:          http.StatusOK,
		DownstreamCommitted: true,
		HTTPCommitted:       true,
		ClientVisibleBytes:  16,
		BufferedStream:      true,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindProvider,
			OriginHint: execution.ErrorOriginUpstream,
			ScopeHint:  execution.ErrorScopeRequest,
			Code:       "upstream_protocol_error",
			Summary:    "upstream protocol error",
		},
	}
	decision := JudgeExecution(attempt, DecisionContext{
		Method:                 http.MethodPost,
		Operation:              execution.OperationChatCompletion,
		BufferedReplayEligible: true,
	})
	if decision.Retry != RetryNextCandidate {
		t.Fatalf("HTTP-committed buffered attempt decision = %#v, want retry before release", decision)
	}

	attempt.BufferedStream = false
	decision = JudgeExecution(attempt, DecisionContext{
		Method:                 http.MethodPost,
		Operation:              execution.OperationChatCompletion,
		BufferedReplayEligible: true,
	})
	if decision.Retry != RetryNone {
		t.Fatalf("legacy committed attempt decision = %#v, must remain non-retryable", decision)
	}
}
