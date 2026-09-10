package health

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

// 上游以可重试状态（408/429/5xx）失败、且下游只收到心跳时，网关必须换候选。
// 证据一旦在 buffered 路径上被保留，upstream.host_error.replay_unsafe 的默认值就会
// 把这类失败重新变回终局，所以重放许可必须由 buffered 合同单独给出。
func TestJudgeExecutionSwitchesCandidateOnTransientUpstreamStatusBeforeRelease(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		scope      execution.ErrorScope
		wantRule   RuleID
		wantEffect Effect
	}{
		{
			name:       "service unavailable skips the group",
			status:     http.StatusServiceUnavailable,
			scope:      execution.ErrorScopeGroup,
			wantRule:   "buffered_stream.retry_before_release_upstream_status",
			wantEffect: EffectSkipGroup,
		},
		{
			name:       "gateway timeout skips the group",
			status:     http.StatusGatewayTimeout,
			scope:      execution.ErrorScopeGroup,
			wantRule:   "buffered_stream.retry_before_release_upstream_status",
			wantEffect: EffectSkipGroup,
		},
		{
			name:       "request timeout keeps the client error effect",
			status:     http.StatusRequestTimeout,
			scope:      execution.ErrorScopeRequest,
			wantRule:   "buffered_stream.retry_before_release_upstream_status",
			wantEffect: EffectNone,
		},
		{
			name:       "rate limit keeps its credential cooldown",
			status:     http.StatusTooManyRequests,
			scope:      execution.ErrorScopeCredential,
			wantRule:   "rate_limit.credential.default_cooldown",
			wantEffect: EffectCooldownCredential,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := JudgeExecution(bufferedUpstreamStatusAttempt(test.status, test.scope), DecisionContext{
				BufferedReplayEligible: true,
			})
			if decision.Retry != RetryNextCandidate || decision.RuleID != test.wantRule {
				t.Fatalf("JudgeExecution() = %#v, want retry %v rule %q", decision, RetryNextCandidate, test.wantRule)
			}
			if decision.Effect != test.wantEffect {
				t.Fatalf("JudgeExecution() = %#v, want effect %v", decision, test.wantEffect)
			}
		})
	}
}

// buffered 合同只在「心跳已提交、payload 未释放」这一种已提交状态下放宽重放；
// 未分类的客户端错误、已释放 payload、已提交的实时流和未标记 buffered 的尝试都保持终局。
func TestJudgeExecutionKeepsTransientUpstreamStatusFinalOutsideBufferedReleaseWindow(t *testing.T) {
	tests := []struct {
		name    string
		attempt ExecutionAttempt
	}{
		{
			name:    "released payload",
			attempt: withReleasedPayload(bufferedUpstreamStatusAttempt(http.StatusServiceUnavailable, execution.ErrorScopeGroup)),
		},
		{
			name: "live stream already committed",
			attempt: func() ExecutionAttempt {
				attempt := bufferedUpstreamStatusAttempt(http.StatusServiceUnavailable, execution.ErrorScopeGroup)
				attempt.BufferedStream = false
				attempt.DownstreamCommitted = true
				return attempt
			}(),
		},
		{
			name: "client error stays final",
			attempt: func() ExecutionAttempt {
				attempt := bufferedUpstreamStatusAttempt(http.StatusNotFound, execution.ErrorScopeRequest)
				attempt.Evidence.Hint = ""
				return attempt
			}(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := JudgeExecution(test.attempt, DecisionContext{BufferedReplayEligible: true})
			if decision.Retry != RetryNone {
				t.Fatalf("JudgeExecution() = %#v, want no retry", decision)
			}
		})
	}
}

func bufferedUpstreamStatusAttempt(status int, scope execution.ErrorScope) ExecutionAttempt {
	return ExecutionAttempt{
		DispatchState:      execution.DispatchMaybeSent,
		StatusCode:         status,
		BufferedStream:     true,
		HTTPCommitted:      true,
		ClientVisibleBytes: 14,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			OriginHint: execution.ErrorOriginUpstream,
			ScopeHint:  scope,
			StatusCode: status,
			Code:       "upstream_error",
			Summary:    "upstream returned an error status",
		},
	}
}

func withReleasedPayload(attempt ExecutionAttempt) ExecutionAttempt {
	attempt.PayloadReleased = true
	attempt.DownstreamCommitted = true
	attempt.ClientVisibleBytes = 4096
	return attempt
}
