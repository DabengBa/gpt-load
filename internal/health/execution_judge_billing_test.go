package health

import (
	"net/http"
	"testing"
	"time"

	"gpt-load/internal/execution"
)

// 余额类失败是凭据级资损信号：换候选保住本请求，同时把该凭据冷却 24h——
// 第一次命中即停止烧请求，充值后无需人工解锁即可自愈。
func TestJudgeExecutionBillingDecisionCoolsCredentialForADay(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)

	decision := JudgeExecution(ExecutionAttempt{
		DispatchState: execution.DispatchMaybeSent,
		StatusCode:    http.StatusPaymentRequired,
		Now:           now,
		Evidence: &execution.ErrorEvidence{
			Kind:         execution.ErrorKindHTTP,
			Hint:         execution.FailureHintInsufficientBalance,
			OriginHint:   execution.ErrorOriginUpstream,
			ScopeHint:    execution.ErrorScopeCredential,
			StatusCode:   http.StatusPaymentRequired,
			Code:         "insufficient_balance",
			Summary:      "Insufficient Balance",
			ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
		},
	}, DecisionContext{DefaultRateLimitCooldown: time.Minute})

	want := Decision{
		Category:      FailureCategoryBilling,
		Origin:        execution.ErrorOriginUpstream,
		Scope:         execution.ErrorScopeCredential,
		Retry:         RetryNextCandidate,
		Effect:        EffectCooldownCredential,
		CooldownUntil: now.Add(24 * time.Hour),
		RuleID:        RuleID("billing.insufficient_balance"),
	}
	if decision != want {
		t.Fatalf("JudgeExecution() = %#v, want %#v", decision, want)
	}
}

func TestJudgeExecutionBillingClassificationSignals(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)
	tests := []struct {
		name     string
		attempt  ExecutionAttempt
		category FailureCategory
	}{
		{
			name: "insufficient_quota marker beats 429 status",
			attempt: ExecutionAttempt{
				DispatchState: execution.DispatchMaybeSent,
				StatusCode:    http.StatusTooManyRequests,
				Now:           now,
				Evidence: &execution.ErrorEvidence{
					Kind:       execution.ErrorKindHTTP,
					StatusCode: http.StatusTooManyRequests,
					Code:       "insufficient_quota",
					Summary:    "You exceeded your current quota",
				},
			},
			category: FailureCategoryBilling,
		},
		{
			name: "credit balance message on 400",
			attempt: ExecutionAttempt{
				DispatchState: execution.DispatchMaybeSent,
				StatusCode:    http.StatusBadRequest,
				Now:           now,
				Evidence: &execution.ErrorEvidence{
					Kind:       execution.ErrorKindHTTP,
					StatusCode: http.StatusBadRequest,
					Type:       "invalid_request_error",
					Summary:    "Your credit balance is too low to access the API",
				},
			},
			category: FailureCategoryBilling,
		},
		{
			name: "chinese balance message on 403",
			attempt: ExecutionAttempt{
				DispatchState: execution.DispatchMaybeSent,
				StatusCode:    http.StatusForbidden,
				Now:           now,
				Evidence: &execution.ErrorEvidence{
					Kind:       execution.ErrorKindHTTP,
					StatusCode: http.StatusForbidden,
					Summary:    "当前账户余额不足，请充值后重试",
				},
			},
			category: FailureCategoryBilling,
		},
		{
			name: "self-healing quota stays rate limited",
			attempt: ExecutionAttempt{
				DispatchState: execution.DispatchMaybeSent,
				StatusCode:    http.StatusTooManyRequests,
				Now:           now,
				Evidence: &execution.ErrorEvidence{
					Kind:       execution.ErrorKindHTTP,
					StatusCode: http.StatusTooManyRequests,
					Code:       "resource_exhausted",
					Summary:    "quota_exceeded for requests per minute",
				},
			},
			category: FailureCategoryRateLimited,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := JudgeExecution(test.attempt, DecisionContext{DefaultRateLimitCooldown: time.Minute})
			if decision.Category != test.category {
				t.Fatalf("JudgeExecution().Category = %q, want %q", decision.Category, test.category)
			}
		})
	}
}

// 已承诺下游的 billing 证据仍保冷却效果：scope 是凭据级，committed 不会抹掉它。
func TestJudgeExecutionBillingCooldownSurvivesCommitted(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)

	decision := JudgeExecution(ExecutionAttempt{
		DispatchState:       execution.DispatchMaybeSent,
		StatusCode:          http.StatusPaymentRequired,
		DownstreamCommitted: true,
		Now:                 now,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			Hint:       execution.FailureHintInsufficientBalance,
			OriginHint: execution.ErrorOriginUpstream,
			ScopeHint:  execution.ErrorScopeCredential,
			StatusCode: http.StatusPaymentRequired,
			Summary:    "Insufficient Balance",
		},
	}, DecisionContext{DefaultRateLimitCooldown: time.Minute})

	if decision.Category != FailureCategoryBilling ||
		decision.Effect != EffectCooldownCredential ||
		!decision.CooldownUntil.Equal(now.Add(24*time.Hour)) {
		t.Fatalf("JudgeExecution() = %#v, want billing cooldown", decision)
	}
}
