package health

import (
	"net/http"
	"testing"
	"time"

	"gpt-load/internal/execution"
	"gpt-load/internal/state"
)

// 余额类失败是凭据级资损信号：换候选保住本请求。冷却时长由共享系统参数
// blacklist_release_seconds 决定，judge 只负责重试指令与效果选择。
func TestJudgeExecutionBillingDecisionUsesConfiguredCooldown(t *testing.T) {
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
	}, DecisionContext{
		DefaultRateLimitCooldown: time.Minute,
		BillingCooldown:          45 * time.Minute,
	})

	want := Decision{
		Category:      FailureCategoryBilling,
		Origin:        execution.ErrorOriginUpstream,
		Scope:         execution.ErrorScopeCredential,
		Retry:         RetryNextCandidate,
		Effect:        EffectCooldownCredential,
		CooldownUntil: now.Add(45 * time.Minute),
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

// 已承诺下游的 billing 证据仍保冷却效果：scope 是凭据级，committed 不会抹掉它，
// 且冷却时长同样来自共享配置。
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
	}, DecisionContext{
		DefaultRateLimitCooldown: time.Minute,
		BillingCooldown:          10 * time.Minute,
	})

	if decision.Category != FailureCategoryBilling ||
		decision.Effect != EffectCooldownCredential ||
		!decision.CooldownUntil.Equal(now.Add(10*time.Minute)) {
		t.Fatalf("JudgeExecution() = %#v, want billing cooldown", decision)
	}
}

// 未注入共享冷却时，billing 仍必须给出合法决策：Decision.Validate 要求
// cooldown 效果必须带截止时间，取值使用系统熔断释放参数的出厂默认值
// （blacklist_release_seconds），不退到限流冷却。
func TestJudgeExecutionBillingWithoutConfiguredCooldownFallsBack(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)

	decision := JudgeExecution(ExecutionAttempt{
		DispatchState: execution.DispatchMaybeSent,
		StatusCode:    http.StatusPaymentRequired,
		Now:           now,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			Hint:       execution.FailureHintInsufficientBalance,
			ScopeHint:  execution.ErrorScopeCredential,
			StatusCode: http.StatusPaymentRequired,
		},
	}, DecisionContext{DefaultRateLimitCooldown: 7 * time.Minute})

	if err := decision.Validate(); err != nil {
		t.Fatalf("JudgeExecution() returned invalid decision %#v: %v", decision, err)
	}
	want := time.Duration(state.DefaultRuntimeSettings().BlacklistReleaseSeconds) * time.Second
	if want != 3600*time.Second {
		t.Fatalf("shipped blacklist release default = %v, want 3600s", want)
	}
	if !decision.CooldownUntil.Equal(now.Add(want)) {
		t.Fatalf("cooldown = %v, want shipped blacklist release default %v", decision.CooldownUntil, want)
	}
}
