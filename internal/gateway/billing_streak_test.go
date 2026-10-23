package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func insufficientBalanceResult() UpstreamResult {
	body := []byte(`{"error":{"code":"insufficient_balance","message":"Insufficient balance"}}`)
	return UpstreamResult{
		DispatchState:      execution.DispatchMaybeSent,
		ResponseStarted:    true,
		StatusCode:         http.StatusPaymentRequired,
		Header:             make(http.Header),
		Body:               body,
		ClassificationBody: body,
		RequestWritten:     true,
	}
}

func serveBillingRequest(t *testing.T, engine *gin.Engine) {
	t.Helper()
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"gpt-4o"}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	engine.ServeHTTP(httptest.NewRecorder(), request)
}

// 冷却时长复用系统熔断释放参数 blacklist_release_seconds。
func TestHandlerBillingCooldownUsesPublishedReleaseSeconds(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		insufficientBalanceResult(),
		{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
		insufficientBalanceResult(),
		{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
	}}
	handler, manager, registry := newHandlerForTest(t, forwarder, "sk-first", "sk-second")
	handler.now = func() time.Time { return now }
	publishHandlerPolicySettings(t, handler, manager, 2, config.Settings{
		state.SettingBlacklistReleaseSeconds: json.Number("240"),
	}, nil)
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	serveBillingRequest(t, engine)
	serveBillingRequest(t, engine)

	until, cooling := registry.CredentialCooldownUntil(1)
	if !cooling || !until.Equal(now.Add(240*time.Second)) {
		t.Fatalf("credential cooldown = %v/%t, want published 240s", until, cooling)
	}
}

// 余额不足要连续两次才冷却：第一次只换候选保住本请求，凭据继续可用；
// 中间出现成功则重新计数。
func TestHandlerBillingCoolsCredentialOnlyAfterConsecutiveFailures(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		insufficientBalanceResult(),
		{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
		insufficientBalanceResult(),
		{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
	}}
	engine, handler, registry, _ := newStatsHandlerTestRuntime(t, forwarder, "sk-first", "sk-second")
	handler.now = func() time.Time { return now }

	// 请求 1：sk-first 报余额不足 → 换候选成功，凭据不冷却。
	serveBillingRequest(t, engine)
	if until, cooling := registry.CredentialCooldownUntil(1); cooling && until.After(now) {
		t.Fatalf("credential cooldown after first billing = %v, want none", until)
	}

	// 请求 2：sk-first 再报余额不足 → 连续两次，冷却一个释放窗口。
	serveBillingRequest(t, engine)
	until, cooling := registry.CredentialCooldownUntil(1)
	if !cooling || !until.Equal(now.Add(3600*time.Second)) {
		t.Fatalf("credential cooldown after second billing = %v/%t, want default 3600s", until, cooling)
	}
}

// 结果入口统一：非 billing 的任何决策都打断连续段，包括不产生任何效果的
// 上游 400（EffectNone）与模型 scope 的提前返回路径。
func TestHandlerBillingStreakResetsOnNonBillingDecision(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)
	clientError := UpstreamResult{
		DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
		StatusCode:         http.StatusBadRequest,
		Header:             make(http.Header),
		Body:               []byte(`{"error":{"type":"invalid_request_error","message":"bad input"}}`),
		ClassificationBody: []byte(`{"error":{"type":"invalid_request_error","message":"bad input"}}`),
		RequestWritten:     true,
	}
	modelUnavailable := UpstreamResult{
		DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
		StatusCode:         http.StatusNotFound,
		Header:             make(http.Header),
		Body:               []byte(`{"error":{"code":"model_not_found","message":"model not found"}}`),
		ClassificationBody: []byte(`{"error":{"code":"model_not_found","message":"model not found"}}`),
		RequestWritten:     true,
	}
	tests := []struct {
		name     string
		interrup UpstreamResult
	}{
		{name: "effect-less client error", interrup: clientError},
		{name: "model scoped failure", interrup: modelUnavailable},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := &scriptedForwarder{results: []UpstreamResult{
				insufficientBalanceResult(),
				{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
				test.interrup,
				{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
				insufficientBalanceResult(),
				{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
			}}
			engine, handler, registry, _ := newStatsHandlerTestRuntime(t, forwarder, "sk-first", "sk-second")
			handler.now = func() time.Time { return now }

			// billing → 打断结果 → billing：只有一次连续 billing，不触发冷却。
			serveBillingRequest(t, engine)
			serveBillingRequest(t, engine)
			serveBillingRequest(t, engine)

			if until, cooling := registry.CredentialCooldownUntil(1); cooling && until.After(now) {
				t.Fatalf("credential cooldown = %v, want none after an interrupted streak", until)
			}
			if got := registry.Snapshot()[0].BillingStreak; got != 1 {
				t.Fatalf("billing streak = %d, want 1 after billing→interruption→billing", got)
			}
		})
	}
}

// 触发后重置：冷却落地后连续段清零，窗口结束后的下一次 billing 重新从 1 计数，
// 单次不足阈值不再延长冷却。
func TestHandlerBillingStreakRestartsAfterCooldown(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		insufficientBalanceResult(),
		{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
		insufficientBalanceResult(),
		{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true},
	}}
	engine, handler, registry, _ := newStatsHandlerTestRuntime(t, forwarder, "sk-first", "sk-second")
	handler.now = func() time.Time { return now }

	serveBillingRequest(t, engine)
	serveBillingRequest(t, engine)
	if got := registry.Snapshot()[0].BillingStreak; got != 0 {
		t.Fatalf("billing streak after cooldown = %d, want reset 0", got)
	}
	until, cooling := registry.CredentialCooldownUntil(1)
	if !cooling || !until.Equal(now.Add(3600*time.Second)) {
		t.Fatalf("credential cooldown = %v/%t, want default 3600s", until, cooling)
	}
}

// 计数与冷却都必须基于产出决策的凭据版本：版本已变时不推进连续段、不写冷却。
func TestHandlerBillingStreakIgnoresStaleCredentialVersion(t *testing.T) {
	now := time.Date(2026, time.September, 25, 8, 0, 0, 0, time.UTC)
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 1, GroupID: 1, Version: 3, IdentityGeneration: 1,
		Fingerprint: "credential-1", EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatal(err)
	}
	handler := &Handler{
		registry:  registry,
		stats:     health.NewStatsStore(),
		mutations: health.NewMutationCoordinator(),
	}
	billing := health.Decision{
		Category:      health.FailureCategoryBilling,
		Scope:         execution.ErrorScopeCredential,
		Effect:        health.EffectCooldownCredential,
		CooldownUntil: now.Add(time.Hour),
		RuleID:        "billing.insufficient_balance",
	}
	group := state.GroupView{ID: 1}

	for range billingStreakThreshold {
		handler.applyGroupDecisionEffectForEntry(group, 1, 2, "", billing, http.StatusPaymentRequired, now)
	}
	view := registry.Snapshot()[0]
	if view.BillingStreak != 0 || view.CooldownUntil.After(now) {
		t.Fatalf("stale version mutated credential = %#v", view)
	}
	for range billingStreakThreshold {
		handler.applyGroupDecisionEffectForEntry(group, 1, 3, "", billing, http.StatusPaymentRequired, now)
	}
	view = registry.Snapshot()[0]
	if !view.CooldownUntil.Equal(now.Add(time.Hour)) || view.BillingStreak != 0 {
		t.Fatalf("current version credential = %#v, want cooldown and cleared streak", view)
	}
}
