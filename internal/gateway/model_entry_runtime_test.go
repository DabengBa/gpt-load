package gateway

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/state"
)

func newEntryRuntimeTestHandler(t *testing.T) (*Handler, *state.CredentialRegistry) {
	t.Helper()
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 1, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "credential-1",
		EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	handler := &Handler{
		registry:  registry,
		stats:     health.NewStatsStore(),
		mutations: health.NewMutationCoordinator(),
	}
	return handler, registry
}

func modelScopeDecision(effect health.Effect, until time.Time) health.Decision {
	return health.Decision{
		Category:      health.FailureCategoryModelUnavailable,
		Origin:        execution.ErrorOriginUpstream,
		Scope:         execution.ErrorScopeModel,
		Retry:         health.RetryNextCandidate,
		Effect:        effect,
		CooldownUntil: until,
		RuleID:        "model.unavailable",
	}
}

func TestHandlerModelScopeCooldownWritesRouteEntryOnly(t *testing.T) {
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	until := now.Add(time.Hour)

	handler.applyGroupDecisionEffectForEntry(
		state.GroupView{ID: 1, BlacklistThreshold: 3}, 1, 0, "e000000000001",
		modelScopeDecision(health.EffectCooldownCredential, until),
		http.StatusNotFound, now,
	)

	view, ok := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now)
	if !ok || view.RuntimeState(now) != state.EntryRuntimeCooldown || !view.CooldownUntil.Equal(until) {
		t.Fatalf("entry runtime = %#v exists=%t, want cooldown until %v", view, ok, until)
	}
	if _, exists := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000002"}, now); exists {
		t.Fatal("sibling route entry gained runtime state")
	}
	if cooldownUntil, exists := registry.CredentialCooldownUntil(1); exists && cooldownUntil.After(now) {
		t.Fatalf("credential cooldown = %v, want untouched", cooldownUntil)
	}
}

// Design §4.1: counting is driven by the decision features, not by Effect.
// The decision below carries EffectRecordCredentialFailure yet still counts
// toward the entry breaker because its features match a model-level failure.
func TestHandlerModelScopeFailuresBlacklistRouteEntryOnly(t *testing.T) {
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	group := state.GroupView{
		ID:                  1,
		ModelBreakerByEntry: map[uint]map[string]*state.EntryCircuitBreaker{},
	}
	group.ModelBreakerByEntry[1] = map[string]*state.EntryCircuitBreaker{
		"e000000000001": {BlacklistThreshold: intPtrForEntryTest(2)},
	}

	for range 2 {
		handler.applyGroupDecisionEffectForEntry(
			group, 1, 0, "e000000000001",
			modelScopeDecision(health.EffectRecordCredentialFailure, time.Time{}),
			http.StatusNotFound, now,
		)
	}

	view, ok := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now)
	if !ok || view.RuntimeState(now) != state.EntryRuntimeBlacklisted || view.FailureCount != 2 {
		t.Fatalf("entry runtime = %#v exists=%t, want blacklisted with 2 failures", view, ok)
	}
	if view.CooldownUntil != (time.Time{}) {
		t.Fatalf("entry cooldown = %v, want zero: cooldown fallback does not apply", view.CooldownUntil)
	}
	candidates := registry.CollectCredentialCandidates([]uint{1}, func(uint) bool { return false }, now)
	if len(candidates) != 1 || candidates[0].ID != 1 {
		t.Fatalf("credential candidates = %#v, want credential 1 still available", candidates)
	}
}

// Compatibility C1 and design §3.2: an entry without an explicit per-entry
// blacklist threshold never counts model-level failures and does not inherit
// the group-level BlacklistThreshold.
func TestHandlerModelScopeFailureWithoutBreakerThresholdDoesNotCount(t *testing.T) {
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)

	for range 5 {
		handler.applyGroupDecisionEffectForEntry(
			state.GroupView{ID: 1, BlacklistThreshold: 1}, 1, 0, "e000000000001",
			modelScopeDecision(health.EffectCooldownCredential, now.Add(time.Hour)),
			http.StatusNotFound, now,
		)
	}

	view, ok := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now)
	if !ok {
		t.Fatal("entry runtime missing, want cooldown-only state")
	}
	if view.FailureCount != 0 {
		t.Fatalf("entry failure count = %d, want 0 without configured threshold", view.FailureCount)
	}
	if view.Blacklisted {
		t.Fatal("entry blacklisted without a configured per-entry threshold")
	}
	if view.RuntimeState(now) != state.EntryRuntimeCooldown {
		t.Fatalf("entry runtime state = %q, want decision-default cooldown", view.RuntimeState(now))
	}
}

func TestHandlerCredentialScopeEffectStillWritesCredential(t *testing.T) {
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	until := now.Add(30 * time.Minute)

	handler.applyGroupDecisionEffectForEntry(
		state.GroupView{ID: 1, BlacklistThreshold: 3}, 1, 0, "e000000000001",
		health.Decision{
			Category:      health.FailureCategoryRateLimited,
			Origin:        execution.ErrorOriginUpstream,
			Scope:         execution.ErrorScopeCredential,
			Retry:         health.RetryNextCandidate,
			Effect:        health.EffectCooldownCredential,
			CooldownUntil: until,
			RuleID:        "rate_limit.credential.default_cooldown",
		},
		http.StatusTooManyRequests, now,
	)

	if cooldownUntil, cooling := registry.CredentialCooldownUntil(1); !cooling || !cooldownUntil.Equal(until) {
		t.Fatalf("credential cooldown = %v/%t, want %v", cooldownUntil, cooling, until)
	}
	if _, exists := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now); exists {
		t.Fatal("credential-scope effect created route entry state")
	}
}

func TestHandlerModelScopeWithoutEntryIDFallsBackToCredential(t *testing.T) {
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	until := now.Add(time.Hour)

	handler.applyGroupDecisionEffectForEntry(
		state.GroupView{ID: 1, BlacklistThreshold: 3}, 1, 0, "",
		modelScopeDecision(health.EffectCooldownCredential, until),
		http.StatusNotFound, now,
	)

	if cooldownUntil, cooling := registry.CredentialCooldownUntil(1); !cooling || !cooldownUntil.Equal(until) {
		t.Fatalf("credential cooldown = %v/%t, want legacy credential path", cooldownUntil, cooling)
	}
}

func TestHandlerModelUnavailableCooldownIsolatesRouteEntryEndToEnd(t *testing.T) {
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	forwarder := &scriptedForwarder{results: []UpstreamResult{{
		StatusCode:         http.StatusNotFound,
		Header:             make(http.Header),
		Body:               []byte(`{"error":{"code":"model_not_found"}}`),
		ClassificationBody: []byte(`{"error":{"code":"model_not_found"}}`),
		RequestWritten:     true,
	}}}
	engine, handler, registry, _ := newStatsHandlerTestRuntime(t, forwarder, "sk-one")
	handler.now = func() time.Time { return now }

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"gpt-4o"}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	engine.ServeHTTP(httptest.NewRecorder(), request)

	until := now.Add(time.Hour)
	view, ok := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now)
	if !ok || view.RuntimeState(now) != state.EntryRuntimeCooldown || !view.CooldownUntil.Equal(until) {
		t.Fatalf("entry runtime = %#v exists=%t, want cooldown until %v", view, ok, until)
	}
	if cooldownUntil, exists := registry.CredentialCooldownUntil(1); exists && cooldownUntil.After(now) {
		t.Fatalf("credential cooldown = %v, want untouched", cooldownUntil)
	}
}

func intPtrForEntryTest(value int) *int {
	return &value
}
