package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/state"
)

const breakerTestEntryID = "e000000000001"

func breakerIntPtr(value int) *int {
	return &value
}

// modelFailureDecision builds the Decision shapes produced by the judge for
// each of the four upstream model failure rule families (design §4.1).
func modelFailureDecision(ruleID string, effect health.Effect, cooldownUntil time.Time) health.Decision {
	return health.Decision{
		Category:      health.FailureCategoryModelUnavailable,
		Origin:        execution.ErrorOriginUpstream,
		Scope:         execution.ErrorScopeModel,
		Retry:         health.RetryNextCandidate,
		Effect:        effect,
		CooldownUntil: cooldownUntil,
		RuleID:        health.RuleID(ruleID),
	}
}

func breakerGroupView(threshold, cooldownSeconds *int) state.GroupView {
	return state.GroupView{
		ID: 1,
		ModelBreakerByEntry: map[uint]map[string]*state.EntryCircuitBreaker{
			1: {breakerTestEntryID: {
				BlacklistThreshold: threshold,
				CooldownSeconds:    cooldownSeconds,
			}},
		},
	}
}

func breakerEntryView(registry *state.CredentialRegistry) state.EntryRuntimeView {
	view, _ := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: breakerTestEntryID}, time.Time{})
	return view
}

// All four judge model-failure rule families count toward the per-entry
// breaker although their Effect differs (EffectCooldownCredential vs
// EffectNone): counting follows the decision features, not Effect.
func TestModelEntryBreakerCountsAllFourModelFailureRuleFamilies(t *testing.T) {
	families := []struct {
		ruleID string
		effect health.Effect
	}{
		{"model.unavailable", health.EffectCooldownCredential},
		{"images.model_unavailable", health.EffectNone},
		{"embeddings.model_unavailable", health.EffectNone},
		{"candidate.unavailable", health.EffectNone},
	}
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	for _, family := range families {
		t.Run(family.ruleID, func(t *testing.T) {
			handler, registry := newEntryRuntimeTestHandler(t)
			group := breakerGroupView(breakerIntPtr(2), nil)

			for range 2 {
				handler.applyGroupDecisionEffectForEntry(
					group, 1, 0, breakerTestEntryID,
					modelFailureDecision(family.ruleID, family.effect, time.Time{}),
					http.StatusNotFound, now,
				)
			}

			view := breakerEntryView(registry)
			if view.FailureCount != 2 || view.RuntimeState(now) != state.EntryRuntimeBlacklisted {
				t.Fatalf("entry runtime = %#v, want blacklisted after 2 counted failures", view)
			}
		})
	}
}

func TestModelEntryBreakerCountsBelowThresholdWithoutBlacklisting(t *testing.T) {
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	group := breakerGroupView(breakerIntPtr(3), nil)

	handler.applyGroupDecisionEffectForEntry(
		group, 1, 0, breakerTestEntryID,
		modelFailureDecision("model.unavailable", health.EffectCooldownCredential, time.Time{}),
		http.StatusNotFound, now,
	)

	view := breakerEntryView(registry)
	if view.FailureCount != 1 || view.RuntimeState(now) != state.EntryRuntimeAvailable || view.Blacklisted {
		t.Fatalf("entry runtime = %#v, want available with 1 counted failure", view)
	}
}

// safety.replay_unknown is the single excluded rule: unknown replay safety
// means it is ambiguous whether the request reached the upstream, so the
// failure neither counts nor cools the entry (design §4.1, compatibility C1).
func TestModelEntryBreakerExcludesReplayUnknownRule(t *testing.T) {
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	group := breakerGroupView(breakerIntPtr(1), breakerIntPtr(0))

	handler.applyGroupDecisionEffectForEntry(
		group, 1, 0, breakerTestEntryID,
		health.Decision{
			Category: health.FailureCategoryModelUnavailable,
			Origin:   execution.ErrorOriginUpstream,
			Scope:    execution.ErrorScopeModel,
			Retry:    health.RetryNone,
			Effect:   health.EffectNone,
			RuleID:   "safety.replay_unknown",
		},
		http.StatusNotFound, now,
	)

	if _, exists := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: breakerTestEntryID}, now); exists {
		t.Fatal("replay_unknown decision created entry runtime state")
	}
	if cooldownUntil, cooling := registry.CredentialCooldownUntil(1); cooling && cooldownUntil.After(now) {
		t.Fatalf("credential cooldown = %v, want untouched", cooldownUntil)
	}
}

func TestModelEntryBreakerCooldownSecondsOverridesDecisionDefault(t *testing.T) {
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	group := breakerGroupView(nil, breakerIntPtr(60))
	decisionDefault := now.Add(time.Hour)

	handler.applyGroupDecisionEffectForEntry(
		group, 1, 0, breakerTestEntryID,
		modelFailureDecision("model.unavailable", health.EffectCooldownCredential, decisionDefault),
		http.StatusNotFound, now,
	)

	view := breakerEntryView(registry)
	if !view.CooldownUntil.Equal(now.Add(60 * time.Second)) {
		t.Fatalf("entry cooldown until = %v, want configured 60s instead of decision default %v",
			view.CooldownUntil, decisionDefault)
	}
	if view.FailureCount != 0 {
		t.Fatalf("entry failure count = %d, want 0 without configured threshold", view.FailureCount)
	}
}

// cooldown_seconds = 0 counts the failure but must not leave any cooldown
// deadline behind (no SetEntryCooldown call), so inspection never reports a
// stale entry_cooldown_until_ms (design §4.1, review fix 6).
func TestModelEntryBreakerZeroCooldownCountsWithoutCooldown(t *testing.T) {
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	group := breakerGroupView(breakerIntPtr(5), breakerIntPtr(0))

	handler.applyGroupDecisionEffectForEntry(
		group, 1, 0, breakerTestEntryID,
		modelFailureDecision("model.unavailable", health.EffectCooldownCredential, now.Add(time.Hour)),
		http.StatusNotFound, now,
	)

	view := breakerEntryView(registry)
	if view.FailureCount != 1 {
		t.Fatalf("entry failure count = %d, want 1", view.FailureCount)
	}
	if view.CooldownUntil != (time.Time{}) {
		t.Fatalf("entry cooldown until = %v, want zero with cooldown_seconds=0", view.CooldownUntil)
	}
	if view.RuntimeState(now) != state.EntryRuntimeAvailable {
		t.Fatalf("entry runtime state = %q, want available", view.RuntimeState(now))
	}
}

// Without an explicit per-entry cooldown the judge's own cooldown applies
// (model.unavailable carries a 1h cooldown; the others carry none).
func TestModelEntryBreakerWithoutCooldownConfigKeepsDecisionCooldown(t *testing.T) {
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	group := breakerGroupView(breakerIntPtr(5), nil)
	decisionDefault := now.Add(time.Hour)

	handler.applyGroupDecisionEffectForEntry(
		group, 1, 0, breakerTestEntryID,
		modelFailureDecision("model.unavailable", health.EffectCooldownCredential, decisionDefault),
		http.StatusNotFound, now,
	)

	view := breakerEntryView(registry)
	if !view.CooldownUntil.Equal(decisionDefault) || view.FailureCount != 1 {
		t.Fatalf("entry runtime = %#v, want decision default cooldown %v with 1 failure",
			view, decisionDefault)
	}
}

// Compatibility C5: two entries of one group that share the same upstream
// model keep independent breaker state; only the entry the failures were
// recorded for reaches its threshold.
func TestModelEntryBreakerIsolatesSiblingEntriesOfSameUpstream(t *testing.T) {
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	handler, registry := newEntryRuntimeTestHandler(t)
	group := state.GroupView{ID: 1}
	group.ModelBreakerByEntry = map[uint]map[string]*state.EntryCircuitBreaker{
		1: {
			"e000000000001": {BlacklistThreshold: breakerIntPtr(2)},
		},
	}

	for range 2 {
		handler.applyGroupDecisionEffectForEntry(
			group, 1, 0, "e000000000001",
			modelFailureDecision("model.unavailable", health.EffectCooldownCredential, time.Time{}),
			http.StatusNotFound, now,
		)
	}

	blacklisted, ok := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now)
	if !ok || !blacklisted.Blacklisted || blacklisted.FailureCount != 2 {
		t.Fatalf("entry A runtime = %#v/%t, want blacklisted with 2 failures", blacklisted, ok)
	}
	if sibling, exists := registry.EntryRuntime(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000002"}, now); exists {
		t.Fatalf("entry B runtime = %#v, want no state", sibling)
	}
}

// Success paths (design §4.2): a confirmed non-stream 2xx response clears the
// entry failure counter that earlier model failures accumulated.
func TestHandlerNonStreamSuccessClearsEntryFailureCount(t *testing.T) {
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{
			StatusCode: http.StatusNotFound, Header: make(http.Header),
			Body:               []byte(`{"error":{"code":"model_not_found"}}`),
			ClassificationBody: []byte(`{"error":{"code":"model_not_found"}}`),
			RequestWritten:     true,
		},
		{
			StatusCode: http.StatusOK, Header: make(http.Header),
			Body:           []byte(`{"ok":true}`),
			RequestWritten: true,
		},
	}}
	handler, manager, registry := newHandlerForTest(t, forwarder, "sk-one")
	publishBreakerSnapshot(t, handler, manager, &state.EntryCircuitBreaker{
		BlacklistThreshold: breakerIntPtr(5),
		CooldownSeconds:    breakerIntPtr(0),
	})
	handler.now = func() time.Time { return now }
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	serveBreakerRequest(engine, `{"model":"gpt-4o"}`)
	if view := breakerEntryView(registry); view.FailureCount != 1 {
		t.Fatalf("entry failure count after model failure = %d, want 1", view.FailureCount)
	}

	serveBreakerRequest(engine, `{"model":"gpt-4o"}`)
	if view := breakerEntryView(registry); view.FailureCount != 0 {
		t.Fatalf("entry failure count after 2xx success = %d, want 0", view.FailureCount)
	}
}

// Streaming success (CleanEOF) clears the entry failure counter symmetrically.
func TestHandlerStreamCleanEOFSuccessClearsEntryFailureCount(t *testing.T) {
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	forwarder := &scriptedForwarder{streamResults: []UpstreamResult{
		{
			StatusCode: http.StatusNotFound, Header: make(http.Header),
			Body:               []byte(`{"error":{"code":"model_not_found"}}`),
			ClassificationBody: []byte(`{"error":{"code":"model_not_found"}}`),
			RequestWritten:     true,
		},
		{
			StatusCode: http.StatusOK, Header: make(http.Header),
			Committed: true, RequestWritten: true,
			Stream: StreamObservation{EndReason: StreamEndCleanEOF},
		},
	}}
	handler, manager, registry := newHandlerForTest(t, forwarder, "sk-one")
	publishBreakerSnapshot(t, handler, manager, &state.EntryCircuitBreaker{
		BlacklistThreshold: breakerIntPtr(5),
		CooldownSeconds:    breakerIntPtr(0),
	})
	handler.now = func() time.Time { return now }
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	serveBreakerRequest(engine, `{"model":"gpt-4o","stream":true}`)
	if view := breakerEntryView(registry); view.FailureCount != 1 {
		t.Fatalf("entry failure count after stream model failure = %d, want 1", view.FailureCount)
	}

	serveBreakerRequest(engine, `{"model":"gpt-4o","stream":true}`)
	if view := breakerEntryView(registry); view.FailureCount != 0 {
		t.Fatalf("entry failure count after CleanEOF stream = %d, want 0", view.FailureCount)
	}
}

// publishBreakerSnapshot republishes the fixture snapshot with one entry that
// carries a per-entry circuit breaker configuration.
func publishBreakerSnapshot(
	t *testing.T,
	handler *Handler,
	manager *state.Manager,
	breaker *state.EntryCircuitBreaker,
) {
	t.Helper()
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{{
			ConnectionType: "api_key", ID: 1, Name: "openai", ChannelID: channel.OpenAI,
			Params: json.RawMessage(`{}`),
			Models: []state.ModelConfig{{
				ID: "gpt-4o", EntryID: breakerTestEntryID, CircuitBreaker: breaker,
			}},
			Enabled: true,
		}},
		Credentials: []state.CredentialConfig{{
			ID: 1, GroupID: 1,
			Version: 1, IdentityGeneration: 1, Fingerprint: "credential-1",
		}},
		AccessKeys: []state.AccessKeyConfig{{
			ID:      1,
			Name:    "client",
			KeyHash: handler.encryption.Hash("gl-client"),
			Status:  state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() breaker snapshot error = %v", err)
	}
}

func serveBreakerRequest(engine http.Handler, body string) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	request.Header.Set("Authorization", "Bearer gl-client")
	engine.ServeHTTP(httptest.NewRecorder(), request)
}
