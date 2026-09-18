package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
)

// affinityBucketGroup builds one eligible route entry for the cross-bucket
// fixtures. The buckets are expressed only through the frozen snapshot:
// priority is the entry priority tier, channelID selects native versus
// converted route mode, and the Responses store bucket is derived from the
// channel's store handling by the scheduler, never injected by the test.
func affinityBucketGroup(
	id uint,
	channelID channel.ID,
	model string,
	priority int,
	weight int,
) state.GroupConfig {
	return state.GroupConfig{
		ConnectionType: "api_key",
		ID:             id,
		Name:           fmt.Sprintf("bucket-%d", id),
		ChannelID:      channelID,
		Params:         json.RawMessage(`{}`),
		Enabled:        true,
		Models: []state.ModelConfig{{
			ID: model, EntryID: "e000000000001",
			Priority: new(priority), Weight: new(weight),
		}},
	}
}

// affinityBucketRuntimeInput republishes the same two credential identities
// used by newHandlerForTest under a caller-owned group shape, so a binding
// established before the publish stays resolvable by (GroupID, CredentialID,
// IdentityGeneration).
func affinityBucketRuntimeInput(
	handler *Handler,
	settings config.Settings,
	groups ...state.GroupConfig,
) state.CompileInput {
	return state.CompileInput{
		SystemSettings:  settings,
		ChannelRegistry: channel.NewRegistry(),
		Groups:          groups,
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "credential-1"},
			{ID: 2, GroupID: 2, Version: 1, IdentityGeneration: 2, Fingerprint: "credential-2"},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}
}

func publishAffinityBucketConfig(
	t *testing.T,
	handler *Handler,
	settings config.Settings,
	groups ...state.GroupConfig,
) {
	t.Helper()
	if _, err := handler.manager.Publish(affinityBucketRuntimeInput(handler, settings, groups...)); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
}

func serveAffinityBucketRequest(
	t *testing.T,
	engine http.Handler,
	kind string,
	body string,
) *httptest.ResponseRecorder {
	t.Helper()
	path := "/v1/chat/completions"
	if kind == "responses" {
		path = "/v1/responses"
	}
	request := httptest.NewRequest(http.MethodPost, path, bytes.NewBufferString(body))
	request.Header.Set("Authorization", "Bearer gl-client")
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("response = %d %s, want 200", response.Code, response.Body.String())
	}
	return response
}

func assertAffinityInputCredentials(
	t *testing.T,
	inputs []ForwardInput,
	want []uint,
) {
	t.Helper()
	got := make([]uint, 0, len(inputs))
	for _, input := range inputs {
		got = append(got, input.Credential.ID)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("Forward credentials = %#v, want %#v", got, want)
	}
}

func assertAffinityStoreTargetCredential(
	t *testing.T,
	store *affinityStoreStub,
	wantCredentialID uint,
	wantGroupID uint,
) {
	t.Helper()
	bindings := store.targets()
	if len(bindings) != 1 {
		t.Fatalf("durable bindings = %#v, want exactly one binding", bindings)
	}
	for _, target := range bindings {
		if target.CredentialID != wantCredentialID || target.GroupID != wantGroupID {
			t.Fatalf("durable target = %#v, want credential %d/group %d", target, wantCredentialID, wantGroupID)
		}
	}
}

// logAffinityBucketEvidence prints the frozen Forward facts and request-log
// observation for one request so the U008 evidence file carries the real
// credential, route/bucket, state and attempt order rather than only an "ok".
func logAffinityBucketEvidence(
	t *testing.T,
	label string,
	inputs []ForwardInput,
	event telemetry.RequestEvent,
) {
	t.Helper()
	credentialIDs := make([]uint, 0, len(inputs))
	routeModes := make([]execution.RouteMode, 0, len(inputs))
	storeDowngraded := make([]bool, 0, len(inputs))
	channelIDs := make([]string, 0, len(inputs))
	for _, input := range inputs {
		credentialIDs = append(credentialIDs, input.Credential.ID)
		routeModes = append(routeModes, input.RouteMode)
		storeDowngraded = append(storeDowngraded, input.ResponsesStoreDowngraded)
		channelIDs = append(channelIDs, input.ChannelID)
	}
	attemptCredentials := make([]uint, 0, len(event.Attempts))
	attemptRoutes := make([]execution.RouteMode, 0, len(event.Attempts))
	for _, attempt := range event.Attempts {
		attemptCredentials = append(attemptCredentials, attempt.CredentialID)
		attemptRoutes = append(attemptRoutes, attempt.RouteMode)
	}
	t.Logf(
		"%s: Forward credentials=%v routeModes=%v storeDowngraded=%v channelIDs=%v | requestLog state=%q hit=%v attemptCredentials=%v attemptRoutes=%v",
		label, credentialIDs, routeModes, storeDowngraded, channelIDs,
		event.AffinityState, event.AffinityHit, attemptCredentials, attemptRoutes,
	)
}

// logAffinityFixtureGroups records the published route-entry facts (channel,
// priority tier, weight) that place the bound credential in its competing
// bucket, so the evidence file shows the bucket construction, not just the
// resulting route mode.
func logAffinityFixtureGroups(t *testing.T, groups []state.GroupConfig) {
	t.Helper()
	for _, group := range groups {
		model := group.Models[0]
		priority, weight := 0, 0
		if model.Priority != nil {
			priority = *model.Priority
		}
		if model.Weight != nil {
			weight = *model.Weight
		}
		t.Logf(
			"fixture group=%d channel=%s model=%s priority=%d weight=%d",
			group.ID, group.ChannelID, model.ID, priority, weight,
		)
	}
}

// TestAffinityHitIsFirstActualForwardAcrossBuckets proves R1 at the real
// gateway boundary: after a binding is established through the handler, the
// next request still reaches Forward with the bound credential first even when
// that credential sits in a lower priority tier, on a converted route mode, or
// inside the Responses store-downgraded bucket. The assertion reads the real
// ForwardInput credential/route facts and the request-log hit state; a bare
// miss would select the competing native/regular/higher-priority bucket first.
func TestAffinityHitIsFirstActualForwardAcrossBuckets(t *testing.T) {
	tests := []struct {
		name                string
		kind                string
		groups              []state.GroupConfig
		wantRouteMode       execution.RouteMode
		wantStoreDowngraded bool
		wantChannelID       channel.ID
	}{
		{
			name: "lower_priority_bucket",
			kind: "chat",
			groups: []state.GroupConfig{
				affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 2, 1),
				affinityBucketGroup(2, channel.OpenAI, "gpt-4o", 1, 1),
			},
			wantRouteMode: execution.RouteNative,
			wantChannelID: channel.OpenAI,
		},
		{
			name: "converted_route_bucket",
			kind: "chat",
			groups: []state.GroupConfig{
				affinityBucketGroup(1, channel.Anthropic, "gpt-4o", 1, 1),
				affinityBucketGroup(2, channel.OpenAI, "gpt-4o", 1, 1),
			},
			wantRouteMode: execution.RouteConverted,
			wantChannelID: channel.Anthropic,
		},
		{
			name: "responses_store_downgraded_bucket",
			kind: "responses",
			groups: []state.GroupConfig{
				affinityBucketGroup(1, channel.Anthropic, "gpt-4o", 1, 1),
				affinityBucketGroup(2, channel.OpenAI, "gpt-4o", 1, 1),
			},
			wantRouteMode:       execution.RouteConverted,
			wantStoreDowngraded: true,
			wantChannelID:       channel.Anthropic,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := `{"model":"gpt-4o","messages":[{"role":"user","content":"cross bucket stable conversation"}]}`
			if test.kind == "responses" {
				body = `{"model":"gpt-4o","input":"cross bucket stable conversation","prompt_cache_key":"cross-bucket-store-key","store":true}`
			}
			forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
			handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
			if test.kind == "responses" {
				handler.dialects = dialect.NewSet(dialect.NewOpenAIResponses())
			}
			store := &affinityStoreStub{}
			handler.affinityStore = store
			sink := &recordingRequestLogSink{}
			handler.requestLogSink = sink
			// A bare miss on every published shape would pick the competing
			// bucket (higher priority, native mode, or regular store class), so
			// the expected first Forward credential can only come from affinity.
			useAffinityRandomValues(handler, 0)
			engine := newAffinityTestEngine(t, handler)

			serveAffinityBucketRequest(t, engine, test.kind, body)
			assertAffinityStoreTargetCredential(t, store, 1, 1)

			publishAffinityBucketConfig(
				t, handler,
				config.Settings{state.SettingRetryCount: testDefaultRetryBudget},
				test.groups...,
			)
			logAffinityFixtureGroups(t, test.groups)
			serveAffinityBucketRequest(t, engine, test.kind, body)

			assertAffinityInputCredentials(t, forwarder.inputs, []uint{1, 1})
			second := forwarder.inputs[1]
			if second.RouteMode != test.wantRouteMode ||
				second.ResponsesStoreDowngraded != test.wantStoreDowngraded ||
				second.ChannelID != string(test.wantChannelID) {
				t.Fatalf(
					"second Forward route facts = mode %q downgraded %v channel %q, want %q/%v/%q",
					second.RouteMode, second.ResponsesStoreDowngraded, second.ChannelID,
					test.wantRouteMode, test.wantStoreDowngraded, test.wantChannelID,
				)
			}

			events := sink.snapshot()
			if len(events) != 2 {
				t.Fatalf("request log events = %d, want 2", len(events))
			}
			assertAffinityStates(t, events, []telemetry.AffinityState{
				telemetry.AffinityStateCacheMiss,
				telemetry.AffinityStateHit,
			})
			assertAffinityHits(t, events, []bool{false, true})
			attempts := events[1].Attempts
			if len(attempts) != 1 || attempts[0].CredentialID != 1 ||
				attempts[0].RouteMode != test.wantRouteMode {
				t.Fatalf("second request attempts = %#v, want one bound credential 1 on %q", attempts, test.wantRouteMode)
			}
			assertAffinityStoreTargetCredential(t, store, 1, 1)
			logAffinityBucketEvidence(t, test.name, forwarder.inputs, events[1])
		})
	}
}

// TestAffinityCrossBucketProviderFailureFallsBackAfterBoundAttempt proves R2
// across a bucket boundary: the bound credential in the lower priority tier is
// actually attempted first, only its real retryable 401 lets the same request
// fall back to the other credential, and the durable binding migrates only
// after that dispatched failure.
func TestAffinityCrossBucketProviderFailureFallsBackAfterBoundAttempt(t *testing.T) {
	const body = `{"model":"gpt-4o","messages":[{"role":"user","content":"cross bucket failure fallback"}]}`
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		successfulAffinityResult(),
		affinityUpstreamFailure(),
		successfulAffinityResult(),
	}}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	store := &affinityStoreStub{}
	handler.affinityStore = store
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0)
	engine := newAffinityTestEngine(t, handler)

	serveAffinityBucketRequest(t, engine, "chat", body)
	assertAffinityStoreTargetCredential(t, store, 1, 1)

	groups := []state.GroupConfig{
		affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 2, 1),
		affinityBucketGroup(2, channel.OpenAI, "gpt-4o", 1, 1),
	}
	publishAffinityBucketConfig(
		t, handler,
		config.Settings{state.SettingRetryCount: testDefaultRetryBudget},
		groups...,
	)
	logAffinityFixtureGroups(t, groups)
	serveAffinityBucketRequest(t, engine, "chat", body)

	assertAffinityInputCredentials(t, forwarder.inputs, []uint{1, 1, 2})
	if forwarder.inputs[1].RouteMode != execution.RouteNative {
		t.Fatalf("bound attempt route mode = %q, want native", forwarder.inputs[1].RouteMode)
	}
	events := sink.snapshot()
	if len(events) != 2 {
		t.Fatalf("request log events = %d, want 2", len(events))
	}
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss,
		telemetry.AffinityStateHit,
	})
	assertAffinityHits(t, events, []bool{false, true})
	attempts := events[1].Attempts
	if len(attempts) != 2 || attempts[0].CredentialID != 1 || attempts[1].CredentialID != 2 {
		t.Fatalf("failure request attempts = %#v, want bound credential 1 then fallback credential 2", attempts)
	}
	if !attempts[0].WillRetry {
		t.Fatalf("failed bound attempt WillRetry = %v, want true", attempts[0].WillRetry)
	}
	assertAffinityStoreTargetCredential(t, store, 2, 2)
	logAffinityBucketEvidence(t, "cross_bucket_provider_failure", forwarder.inputs, events[1])
}

// TestAffinityBindingSurvivesUnrelatedWeightAndCatalogLikeRepublish proves
// R3/R7 at the handler boundary: an entry weight change on the competing route
// and a catalog-like revision publish (a Models.dev/catalog-shaped snapshot
// republish that this test does NOT claim to exercise through the external
// Models.dev network client) each bump the snapshot revision without touching
// the bound target's group, model, identity, or route eligibility. The durable
// row and the first actual Forward credential must survive both publishes.
func TestAffinityBindingSurvivesUnrelatedWeightAndCatalogLikeRepublish(t *testing.T) {
	const body = `{"model":"gpt-4o","messages":[{"role":"user","content":"unrelated publish stable conversation"}]}`
	tests := []struct {
		name         string
		settings     config.Settings
		groups       []state.GroupConfig
		secondRandom int64
	}{
		{
			name:     "unrelated_entry_weight_change (R3/R7)",
			settings: config.Settings{state.SettingRetryCount: testDefaultRetryBudget},
			groups: []state.GroupConfig{
				affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 1, 1),
				affinityBucketGroup(2, channel.OpenAI, "gpt-4o", 1, 5),
			},
			// A bare miss with ticket 4 falls into the heavier competing entry,
			// so the bound credential can only be attempted first through affinity.
			secondRandom: 4,
		},
		{
			name: "catalog_like_revision_publish (R3/R7)",
			settings: config.Settings{
				state.SettingRetryCount:               testDefaultRetryBudget,
				state.SettingModelsDevAutoSyncEnabled: true,
			},
			groups: []state.GroupConfig{
				affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 1, 1),
				affinityBucketGroup(2, channel.OpenAI, "gpt-4o", 1, 1),
				affinityBucketGroup(3, channel.OpenAI, "catalog-derived-model", 1, 1),
			},
			// A bare miss with ticket 1 selects the competing credential.
			secondRandom: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
			handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
			store := &affinityStoreStub{}
			handler.affinityStore = store
			sink := &recordingRequestLogSink{}
			handler.requestLogSink = sink
			useAffinityRandomValues(handler, 0, test.secondRandom)
			engine := newAffinityTestEngine(t, handler)

			serveAffinityBucketRequest(t, engine, "chat", body)
			assertAffinityStoreTargetCredential(t, store, 1, 1)
			revisionBefore := manager.Current().Revision

			publishAffinityBucketConfig(t, handler, test.settings, test.groups...)
			logAffinityFixtureGroups(t, test.groups)
			if next := manager.Current().Revision; next <= revisionBefore {
				t.Fatalf("revision after publish = %d, want > %d", next, revisionBefore)
			}
			serveAffinityBucketRequest(t, engine, "chat", body)

			assertAffinityInputCredentials(t, forwarder.inputs, []uint{1, 1})
			events := sink.snapshot()
			if len(events) != 2 {
				t.Fatalf("request log events = %d, want 2", len(events))
			}
			assertAffinityStates(t, events, []telemetry.AffinityState{
				telemetry.AffinityStateCacheMiss,
				telemetry.AffinityStateHit,
			})
			assertAffinityHits(t, events, []bool{false, true})
			attempts := events[1].Attempts
			if len(attempts) != 1 || attempts[0].CredentialID != 1 {
				t.Fatalf("published request attempts = %#v, want only bound credential 1", attempts)
			}
			assertAffinityStoreTargetCredential(t, store, 1, 1)
			logAffinityBucketEvidence(t, test.name, forwarder.inputs, events[1])
		})
	}
}
