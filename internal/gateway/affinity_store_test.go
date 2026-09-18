package gateway

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
	"gorm.io/gorm"

	"gpt-load/internal/affinity"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
	"gpt-load/internal/storage"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
)

// affinityStoreStub is an in-memory gateway.BindingStore used to prove the
// two-level cache/store contract without a database.
type affinityStoreStub struct {
	mu        sync.Mutex
	bindings  map[affinity.Key]affinity.Target
	lookupErr error
	upsertErr error
	lookups   int
	upserts   int
}

func (store *affinityStoreStub) Lookup(_ context.Context, key affinity.Key) (affinity.Target, bool, error) {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.lookups++
	if store.lookupErr != nil {
		return affinity.Target{}, false, store.lookupErr
	}
	target, found := store.bindings[key]
	return target, found, nil
}

func (store *affinityStoreStub) Upsert(_ context.Context, key affinity.Key, target affinity.Target) error {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.upserts++
	if store.upsertErr != nil {
		return store.upsertErr
	}
	if store.bindings == nil {
		store.bindings = make(map[affinity.Key]affinity.Target)
	}
	store.bindings[key] = target
	return nil
}

func (store *affinityStoreStub) counters() (int, int) {
	store.mu.Lock()
	defer store.mu.Unlock()
	return store.lookups, store.upserts
}

func (store *affinityStoreStub) targets() map[affinity.Key]affinity.Target {
	store.mu.Lock()
	defer store.mu.Unlock()
	copy := make(map[affinity.Key]affinity.Target, len(store.bindings))
	for key, target := range store.bindings {
		copy[key] = target
	}
	return copy
}

type orderedAffinityStore struct {
	mu           sync.Mutex
	target       affinity.Target
	firstEntered chan struct{}
	releaseFirst chan struct{}
	upserts      int
}

func (store *orderedAffinityStore) Lookup(context.Context, affinity.Key) (affinity.Target, bool, error) {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.upserts == 0 || !store.target.Valid() {
		return affinity.Target{}, false, nil
	}
	return store.target, true, nil
}

func (store *orderedAffinityStore) Upsert(_ context.Context, _ affinity.Key, target affinity.Target) error {
	store.mu.Lock()
	store.upserts++
	upsert := store.upserts
	store.mu.Unlock()
	if upsert == 1 {
		close(store.firstEntered)
		<-store.releaseFirst
	}
	store.mu.Lock()
	store.target = target
	store.mu.Unlock()
	return nil
}

func (store *orderedAffinityStore) lastTarget() affinity.Target {
	store.mu.Lock()
	defer store.mu.Unlock()
	return store.target
}

func serveAffinityResponse(t *testing.T, engine http.Handler, body string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	return response
}

func TestHandlerRecoversAffinityFromDurableStoreAfterCacheLoss(t *testing.T) {
	const body = `{"model":"gpt-4o","messages":[{"role":"user","content":"stable durable conversation"}]}`
	store := &affinityStoreStub{}

	firstForwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
	first, _, _ := newHandlerForTest(t, firstForwarder, "sk-one", "sk-two")
	first.affinityStore = store
	useAffinityRandomValues(first, 0)
	serveAffinityResponse(t, newAffinityTestEngine(t, first), body)
	assertAffinityAttemptKeys(t, firstForwarder.inputs, []string{"sk-one"})
	if _, upserts := store.counters(); upserts != 1 {
		t.Fatalf("durable upserts after success = %d, want 1", upserts)
	}

	// A brand-new handler has an empty hot cache. A bare miss with this random
	// value would select the second credential, so the hit proves read-through.
	secondForwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
	second, _, _ := newHandlerForTest(t, secondForwarder, "sk-one", "sk-two")
	second.affinityStore = store
	sink := &recordingRequestLogSink{}
	second.requestLogSink = sink
	useAffinityRandomValues(second, 1)
	response := serveAffinityResponse(t, newAffinityTestEngine(t, second), body)
	if response.Code != http.StatusOK {
		t.Fatalf("second response = %d %s, want 200", response.Code, response.Body.String())
	}
	assertAffinityAttemptKeys(t, secondForwarder.inputs, []string{"sk-one"})
	assertAffinityStates(t, sink.snapshot(), []telemetry.AffinityState{telemetry.AffinityStateHit})
	assertAffinityHits(t, sink.snapshot(), []bool{true})
}

func TestHandlerSkipsDurableLookupWhenAllAllowedGroupsDisableAffinity(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	for groupID, group := range manager.Current().Groups {
		group.AffinityEnabled = false
		manager.Current().Groups[groupID] = group
	}
	store := &affinityStoreStub{lookupErr: errors.New("durable store unavailable")}
	handler.affinityStore = store
	useAffinityRandomValues(handler, 0)

	response := serveAffinityResponse(
		t, newAffinityTestEngine(t, handler),
		`{"model":"gpt-4o","messages":[{"role":"user","content":"disabled affinity"}]}`,
	)
	if response.Code != http.StatusOK {
		t.Fatalf("response = %d %s, want 200", response.Code, response.Body.String())
	}
	lookups, _ := store.counters()
	if lookups != 0 {
		t.Fatalf("durable lookups with all groups disabled = %d, want 0", lookups)
	}
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one"})
}

func TestHandlerMigratesDurableBindingAfterBoundStreamProviderFailure(t *testing.T) {
	forwarder := &scriptedForwarder{streamResults: []UpstreamResult{
		{StatusCode: http.StatusOK, Committed: true, Stream: StreamObservation{EndReason: StreamEndCleanEOF}},
		affinityUpstreamFailure(),
		{StatusCode: http.StatusOK, Committed: true, Stream: StreamObservation{EndReason: StreamEndCleanEOF}},
	}}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	store := &affinityStoreStub{}
	handler.affinityStore = store
	useAffinityRandomValues(handler, 0, 1)
	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"migrate stream binding"}]}`

	serveAffinityResponse(t, engine, body)
	serveAffinityResponse(t, engine, body)

	bindings := store.targets()
	if len(bindings) != 1 {
		t.Fatalf("durable bindings = %#v, want one binding", bindings)
	}
	for _, target := range bindings {
		if target.CredentialID != 2 || target.GroupID != 2 {
			t.Fatalf("durable stream target after provider failure = %#v, want credential 2/group 2", target)
		}
	}
	assertAffinityAttemptKeys(t, forwarder.streamInputs, []string{"sk-one", "sk-one", "sk-two"})
}

func TestHandlerPreservesDurableBindingWhenBoundTargetWasNotDispatched(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	store := &affinityStoreStub{}
	handler.affinityStore = store
	useAffinityRandomValues(handler, 0, 1)
	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"preserve disabled target"}]}`

	serveAffinityResponse(t, engine, body)
	group := manager.Current().Groups[1]
	group.AffinityEnabled = false
	manager.Current().Groups[1] = group
	serveAffinityResponse(t, engine, body)

	bindings := store.targets()
	if len(bindings) != 1 {
		t.Fatalf("durable bindings = %#v, want one binding", bindings)
	}
	for _, target := range bindings {
		if target.CredentialID != 1 || target.GroupID != 1 {
			t.Fatalf("durable target after disabled bound request = %#v, want credential 1/group 1", target)
		}
	}
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-two"})
}

func TestHandlerMigratesDurableBindingAfterBoundProviderFailure(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		successfulAffinityResult(), affinityUpstreamFailure(), successfulAffinityResult(),
	}}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	store := &affinityStoreStub{}
	handler.affinityStore = store
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, 1)
	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"migrate after provider failure"}]}`

	serveAffinityResponse(t, engine, body)
	serveAffinityResponse(t, engine, body)

	bindings := store.targets()
	if len(bindings) != 1 {
		t.Fatalf("durable bindings = %#v, want one binding", bindings)
	}
	for _, target := range bindings {
		if target.CredentialID != 2 || target.GroupID != 2 {
			t.Fatalf("durable target after provider failure = %#v, want credential 2/group 2", target)
		}
	}
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one", "sk-two"})
	events := sink.snapshot()
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss, telemetry.AffinityStateHit,
	})
	assertAffinityHits(t, events, []bool{false, true})
	attempts := events[1].Attempts
	if len(attempts) != 2 || attempts[0].CredentialID != 1 || attempts[1].CredentialID != 2 {
		t.Fatalf("migration request attempts = %#v, want bound credential 1 then fallback credential 2", attempts)
	}
}

func TestHandlerSerializesAffinityCacheAndDurableWritesPerKey(t *testing.T) {
	store := &orderedAffinityStore{
		firstEntered: make(chan struct{}),
		releaseFirst: make(chan struct{}),
	}
	handler, manager, _ := newHandlerForTest(t, &scriptedForwarder{}, "sk-one", "sk-two")
	handler.affinityStore = store
	if !handler.affinityCache.Configure(manager.Current().Revision, 10, time.Hour) {
		t.Fatal("Configure() failed")
	}
	key := affinity.Key("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	observation := handler.affinityCache.Lookup(key)
	selection := scheduler.Selection{CredentialID: 1, GroupID: 1, Group: manager.Current().Groups[1]}
	first := requestAffinity{key: key, observation: observation}
	firstDone := make(chan struct{})
	go func() {
		handler.recordAffinitySuccess(t.Context(), first, selection, state.CredentialRef{ID: 1, GroupID: 1, IdentityGeneration: 1})
		close(firstDone)
	}()
	select {
	case <-store.firstEntered:
	case <-time.After(time.Second):
		t.Fatal("first durable upsert did not start")
	}

	second := requestAffinity{key: key, observation: handler.affinityCache.Lookup(key)}
	secondSelection := selection
	secondSelection.CredentialID = 2
	secondSelection.GroupID = 2
	secondSelection.Group = manager.Current().Groups[2]
	secondDone := make(chan struct{})
	go func() {
		handler.recordAffinitySuccess(t.Context(), second, secondSelection, state.CredentialRef{ID: 2, GroupID: 2, IdentityGeneration: 2})
		close(secondDone)
	}()
	select {
	case <-secondDone:
		t.Fatal("second same-key durable upsert completed before first upsert released")
	case <-time.After(100 * time.Millisecond):
	}
	close(store.releaseFirst)
	select {
	case <-firstDone:
	case <-time.After(time.Second):
		t.Fatal("first affinity write did not finish")
	}
	select {
	case <-secondDone:
	case <-time.After(time.Second):
		t.Fatal("second affinity write did not finish")
	}
	if target := store.lastTarget(); target.CredentialID != 2 || target.GroupID != 2 {
		t.Fatalf("last durable target = %#v, want credential 2/group 2", target)
	}
}

func TestHandlerFailsClosedWhenDurableAffinityLookupFails(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	handler.affinityStore = &affinityStoreStub{lookupErr: errors.New("durable store unavailable")}
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	engine := newAffinityTestEngine(t, handler)

	response := serveAffinityResponse(
		t, engine,
		`{"model":"gpt-4o","messages":[{"role":"user","content":"stable durable conversation"}]}`,
	)
	if response.Code != reasonConfigurationChanged.Status ||
		!strings.Contains(response.Body.String(), reasonConfigurationChanged.Code) {
		t.Fatalf("response = %d %s, want %d %s",
			response.Code, response.Body.String(), reasonConfigurationChanged.Status, reasonConfigurationChanged.Code)
	}
	if len(forwarder.inputs) != 0 {
		t.Fatalf("attempts after durable lookup failure = %d, want 0", len(forwarder.inputs))
	}
	assertAffinityStates(t, sink.snapshot(), []telemetry.AffinityState{telemetry.AffinityStateCacheUnavailable})
	assertAffinityHits(t, sink.snapshot(), []bool{false})
}

func TestWebsocketMigratesDurableBindingAfterBoundProviderFailure(t *testing.T) {
	handler, engine, sink := newAffinityWebsocketFixture(t, "http://127.0.0.1:1/v1")
	store := &affinityStoreStub{}
	handler.affinityStore = store
	useAffinityRandomValues(handler, 0, 0)
	var opens atomic.Int32
	handler.forwarder = websocketScriptForwarder{
		AttemptForwarder: handler.forwarder,
		open: func(context.Context, ForwardInput) (execution.WebsocketSession, execution.WebsocketResult) {
			openNumber := opens.Add(1)
			if openNumber == 2 {
				return nil, execution.WebsocketResult{DispatchState: execution.DispatchNotSent, Error: &execution.ErrorEvidence{
					Kind: execution.ErrorKindTransport, OriginHint: execution.ErrorOriginUpstream,
					ScopeHint: execution.ErrorScopeCredential, ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
				}}
			}
			session := &websocketScriptSession{done: make(chan struct{})}
			session.turn = func(ctx context.Context, _ []byte, emit func(context.Context, []byte) error) execution.WebsocketResult {
				if err := emit(ctx, websocketCompleted("ws-affinity", "")); err != nil {
					return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent, Error: &execution.ErrorEvidence{
						Kind: execution.ErrorKindCanceled, OriginHint: execution.ErrorOriginDownstream,
						ScopeHint: execution.ErrorScopeRequest,
					}}
				}
				return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent}
			}
			return session, execution.WebsocketResult{DispatchState: execution.DispatchNotSent}
		},
	}
	server := httptest.NewServer(engine)
	defer server.Close()
	body := `{"type":"response.create","model":"public","prompt_cache_key":"ws-durable-migrate","input":"hello"}`

	first := dialGatewayWebsocket(t, server.URL)
	if err := first.WriteMessage(websocket.TextMessage, []byte(body)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := first.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	_ = first.Close()

	second := dialGatewayWebsocket(t, server.URL)
	if err := second.WriteMessage(websocket.TextMessage, []byte(body)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := second.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	if opens.Load() != 3 {
		t.Fatalf("websocket opens = %d, want bound attempt plus fallback", opens.Load())
	}
	bindings := store.targets()
	if len(bindings) != 1 {
		t.Fatalf("durable bindings = %#v, want one binding", bindings)
	}
	for _, target := range bindings {
		if target.CredentialID != 2 || target.GroupID != 2 {
			t.Fatalf("durable websocket target after provider failure = %#v, want credential 2/group 2", target)
		}
	}
	if events := waitWebsocketLogs(t, sink, 2); len(events) != 2 {
		t.Fatalf("websocket request logs = %d, want 2", len(events))
	}
}

func TestHandlerKeepsResponseWhenDurableAffinityUpsertFails(t *testing.T) {
	var logs bytes.Buffer
	logger := logrus.New()
	logger.SetOutput(&logs)
	logger.SetFormatter(&logrus.JSONFormatter{DisableTimestamp: true})
	logger.SetLevel(logrus.ErrorLevel)

	forwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	handler.logger = logger
	handler.affinityStore = &affinityStoreStub{upsertErr: errors.New("durable store write failed")}
	useAffinityRandomValues(handler, 0)
	engine := newAffinityTestEngine(t, handler)

	response := serveAffinityResponse(
		t, engine,
		`{"model":"gpt-4o","messages":[{"role":"user","content":"stable durable conversation"}]}`,
	)
	if response.Code != http.StatusOK {
		t.Fatalf("response = %d %s, want 200", response.Code, response.Body.String())
	}
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one"})
	if !strings.Contains(logs.String(), "affinity_binding_persist_failed") {
		t.Fatalf("persistence failure was not logged: %s", logs.String())
	}
}

func TestWebsocketFailsClosedWhenDurableAffinityLookupFails(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()
	handler, engine, _ := newAffinityWebsocketFixture(t, upstream.URL+"/v1")
	handler.affinityStore = &affinityStoreStub{lookupErr: errors.New("durable store unavailable")}
	server := httptest.NewServer(engine)
	defer server.Close()

	conn := dialGatewayWebsocket(t, server.URL)
	body := `{"type":"response.create","model":"public","prompt_cache_key":"durable-lookup-key"}`
	if err := conn.WriteMessage(websocket.TextMessage, []byte(body)); err != nil {
		t.Fatal(err)
	}
	var event struct {
		Type  string `json:"type"`
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := conn.ReadJSON(&event); err != nil {
		t.Fatal(err)
	}
	if event.Type != "error" || event.Error.Code != reasonConfigurationChanged.Code {
		t.Fatalf("websocket event = %+v, want error %s", event, reasonConfigurationChanged.Code)
	}
}

func TestHandlerRecoversAffinityAfterHotCacheEviction(t *testing.T) {
	const first = `{"model":"gpt-4o","messages":[{"role":"user","content":"evicted conversation"}]}`
	const second = `{"model":"gpt-4o","messages":[{"role":"user","content":"other conversation"}]}`
	forwarder := &scriptedForwarder{results: successfulAffinityResults(4)}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	handler.affinityStore = &affinityStoreStub{}
	if _, err := manager.Publish(affinityRuntimeInput(handler, config.Settings{
		state.SettingRetryCount:       testDefaultRetryBudget,
		state.SettingAffinityCapacity: 1,
	})); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, 0, 1)
	engine := newAffinityTestEngine(t, handler)

	serveAffinityResponse(t, engine, first)
	serveAffinityResponse(t, engine, first)
	serveAffinityResponse(t, engine, second)
	response := serveAffinityResponse(t, engine, first)
	if response.Code != http.StatusOK {
		t.Fatalf("response = %d %s, want 200", response.Code, response.Body.String())
	}
	// The capacity=1 hot cache evicts the first binding, so the fourth request
	// can only hit the first credential through durable read-through.
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one", "sk-two", "sk-one"})
	assertAffinityStates(t, sink.snapshot(), []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss,
		telemetry.AffinityStateHit,
		telemetry.AffinityStateCacheMiss,
		telemetry.AffinityStateHit,
	})
}

// TestHandlerRecoversDurableAffinityAfterHotCacheTTLExpiry proves R5b for the
// TTL axis: after the hot copy expires the binding is still resolved through
// durable read-through and the original credential is attempted first. The
// cache owns its clock, so the frozen snapshot TTL is the controllable knob: a
// one-nanosecond TTL expires the hot copy before the next request while leaving
// capacity and revision untouched, so Configure never clears the cache. The
// test never sleeps.
func TestHandlerRecoversDurableAffinityAfterHotCacheTTLExpiry(t *testing.T) {
	const body = `{"model":"gpt-4o","messages":[{"role":"user","content":"ttl durable conversation"}]}`
	store := &affinityStoreStub{}
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	handler.affinityStore = store
	manager.Current().Settings.AffinityTTL = time.Nanosecond
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, 1)
	engine := newAffinityTestEngine(t, handler)

	serveAffinityResponse(t, engine, body)
	if _, upserts := store.counters(); upserts != 1 {
		t.Fatalf("durable upserts after first request = %d, want 1", upserts)
	}
	lookupsBefore, _ := store.counters()

	response := serveAffinityResponse(t, engine, body)
	if response.Code != http.StatusOK {
		t.Fatalf("second response = %d %s, want 200", response.Code, response.Body.String())
	}
	lookupsAfter, _ := store.counters()
	if lookupsAfter <= lookupsBefore {
		t.Fatalf("durable lookups after TTL expiry = %d, want > %d (hot copy should have expired)", lookupsAfter, lookupsBefore)
	}
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one"})
	events := sink.snapshot()
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss,
		telemetry.AffinityStateHit,
	})
	assertAffinityHits(t, events, []bool{false, true})
}

// TestHandlerPreservesDurableBindingWhenBoundTargetWasNotDispatchable proves
// the strict migration contract for non-dispatch reasons: an identity change, a
// cooldown or a blacklist removes the bound target from the actual attempt
// order, so a successful fallback must not overwrite the durable row with a
// target that never reached the provider.
func TestHandlerPreservesDurableBindingWhenBoundTargetWasNotDispatchable(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(t *testing.T, registry *state.CredentialRegistry)
	}{
		{
			name: "identity generation changed",
			mutate: func(t *testing.T, registry *state.CredentialRegistry) {
				t.Helper()
				refs := registry.CaptureActiveCredentialRefs([]uint{1, 2})
				entries := make([]state.CredentialEntry, 0, len(refs))
				for _, ref := range refs {
					generation := ref.IdentityGeneration
					if ref.ID == 1 {
						generation++
					}
					entries = append(entries, state.CredentialEntry{
						ID: ref.ID, GroupID: ref.GroupID, Version: ref.Version,
						IdentityGeneration: generation, Fingerprint: ref.Fingerprint,
						EncryptedValue: ref.EncryptedValue,
					})
				}
				if err := registry.ReplaceCredentials(entries); err != nil {
					t.Fatalf("ReplaceCredentials() error = %v", err)
				}
			},
		},
		{
			name: "bound credential cooled down",
			mutate: func(t *testing.T, registry *state.CredentialRegistry) {
				t.Helper()
				if !registry.SetCooldown(1, time.Now().Add(time.Hour)) {
					t.Fatal("SetCooldown() = false")
				}
			},
		},
		{
			name: "bound credential blacklisted",
			mutate: func(t *testing.T, registry *state.CredentialRegistry) {
				t.Helper()
				if exists, _ := registry.SetBlacklistedWithChange(1); !exists {
					t.Fatal("SetBlacklistedWithChange() = false")
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
			handler, _, registry := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
			store := &affinityStoreStub{}
			handler.affinityStore = store
			useAffinityRandomValues(handler, 0, 1)
			engine := newAffinityTestEngine(t, handler)
			body := `{"model":"gpt-4o","messages":[{"role":"user","content":"preserve non dispatched binding"}]}`

			serveAffinityResponse(t, engine, body)
			test.mutate(t, registry)
			serveAffinityResponse(t, engine, body)

			bindings := store.targets()
			if len(bindings) != 1 {
				t.Fatalf("durable bindings = %#v, want one preserved binding", bindings)
			}
			for _, target := range bindings {
				if target.CredentialID != 1 || target.GroupID != 1 {
					t.Fatalf("durable target after non-dispatchable bound request = %#v, want credential 1/group 1", target)
				}
			}
			assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-two"})
		})
	}
}

func countAffinityBindingRows(t *testing.T, db *gorm.DB) int64 {
	t.Helper()
	var count int64
	if err := db.Model(&models.SystemSetting{}).
		Where("key LIKE ?", models.InternalSystemSettingPrefix+"affinity.binding.%").
		Count(&count).Error; err != nil {
		t.Fatalf("count affinity rows error = %v", err)
	}
	return count
}

// TestHandlerRecoversDurableAffinityFromReopenedSQLiteStore proves R5a/R5b end
// to end against the production durable authority: a real SQLite
// storage.AffinityStore is written through the gateway request path, the
// database file is reopened, and a brand-new Handler with an empty hot cache
// resolves the same key and attempts the original credential first. The random
// value selects the second credential on a bare miss, so durable read-through is
// the only explanation for the first attempt.
func TestHandlerRecoversDurableAffinityFromReopenedSQLiteStore(t *testing.T) {
	const body = `{"model":"gpt-4o","messages":[{"role":"user","content":"reopened sqlite conversation"}]}`
	path := filepath.Join(t.TempDir(), "affinity-reopen.db")

	firstForwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
	first, _, _ := newHandlerForTest(t, firstForwarder, "sk-one", "sk-two")
	firstDB, err := storage.Open(path)
	if err != nil {
		t.Fatalf("storage.Open() error = %v", err)
	}
	if err := storage.AutoMigrate(firstDB); err != nil {
		t.Fatalf("AutoMigrate() error = %v", err)
	}
	first.affinityStore = storage.NewAffinityStore(firstDB)
	useAffinityRandomValues(first, 0)
	serveAffinityResponse(t, newAffinityTestEngine(t, first), body)
	assertAffinityAttemptKeys(t, firstForwarder.inputs, []string{"sk-one"})
	if rows := countAffinityBindingRows(t, firstDB); rows != 1 {
		t.Fatalf("persisted affinity binding rows = %d, want 1", rows)
	}
	firstSQL, err := firstDB.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	if err := firstSQL.Close(); err != nil {
		t.Fatalf("close first database error = %v", err)
	}

	secondDB, err := storage.Open(path)
	if err != nil {
		t.Fatalf("storage.Open() after reopen error = %v", err)
	}
	if err := storage.AutoMigrate(secondDB); err != nil {
		t.Fatalf("AutoMigrate() after reopen error = %v", err)
	}
	secondSQL, err := secondDB.DB()
	if err != nil {
		t.Fatalf("db.DB() after reopen error = %v", err)
	}
	t.Cleanup(func() { _ = secondSQL.Close() })
	if rows := countAffinityBindingRows(t, secondDB); rows != 1 {
		t.Fatalf("persisted affinity binding rows after reopen = %d, want 1", rows)
	}

	secondForwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
	second, _, _ := newHandlerForTest(t, secondForwarder, "sk-one", "sk-two")
	second.affinityStore = storage.NewAffinityStore(secondDB)
	sink := &recordingRequestLogSink{}
	second.requestLogSink = sink
	useAffinityRandomValues(second, 1)
	response := serveAffinityResponse(t, newAffinityTestEngine(t, second), body)
	if response.Code != http.StatusOK {
		t.Fatalf("reopened response = %d %s, want 200", response.Code, response.Body.String())
	}
	assertAffinityAttemptKeys(t, secondForwarder.inputs, []string{"sk-one"})
	assertAffinityStates(t, sink.snapshot(), []telemetry.AffinityState{telemetry.AffinityStateHit})
	assertAffinityHits(t, sink.snapshot(), []bool{true})
}
