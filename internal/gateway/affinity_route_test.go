package gateway

import (
	"context"
	"fmt"

	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"gpt-load/internal/affinity"
	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

func publishAffinityModelRoutes(t *testing.T, h *Handler, models ...state.ModelConfig) {
	t.Helper()
	group := affinityBucketGroup(1, channel.OpenAI, "bound-model", 1, 1)
	group.Models = models
	input := affinityBucketRuntimeInput(h, config.Settings{}, group)
	input.Credentials = input.Credentials[:1]
	if _, err := h.manager.Publish(input); err != nil {
		t.Fatal(err)
	}
}

func affinityRouteModel(model, entry string) state.ModelConfig {
	return state.ModelConfig{ID: model, Alias: "public", EntryID: entry}
}

func TestHandlerReusesBoundUpstreamModelOfSameCredential(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(2), streamResults: successfulAffinityResults(2)}
			h, _, _ := newHandlerForTest(t, forwarder, "sk-one")
			store := &affinityStoreStub{}
			h.affinityStore = store
			sink := &recordingRequestLogSink{}
			h.requestLogSink = sink
			useAffinityRandomValues(h, 0)
			bound := affinityRouteModel("zen/deepseek-v4.1-flash-free", "e000000000002")
			publishAffinityModelRoutes(t, h, bound)
			engine := newAffinityTestEngine(t, h)
			body := fmt.Sprintf(`{"model":"public","stream":%t,"prompt_cache_key":"same-session","messages":[{"role":"user","content":"hello"}]}`, stream)
			serveAffinityRequest(t, engine, body)
			// A newly eligible sibling sorts before the learned model. Reusing
			// only the credential incorrectly chooses this sibling on turn two.
			publishAffinityModelRoutes(t, h, affinityRouteModel("codebuddy/deepseek-v4.1-flash", "e000000000001"), bound)
			h.affinityCache = affinity.NewCache() // prove durable route recovery too
			serveAffinityRequest(t, engine, body)

			inputs := forwarder.inputs
			if stream {
				inputs = forwarder.streamInputs
			}
			assertAffinityUpstreamModels(t, inputs, []string{bound.ID, bound.ID})
			assertAffinityHits(t, sink.snapshot(), []bool{false, true})
			assertLearnedAffinityRoute(t, store, 1, bound.EntryID, bound.ID)
		})
	}
}

func assertLearnedAffinityRoute(t *testing.T, store *affinityStoreStub, credentialID uint, entryID, model string) {
	t.Helper()
	targets := store.targets()
	if len(targets) != 1 {
		t.Fatalf("bindings = %#v, want one", targets)
	}
	for _, target := range targets {
		if target.CredentialID != credentialID || target.EntryID != entryID || target.UpstreamModelID != model {
			t.Fatalf("binding = %#v, want credential %d entry %q model %q", target, credentialID, entryID, model)
		}
	}
}

func TestHandlerSiblingModelTakesOverHealthBypassedRoute(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(3)}
	h, _, registry := newHandlerForTest(t, forwarder, "sk-one")
	store := &affinityStoreStub{}
	h.affinityStore = store
	sink := &recordingRequestLogSink{}
	h.requestLogSink = sink
	useAffinityRandomValues(h, 0)
	bound := affinityRouteModel("bound-model", "e000000000002")
	sibling := affinityRouteModel("sibling-model", "e000000000001")
	publishAffinityModelRoutes(t, h, bound)
	engine := newAffinityTestEngine(t, h)
	body := `{"model":"public","prompt_cache_key":"same-session","messages":[{"role":"user","content":"hello"}]}`
	serveAffinityRequest(t, engine, body)
	publishAffinityModelRoutes(t, h, sibling, bound)
	registry.SetEntryCooldown(state.RouteEntryKey{GroupID: 1, EntryID: bound.EntryID}, time.Now().Add(time.Hour))
	serveAffinityRequest(t, engine, body)
	serveAffinityRequest(t, engine, body)
	assertAffinityUpstreamModels(t, forwarder.inputs, []string{bound.ID, sibling.ID, sibling.ID})
	assertAffinityHits(t, sink.snapshot(), []bool{false, false, true})
	assertLearnedAffinityRoute(t, store, 1, sibling.EntryID, sibling.ID)
}

func TestHandlerRelearnsAfterBoundRouteRemoved(t *testing.T) {
	for _, keepEntry := range []bool{false, true} {
		t.Run(fmt.Sprintf("same-entry=%t", keepEntry), func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(3)}
			h, _, _ := newHandlerForTest(t, forwarder, "sk-one")
			store := &affinityStoreStub{}
			h.affinityStore = store
			sink := &recordingRequestLogSink{}
			h.requestLogSink = sink
			useAffinityRandomValues(h, 0)
			bound := affinityRouteModel("old-model", "e000000000002")
			publishAffinityModelRoutes(t, h, bound)
			engine := newAffinityTestEngine(t, h)
			body := `{"model":"public","prompt_cache_key":"changed-route","messages":[{"role":"user","content":"hello"}]}`
			serveAffinityRequest(t, engine, body)
			replacement := affinityRouteModel("new-model", "e000000000001")
			if keepEntry {
				replacement.EntryID = bound.EntryID
			}
			publishAffinityModelRoutes(t, h, replacement)
			serveAffinityRequest(t, engine, body)
			serveAffinityRequest(t, engine, body)
			assertAffinityUpstreamModels(t, forwarder.inputs, []string{bound.ID, replacement.ID, replacement.ID})
			assertAffinityHits(t, sink.snapshot(), []bool{false, false, true})
			assertLearnedAffinityRoute(t, store, 1, replacement.EntryID, replacement.ID)
		})
	}
}

func TestRequestRecorderAffinityHitRequiresRouteMatch(t *testing.T) {
	bound := affinity.Target{GroupID: 1, CredentialID: 1, IdentityGeneration: 1, EntryID: "bound", UpstreamModelID: "model"}
	for _, test := range []struct {
		entry, model string
		want         bool
	}{{"bound", "model", true}, {"other", "model", false}, {"bound", "other", false}} {
		t.Run(test.entry+"/"+test.model, func(t *testing.T) {
			r := &requestRecorder{resolvedAffinityCredentialID: 1, resolvedAffinityTarget: bound}
			r.observeAffinityHitComparison(scheduler.Selection{GroupID: 1, CredentialID: 1, EntryID: test.entry, UpstreamModelID: &test.model})
			if r.affinityHit != test.want || !r.affinityHitAssignmentReached {
				t.Fatalf("hit=%t assignment=%t, want %t/true", r.affinityHit, r.affinityHitAssignmentReached, test.want)
			}
		})
	}
}

func TestWebsocketReusesBoundUpstreamModelOfSameCredential(t *testing.T) {
	h, engine, sink := newAffinityWebsocketFixture(t, "http://127.0.0.1:1/v1")
	store := &affinityStoreStub{}

	h.affinityStore = store
	useAffinityRandomValues(h, 0)
	bound := affinityRouteModel("zen-model", "e000000000002")
	publishAffinityModelRoutes(t, h, bound)
	var opened []string
	h.forwarder = websocketScriptForwarder{AttemptForwarder: h.forwarder, open: func(_ context.Context, in ForwardInput) (execution.WebsocketSession, execution.WebsocketResult) {
		opened = append(opened, in.UpstreamModelID)
		session := &websocketScriptSession{done: make(chan struct{})}
		session.turn = func(ctx context.Context, _ []byte, emit func(context.Context, []byte) error) execution.WebsocketResult {
			if err := emit(ctx, websocketCompleted(fmt.Sprintf("route-%d", len(opened)), "")); err != nil {
				return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled}}
			}
			return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent}
		}
		return session, execution.WebsocketResult{DispatchState: execution.DispatchNotSent}
	}}
	server := httptest.NewServer(engine)
	defer server.Close()
	for i := range 2 {
		conn := dialGatewayWebsocket(t, server.URL)
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","prompt_cache_key":"ws-route","store":false}`)); err != nil {
			t.Fatal(err)
		}
		if _, _, err := conn.ReadMessage(); err != nil {
			t.Fatal(err)
		}
		waitWebsocketLogs(t, sink, i+1)
		_ = conn.Close()
		if i == 0 {
			publishAffinityModelRoutes(t, h, affinityRouteModel("a-sibling-model", "e000000000001"), bound)
		}
	}
	if len(opened) != 2 || opened[0] != bound.ID || opened[1] != bound.ID {
		t.Fatalf("upstream models = %#v, want bound model twice", opened)
	}
	assertAffinityHits(t, sink.snapshot(), []bool{false, true})
	assertLearnedAffinityRoute(t, store, 1, bound.EntryID, bound.ID)
}

func TestAffinityRouteWhitespaceAndAliasChange(t *testing.T) {
	for _, rename := range []bool{false, true} {
		t.Run(fmt.Sprintf("rename=%t", rename), func(t *testing.T) {
			f := &scriptedForwarder{results: successfulAffinityResults(3)}
			h, _, _ := newHandlerForTest(t, f, "sk-one")
			store := &affinityStoreStub{}
			h.affinityStore = store
			useAffinityRandomValues(h, 0)
			bound := affinityRouteModel(" zen-model ", "e000000000002")
			publishAffinityModelRoutes(t, h, bound)
			engine := newAffinityTestEngine(t, h)
			body := `{"model":"public","prompt_cache_key":"route-edit","messages":[{"role":"user","content":"hello"}]}`
			serveAffinityRequest(t, engine, body)
			sibling := affinityRouteModel("a-sibling", "e000000000001")
			if rename {
				bound.Alias = "different-alias"
			}
			publishAffinityModelRoutes(t, h, sibling, bound)
			serveAffinityRequest(t, engine, body)
			serveAffinityRequest(t, engine, body)
			want := "zen-model"
			entry := bound.EntryID
			if rename {
				want, entry = sibling.ID, sibling.EntryID
			}
			assertAffinityUpstreamModels(t, f.inputs, []string{"zen-model", want, want})
			assertLearnedAffinityRoute(t, store, 1, entry, want)
		})
	}
}

func TestAffinityUnresolvedTargetDoesNotReportHit(t *testing.T) {
	r := &requestRecorder{resolvedAffinityTarget: affinity.Target{GroupID: 1, CredentialID: 1, IdentityGeneration: 1, EntryID: "entry", UpstreamModelID: "model"}}
	r.observeAffinityHitComparison(scheduler.Selection{GroupID: 1, CredentialID: 1, EntryID: "entry", UpstreamModelID: new("model")})
	if r.affinityHit {
		t.Fatal("unresolved target reported affinity hit")
	}
}
