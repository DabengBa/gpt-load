package gateway

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"gpt-load/internal/affinity"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/storage"
)

func TestAffinityHTTPInFlightQuarantineDoesNotLearn(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			f := &scriptedForwarder{results: successfulAffinityResults(1), streamResults: []UpstreamResult{performanceTestStreamResult(health.Feedback{})}}
			h, _, registry := newHandlerForTest(t, f, "A")
			store := &affinityStoreStub{}
			h.affinityStore = store
			f.onCall = func(int) { registry.SetBlacklisted(1) }
			f.onStreamCall = func(_ int, writer http.ResponseWriter) {
				registry.SetEntryCooldownForEntry(1, "e000000000001", time.Now().Add(time.Hour))
				writer.WriteHeader(http.StatusOK)
				_, _ = writer.Write([]byte("data: {\"ok\":true}\n\n"))
			}
			serveAffinityRequest(t, newAffinityTestEngine(t, h), fmt.Sprintf(`{"model":"gpt-4o","stream":%v,"messages":[{"role":"user","content":"quarantine"}]}`, stream))
			if got := store.targets(); len(got) != 0 {
				t.Fatalf("in-flight quarantine learned durable binding: %#v", got)
			}
		})
	}
}

func TestWebsocketAffinityFinalFeedbackAndQuarantine(t *testing.T) {
	for _, outcome := range []string{"faulty feedback", "credential quarantine", "entry quarantine"} {
		t.Run(outcome, func(t *testing.T) {
			h, engine, sink := newAffinityWebsocketFixture(t, "http://127.0.0.1:1/v1")
			store := &affinityStoreStub{}
			h.affinityStore = store
			clock := &providerFeedbackTestClock{now: time.Now()}
			h.requestNow = clock.Now
			h.forwarder = websocketScriptForwarder{AttemptForwarder: h.forwarder, open: func(_ context.Context, input ForwardInput) (execution.WebsocketSession, execution.WebsocketResult) {
				session := &websocketScriptSession{done: make(chan struct{})}
				session.turn = func(ctx context.Context, _ []byte, emit func(context.Context, []byte) error) execution.WebsocketResult {
					if outcome == "faulty feedback" {
						clock.Advance(31 * time.Second)
					}
					if err := emit(ctx, websocketCompleted("final-feedback", "")); err != nil {
						return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled}}
					}
					switch outcome {
					case "credential quarantine":
						h.registry.SetCooldownWithChange(input.Credential.ID, time.Now().Add(time.Hour))
					case "entry quarantine":
						for _, model := range input.Group.Models {
							h.registry.(entryRuntimeRegistry).SetEntryCooldownForEntry(input.Group.ID, model.EntryID, time.Now().Add(time.Hour))
						}
					}
					return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent}
				}
				return session, execution.WebsocketResult{DispatchState: execution.DispatchNotSent}
			}}
			server := httptest.NewServer(engine)
			defer server.Close()
			conn := dialGatewayWebsocket(t, server.URL)
			defer conn.Close()
			if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","prompt_cache_key":"final-feedback","store":false}`)); err != nil {
				t.Fatal(err)
			}
			if _, _, err := conn.ReadMessage(); err != nil {
				t.Fatal(err)
			}
			logs := waitWebsocketLogs(t, sink, 1)
			if outcome == "faulty feedback" && logs[0].Attempts[0].Feedback.Status != health.FeedbackStatusFaulty {
				t.Fatalf("feedback = %#v, want faulty", logs[0].Attempts[0].Feedback)
			}
			if got := store.targets(); len(got) != 0 {
				t.Fatalf("%s learned durable binding: %#v", outcome, got)
			}
		})
	}
}

func TestAffinityHealthTakeoverHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "unary", true: "stream"}[stream], func(t *testing.T) {
			f := &scriptedForwarder{results: successfulAffinityResults(2), streamResults: []UpstreamResult{performanceTestStreamResult(health.Feedback{}), performanceTestStreamResult(health.Feedback{})}}
			h, _, r := newHandlerForTest(t, f, "sk-one", "sk-two")
			store := &affinityStoreStub{}
			h.affinityStore = store
			useAffinityRandomValues(h, 0)
			body := `{"model":"gpt-4o","messages":[{"role":"user","content":"takeover"}]}`
			if stream {
				body = `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"takeover"}]}`
			}
			engine := newAffinityTestEngine(t, h)
			serveAffinityRequest(t, engine, body)
			if got := store.targets(); len(got) != 1 {
				t.Fatalf("initial durable bindings = %#v, want one", got)
			}
			if exists, _ := r.SetEntryCooldownForEntry(1, "e000000000001", time.Now().Add(time.Hour)); !exists {
				t.Fatal("bound entry missing")
			}
			// Recovery after fallback selection must not revoke the captured bypass.
			recovered := false
			recoverBound := func() {
				recovered = r.RecoverEntryForEntry(1, "e000000000001")
			}
			f.onCall = func(int) { recoverBound() }
			f.onStreamCall = func(int, http.ResponseWriter) { recoverBound() }
			serveAffinityRequest(t, engine, body)
			if !recovered {
				t.Fatal("bound entry did not recover during fallback")
			}
			targets := store.targets()
			if len(targets) != 1 {
				t.Fatalf("HTTP bindings = %#v, want one", targets)
			}
			for _, target := range targets {
				if target.CredentialID != 2 {
					t.Fatalf("takeover target = %#v, want 2", target)
				}

			}
		})
	}
}

func TestAffinityFaultyStreamDoesNotLearn(t *testing.T) {
	f := performanceTestStreamForwarder(performanceTestStreamResult(health.Feedback{Status: health.FeedbackStatusFaulty, Reason: "first_response_slow"}))
	h, _, _ := newHandlerForTest(t, f, "sk-one")
	store := &affinityStoreStub{}
	h.affinityStore = store
	serveAffinityRequest(t, newAffinityTestEngine(t, h), `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"fault"}]}`)
	if len(store.targets()) != 0 {
		t.Fatal("faulty delivered response learned binding")
	}
}

func TestAffinityBoundMissingObservationCannotWrite(t *testing.T) {
	h, m, r := newHandlerForTest(t, &scriptedForwarder{}, "sk-one", "sk-two")
	h.affinityCache.Configure(m.Current().Revision, 1, time.Hour)
	key := affinity.Key("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	ref, _ := r.CredentialRef(2)
	request := requestAffinity{key: key, observation: h.affinityCache.Lookup(key), boundTarget: affinity.Target{GroupID: 1, CredentialID: 1, IdentityGeneration: 1, EntryID: "e000000000001"}, boundAttempted: true, boundProviderFailed: true}
	h.recordAffinitySuccess(t.Context(), request, affinityTakeoverSelection(h, 2), ref)
	if h.affinityCache.Lookup(key).Found() {
		t.Fatal("bound target with missing observation inserted binding")
	}
}

func affinityTakeoverSelection(h *Handler, id uint) scheduler.Selection {
	return scheduler.Selection{CredentialID: id, GroupID: id, EntryID: "e000000000001", Group: h.manager.Current().Groups[id]}
}

func TestWebsocketAffinityWaitsForFinalResult(t *testing.T) {
	h, engine, sink := newAffinityWebsocketFixture(t, "http://127.0.0.1:1/v1")
	store := &affinityStoreStub{}

	h.affinityStore = store
	emitted, release := make(chan struct{}), make(chan struct{})

	h.forwarder = websocketScriptForwarder{AttemptForwarder: h.forwarder, open: func(context.Context, ForwardInput) (execution.WebsocketSession, execution.WebsocketResult) {
		session := &websocketScriptSession{done: make(chan struct{})}
		session.turn = func(ctx context.Context, _ []byte, emit func(context.Context, []byte) error) execution.WebsocketResult {
			if err := emit(ctx, websocketCompleted("final-gate", "")); err != nil {
				t.Error(err)
			}
			close(emitted)
			<-release
			return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindTransport, OriginHint: execution.ErrorOriginUpstream, Code: "late_failure"}}
		}
		return session, execution.WebsocketResult{DispatchState: execution.DispatchNotSent}
	}}
	server := httptest.NewServer(engine)
	defer server.Close()
	conn := dialGatewayWebsocket(t, server.URL)
	defer conn.Close()
	if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","prompt_cache_key":"final-gate","store":false}`)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := conn.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	<-emitted
	if len(store.targets()) != 0 {
		t.Error("affinity recorded inside terminal callback before final result")
	}
	close(release)
	waitWebsocketLogs(t, sink, 1)
	if len(store.targets()) != 0 {
		t.Error("late execution failure learned affinity")
	}
}

func TestAffinityTakeoverConcurrentCBeforeB(t *testing.T) {
	h, m, r := newHandlerForTest(t, &scriptedForwarder{}, "A", "B", "C")
	db, err := storage.Open(filepath.Join(t.TempDir(), "takeover.db"))
	if err != nil {
		t.Fatal(err)
	}
	sql, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sql.Close() })
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	h.affinityStore = storage.NewAffinityStore(db)
	h.affinityCache.Configure(m.Current().Revision, 10, time.Hour)
	key := affinity.Key("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	a := affinity.Target{GroupID: 1, CredentialID: 1, IdentityGeneration: 1, EntryID: "e000000000001"}
	h.affinityCache.RecordSuccess(key, h.affinityCache.Lookup(key), a)
	observed := h.affinityCache.Lookup(key)
	request := requestAffinity{key: key, observation: observed, boundTarget: a, preferredHealthBypassed: true, policy: affinity.DurablePolicy{Capacity: 10, TTL: time.Hour}}
	bReady, cReady, bRelease, cRelease := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	bDone, cDone := make(chan struct{}), make(chan struct{})
	for _, worker := range []struct {
		id                   uint
		ready, release, done chan struct{}
	}{{2, bReady, bRelease, bDone}, {3, cReady, cRelease, cDone}} {
		go func() {
			ref, _ := r.CredentialRef(worker.id)
			close(worker.ready)
			<-worker.release
			h.recordAffinitySuccess(t.Context(), request, affinityTakeoverSelection(h, worker.id), ref)
			close(worker.done)
		}()
	}
	<-bReady
	<-cReady
	close(cRelease)
	<-cDone
	close(bRelease)
	<-bDone
	if got := h.affinityCache.Lookup(key).Target; got.CredentialID != 3 {
		t.Fatalf("cache winner = %#v, want C", got)
	}
	got, found, err := h.affinityStore.Lookup(t.Context(), key, request.policy)
	if err != nil || !found || got.CredentialID != 3 {
		t.Fatalf("durable winner = %#v/%v/%v, want C", got, found, err)
	}
}

func TestAffinityCompletionSelectedHealth(t *testing.T) {
	for _, name := range []string{"healthy", "cooldown", "blacklist", "entry cooldown", "entry blacklist", "identity"} {
		t.Run(name, func(t *testing.T) {
			h, m, r := newHandlerForTest(t, &scriptedForwarder{}, "A", "B")
			h.affinityCache.Configure(m.Current().Revision, 10, time.Hour)
			key := affinity.Key("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
			a := affinity.Target{GroupID: 1, CredentialID: 1, IdentityGeneration: 1, EntryID: "e000000000001"}
			h.affinityCache.RecordSuccess(key, h.affinityCache.Lookup(key), a)
			request := requestAffinity{key: key, observation: h.affinityCache.Lookup(key), boundTarget: a, preferredHealthBypassed: true}
			ref, _ := r.CredentialRef(2)
			switch name {
			case "cooldown":
				r.SetCooldown(2, time.Now().Add(time.Hour))
			case "blacklist":
				r.SetBlacklisted(2)
			case "entry cooldown":
				r.SetEntryCooldownForEntry(2, "e000000000001", time.Now().Add(time.Hour))
			case "entry blacklist":
				r.SetEntryBlacklistedForEntry(2, "e000000000001")
			case "identity":
				ref.IdentityGeneration++
			}
			h.recordAffinitySuccess(t.Context(), request, affinityTakeoverSelection(h, 2), ref)
			want := uint(1)
			if name == "healthy" {
				want = 2
			}
			if got := h.affinityCache.Lookup(key).Target; got.CredentialID != want {
				t.Fatalf("target = %#v, want %d", got, want)
			}
		})
	}
}

func TestWebsocketAffinityHealthTakeover(t *testing.T) {
	h, engine, sink := newAffinityWebsocketFixture(t, "http://127.0.0.1:1/v1")
	store := &affinityStoreStub{}
	h.affinityStore = store
	useAffinityRandomValues(h, 0)
	h.forwarder = websocketScriptForwarder{AttemptForwarder: h.forwarder, open: func(_ context.Context, in ForwardInput) (execution.WebsocketSession, execution.WebsocketResult) {
		session := &websocketScriptSession{done: make(chan struct{})}
		session.turn = func(ctx context.Context, _ []byte, emit func(context.Context, []byte) error) execution.WebsocketResult {
			err := emit(ctx, websocketCompleted(fmt.Sprintf("takeover-%d", in.Credential.ID), ""))
			if err != nil {
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
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","prompt_cache_key":"takeover","store":false}`)); err != nil {
			t.Fatal(err)
		}
		if _, _, err := conn.ReadMessage(); err != nil {
			t.Fatal(err)
		}
		waitWebsocketLogs(t, sink, i+1)
		_ = conn.Close()
		if i == 0 {
			h.registry.SetCooldownWithChange(1, time.Now().Add(time.Hour))
		}
	}
	targets := store.targets()
	if len(targets) != 1 {
		t.Fatalf("WS bindings = %#v, want one", targets)
	}
	for _, target := range targets {
		if target.CredentialID != 2 {
			t.Fatalf("WS target = %#v, want 2", target)
		}
	}
}
