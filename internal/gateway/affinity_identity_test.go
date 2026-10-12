package gateway

import (
	"math/rand"
	"testing"
	"time"
)

func TestAffinityReplacementBeforeHealthObservation(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, _, registry := newHandlerForTest(t, forwarder, "A", "B")
	store := &affinityStoreStub{}
	handler.affinityStore = store
	useAffinityRandomValues(handler, 0)
	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"identity-race"}]}`
	serveAffinityResponse(t, engine, body)
	initial := store.targets()
	if len(initial) != 1 {
		t.Fatalf("initial bindings = %#v, want one", initial)
	}
	handler.newRandom = func() *rand.Rand {
		entries, err := registry.SnapshotGroupCredentialEntriesExact(1, []uint{1})
		if err != nil {
			t.Fatal(err)
		}
		entries[0].IdentityGeneration++
		entries[0].Version++
		entries[0].CooldownUntil = time.Now().Add(time.Hour)
		if err := registry.RestoreGroupCredentialEntriesExact(1, entries); err != nil {
			t.Fatal(err)
		}
		return rand.New(affinityFixedRandSource{})
	}
	serveAffinityResponse(t, engine, body)
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"A", "B"})
	for key, before := range initial {
		if got := store.targets()[key]; got != before {
			t.Fatalf("replacement health migrated old binding: got %#v, want %#v", got, before)
		}
		if got := handler.affinityCache.Lookup(key).Target; got != before {
			t.Fatalf("cache binding = %#v, want %#v", got, before)
		}
	}
}
