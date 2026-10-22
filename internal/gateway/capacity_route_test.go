package gateway

import (
	"testing"

	"gpt-load/internal/affinity"
	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

// R3/R4: a bound route whose snapshot entry disappeared is not the resolved
// route, so it gets no capacity overflow even though the credential matches.
// This is the route identity that keeps the binding from buying capacity it no
// longer owns.
func TestScheduleCapacityNoOverflowForStaleBoundEntry(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
	h, _, _ := newHandlerForTest(t, forwarder, "sk-one")
	group := affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 1, 1)
	group.Models[0].EntryID = "e000000000001"
	group.Models[0].ConcurrencyLimit = 1
	input := affinityBucketRuntimeInput(h, config.Settings{}, group)
	input.Credentials = input.Credentials[:1]
	input.AccessKeys = []state.AccessKeyConfig{{
		ID: 1, Name: "client", KeyHash: h.encryption.Hash("gl-client"), Status: state.AccessKeyStatusActive,
	}}
	if _, err := h.manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	// Fill the single ordinary slot, so any further admission can only come
	// from the bound-route overflow rule.
	key := state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}
	occupied, ok := h.manager.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("occupy")
	}
	defer occupied.Release()

	selection := scheduler.Selection{
		CredentialID: 1, GroupID: 1, EntryID: key.EntryID,
		UpstreamModelID: new("gpt-4o"),
		Group:           h.manager.Current().Groups[1],
	}
	ref, _ := h.registry.CredentialRef(1)
	sameCredentialOtherRoute := affinity.Target{
		GroupID: 1, CredentialID: 1, IdentityGeneration: 1,
		EntryID: "e000000000002", UpstreamModelID: "gpt-4o",
	}
	if _, admitted := h.acquireCapacity(selection, ref, requestAffinity{boundTarget: sameCredentialOtherRoute, preferredCredentialID: 1, preferredIdentityGeneration: 1}); admitted {
		t.Fatal("stale bound route admitted overflow capacity")
	}

	bound := affinity.Target{
		GroupID: 1, CredentialID: 1, IdentityGeneration: 1,
		EntryID: key.EntryID, UpstreamModelID: "gpt-4o",
	}
	reservation, admitted := h.acquireCapacity(selection, ref, requestAffinity{
		boundTarget: bound, preferredCredentialID: 1, preferredIdentityGeneration: 1,
	})
	if !admitted {
		t.Fatal("exact bound route denied overflow capacity")
	}
	reservation.Release()
}
