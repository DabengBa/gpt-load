package scheduler

import (
	"math/rand"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

func TestCapacitySkipDoesNotTrySiblingEntry(t *testing.T) {
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{ID: 1, Name: "one", ChannelID: channel.OpenAI, Params: []byte(`{}`), Enabled: true,
		Models: []state.ModelConfig{{ID: "up", Alias: "public", EntryID: "e000000000001"}, {ID: "other", Alias: "public", EntryID: "e000000000002"}}}})
	iterator := New(snapshot, twoGroupRegistry(t), preferredHealthQuery(0), rand.New(zeroRandSource{}))
	// Compile currently rejects duplicate external/upstream pairs. Exercise the
	// retry dedup boundary directly with two distinct entry identities anyway.
	for _, byTier := range iterator.regular.tierTargetsByMode {
		for tier, targets := range byTier {
			for i := range targets {
				targets[i].target.UpstreamModelID = "up"
			}
			byTier[tier] = targets
		}
	}
	first, err := iterator.NextCandidate()
	if err != nil {
		t.Fatal(err)
	}
	iterator.SkipEntry(state.RouteEntryKey{GroupID: first.GroupID, EntryID: first.EntryID})
	second, err := iterator.NextCandidate()
	if err != nil {
		t.Fatalf("capacity skip excluded same credential/upstream sibling: %v", err)
	}
	if first.EntryID == second.EntryID {
		t.Fatal("full entry selected again")
	}
	iterator.MarkAttempt(second)
	if _, err := iterator.NextCandidate(); err == nil {
		t.Fatal("actual attempt did not deduplicate pair")
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("capacity changed health observation")
	}
}
