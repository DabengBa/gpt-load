package scheduler

import (
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

// R1: an expired model-entry blacklist is released by local maintenance and the
// entry becomes schedulable again for inspection.
func TestInspectEntryReleaseMakesEntryRoutableAgain(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ID: 1, Name: "group", ChannelID: channel.OpenAI, ConnectionType: "api_key",
		Params: []byte(`{}`), Enabled: true,
		Models: []state.ModelConfig{{
			ID: "gpt-4o", Alias: "gpt-4o", EntryID: "e000000000001", Weight: intPointer(100),
		}},
	}})
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 11, GroupID: 1, Version: 1, IdentityGeneration: 1,
		Fingerprint: "fingerprint", EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	key := state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}
	if _, changed := registry.SetEntryBlacklistedWithChange(key); !changed {
		t.Fatal("SetEntryBlacklistedWithChange() changed = false")
	}
	if !registry.SetEntryBlacklistReleaseAt(1, "e000000000001", now.Add(-time.Second)) {
		t.Fatal("SetEntryBlacklistReleaseAt() = false")
	}

	credentials := registry.Snapshot()
	entryRuntime := registry.EntryRuntimeSnapshot()
	blacklisted, err := InspectWithEntryRuntime(snapshot, credentials, entryRuntime, inspectQuery(), now)
	if err != nil {
		t.Fatal(err)
	}
	if blacklisted.Routable || len(blacklisted.Groups) != 1 ||
		blacklisted.Groups[0].Reason != ReasonEntryBlacklisted {
		t.Fatalf("inspection = %#v, want entry blacklisted", blacklisted)
	}

	if _, entries := registry.ReleaseExpiredBlacklists(now); entries != 1 {
		t.Fatalf("ReleaseExpiredBlacklists() entries = %d, want 1", entries)
	}
	credentials = registry.Snapshot()
	entryRuntime = registry.EntryRuntimeSnapshot()
	released, err := InspectWithEntryRuntime(snapshot, credentials, entryRuntime, inspectQuery(), now)
	if err != nil {
		t.Fatal(err)
	}
	if !released.Routable || len(released.Groups) != 1 || !released.Groups[0].Routable {
		t.Fatalf("inspection = %#v, want routable entry after release", released)
	}
}
