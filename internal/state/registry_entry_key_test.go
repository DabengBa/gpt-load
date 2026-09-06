package state

import (
	"testing"
	"time"
)

// Design §2/§C5: the runtime identity of a route entry is (GroupID, EntryID).
// Two entries of the same group that point at the same upstream model (alias
// A/B) must keep fully independent cooldown, failure-count, and blacklist
// state.
func TestEntryRuntimeKeyIsolatesSameGroupSameUpstreamEntries(t *testing.T) {
	registry := NewCredentialRegistry()
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)
	entryA := RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}
	entryB := RouteEntryKey{GroupID: 1, EntryID: "e000000000002"}

	if count, ok := registry.IncrEntryFailureForEntry(1, "e000000000001"); !ok || count != 1 {
		t.Fatalf("IncrEntryFailureForEntry(A) = %d/%t, want 1/true", count, ok)
	}
	if count, ok := registry.IncrEntryFailureForEntry(1, "e000000000001"); !ok || count != 2 {
		t.Fatalf("second IncrEntryFailureForEntry(A) = %d/%t, want 2/true", count, ok)
	}
	if view, ok := registry.EntryRuntime(entryB, now); ok || view.FailureCount != 0 {
		t.Fatalf("entry B gained runtime state from entry A: %#v/%t", view, ok)
	}

	if exists, changed := registry.SetEntryCooldownForEntry(1, "e000000000001", now.Add(time.Minute)); !exists || !changed {
		t.Fatalf("SetEntryCooldownForEntry(A) = %t/%t, want true/true", exists, changed)
	}
	if view, ok := registry.EntryRuntime(entryB, now); ok || view.RuntimeState(now) != EntryRuntimeAvailable {
		t.Fatalf("entry B cooled down together with entry A: %#v/%t", view, ok)
	}

	if _, changed := registry.SetEntryBlacklistedForEntry(1, "e000000000001"); !changed {
		t.Fatal("SetEntryBlacklistedForEntry(A) changed = false")
	}
	if viewB, ok := registry.EntryRuntime(entryB, now); ok &&
		(viewB.RuntimeState(now) != EntryRuntimeAvailable || viewB.FailureCount != 0) {
		t.Fatalf("entry B blacklisted together with entry A: %#v/%t", viewB, ok)
	}
	viewA, ok := registry.EntryRuntime(entryA, now)
	if !ok || viewA.RuntimeState(now) != EntryRuntimeBlacklisted || viewA.FailureCount != 2 {
		t.Fatalf("entry A runtime = %#v/%t, want blacklisted with 2 failures", viewA, ok)
	}
}

func TestEntryRuntimeKeySeparatesSameEntryIDAcrossGroups(t *testing.T) {
	registry := NewCredentialRegistry()
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)

	if _, ok := registry.IncrEntryFailureForEntry(2, "e000000000001"); !ok {
		t.Fatal("IncrEntryFailureForEntry(group 2) = false")
	}
	view, ok := registry.EntryRuntime(RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now)
	if ok || view.FailureCount != 0 {
		t.Fatalf("group 1 entry inherited group 2 state: %#v/%t", view, ok)
	}
}

func TestEntryRuntimeForEntryRejectsInvalidKeys(t *testing.T) {
	registry := NewCredentialRegistry()
	now := time.Date(2026, time.September, 7, 10, 0, 0, 0, time.UTC)

	if _, ok := registry.IncrEntryFailureForEntry(0, "e000000000001"); ok {
		t.Fatal("IncrEntryFailureForEntry(group 0) = true, want false")
	}
	if _, ok := registry.IncrEntryFailureForEntry(1, "  "); ok {
		t.Fatal("IncrEntryFailureForEntry(blank entry id) = true, want false")
	}
	if registry.ClearEntryFailureForEntry(0, "e000000000001") {
		t.Fatal("ClearEntryFailureForEntry(group 0) = true, want false")
	}
	if _, changed := registry.SetEntryCooldownForEntry(1, "  ", now.Add(time.Minute)); changed {
		t.Fatal("SetEntryCooldownForEntry(blank entry id) changed = true, want false")
	}
	if _, changed := registry.SetEntryBlacklistedForEntry(0, "e000000000001"); changed {
		t.Fatal("SetEntryBlacklistedForEntry(group 0) changed = true, want false")
	}
	if registry.RecoverEntryForEntry(0, "e000000000001") {
		t.Fatal("RecoverEntryForEntry(group 0) = true, want false")
	}
	if views := registry.EntryRuntimeSnapshot(); len(views) != 0 {
		t.Fatalf("EntryRuntimeSnapshot() = %#v, want empty", views)
	}
}

// Design §2.2-3: the entry identity, not the upstream model name, owns the
// runtime state, so whitespace around an entry id must not fork a second
// state bucket.
func TestEntryRuntimeKeysTrimEntryIDWhitespace(t *testing.T) {
	registry := NewCredentialRegistry()

	if count, ok := registry.IncrEntryFailureForEntry(1, "e000000000001"); !ok || count != 1 {
		t.Fatalf("IncrEntryFailureForEntry() = %d/%t, want 1/true", count, ok)
	}
	view, ok := registry.EntryRuntime(RouteEntryKey{GroupID: 1, EntryID: "e000000000001 "}, time.Time{})
	if !ok || view.FailureCount != 1 {
		t.Fatalf("trimmed lookup = %#v/%t, want the same state bucket", view, ok)
	}
}
