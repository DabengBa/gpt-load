package state

import (
	"sync"
	"testing"
	"time"
)

func releaseTestRegistry(t *testing.T, entry CredentialEntry) *CredentialRegistry {
	t.Helper()
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{entry}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	return registry
}

func releaseTestEntry(id uint) CredentialEntry {
	return CredentialEntry{
		ID: id, GroupID: 1, Version: 1, IdentityGeneration: 1,
		Fingerprint: "fingerprint", EncryptedValue: "cipher",
	}
}

// R1: a blacklist whose local release deadline has not passed stays blacklisted
// and unschedulable.
func TestReleaseExpiredBlacklistsKeepsFutureDeadlineBlacklisted(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := releaseTestRegistry(t, releaseTestEntry(1))
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("SetBlacklistedWithChange() changed = false")
	}
	if !registry.SetBlacklistReleaseAt(1, now.Add(time.Hour)) {
		t.Fatal("SetBlacklistReleaseAt() = false")
	}

	credentials, entries := registry.ReleaseExpiredBlacklists(now)
	if credentials != 0 || entries != 0 {
		t.Fatalf("ReleaseExpiredBlacklists() = %d/%d, want 0/0", credentials, entries)
	}
	view := registry.Snapshot()[0]
	if !view.Blacklisted || view.BlacklistReleaseAt.IsZero() {
		t.Fatalf("view = %#v, want still blacklisted with deadline", view)
	}
	if candidates := registry.CollectCredentialCandidates([]uint{1}, nil, now); len(candidates) != 0 {
		t.Fatalf("candidates = %#v, want none before release", candidates)
	}
}

// R1: an expired ready credential is released locally and becomes schedulable
// again with its blacklist and failure counter cleared.
func TestReleaseExpiredBlacklistsReleasesExpiredReadyCredential(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := releaseTestRegistry(t, releaseTestEntry(1))
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("SetBlacklistedWithChange() changed = false")
	}
	if _, ok := registry.IncrFailure(1); !ok {
		t.Fatal("IncrFailure() = false")
	}
	if !registry.SetBlacklistReleaseAt(1, now.Add(-time.Second)) {
		t.Fatal("SetBlacklistReleaseAt() = false")
	}

	credentials, entries := registry.ReleaseExpiredBlacklists(now)
	if credentials != 1 || entries != 0 {
		t.Fatalf("ReleaseExpiredBlacklists() = %d/%d, want 1/0", credentials, entries)
	}
	view := registry.Snapshot()[0]
	if view.Blacklisted || view.FailureCount != 0 || !view.BlacklistReleaseAt.IsZero() {
		t.Fatalf("view = %#v, want released with cleared failure count", view)
	}
	candidates := registry.CollectCredentialCandidates([]uint{1}, nil, now)
	if len(candidates) != 1 || candidates[0].ID != 1 {
		t.Fatalf("candidates = %#v, want credential 1 schedulable", candidates)
	}
}

// R1: an expired credential that is not ready stays blacklisted and keeps its
// deadline so a later maintenance run can release it once it is ready.
func TestReleaseExpiredBlacklistsKeepsNonReadyCredentialBlacklisted(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	entry := releaseTestEntry(1)
	entry.AuthState = CredentialAuthStateRefreshing
	registry := releaseTestRegistry(t, entry)
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("SetBlacklistedWithChange() changed = false")
	}
	deadline := now.Add(-time.Second)
	if !registry.SetBlacklistReleaseAt(1, deadline) {
		t.Fatal("SetBlacklistReleaseAt() = false")
	}

	credentials, _ := registry.ReleaseExpiredBlacklists(now)
	if credentials != 0 {
		t.Fatalf("ReleaseExpiredBlacklists() = %d, want 0 for non-ready credential", credentials)
	}
	view := registry.Snapshot()[0]
	if !view.Blacklisted || !view.BlacklistReleaseAt.Equal(deadline) {
		t.Fatalf("view = %#v, want still blacklisted with retained deadline %v", view, deadline)
	}
	if candidates := registry.CollectCredentialCandidates([]uint{1}, nil, now); len(candidates) != 0 {
		t.Fatalf("candidates = %#v, want none for non-ready credential", candidates)
	}
}

// R1: a credential whose deadline already passed while non-ready is released by
// the next maintenance run after it becomes ready.
func TestReleaseExpiredBlacklistsReleasesCredentialThatBecomesReadyLater(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	entry := releaseTestEntry(1)
	entry.AuthState = CredentialAuthStateReauthorizationRequired
	registry := releaseTestRegistry(t, entry)
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("SetBlacklistedWithChange() changed = false")
	}
	if !registry.SetBlacklistReleaseAt(1, now.Add(-time.Minute)) {
		t.Fatal("SetBlacklistReleaseAt() = false")
	}
	if credentials, _ := registry.ReleaseExpiredBlacklists(now); credentials != 0 {
		t.Fatalf("first release = %d, want 0 while not ready", credentials)
	}
	if !registry.SetCredentialAuthState(1, CredentialAuthStateReady) {
		t.Fatal("SetCredentialAuthState() = false")
	}
	if credentials, _ := registry.ReleaseExpiredBlacklists(now); credentials != 1 {
		t.Fatalf("second release = %d, want 1 after ready", credentials)
	}
	if view := registry.Snapshot()[0]; view.Blacklisted {
		t.Fatalf("view = %#v, want released", view)
	}
}

// R1: release increments the credential failure generation so a restore proof
// captured before the release can no longer clear a newer state.
func TestReleaseExpiredBlacklistsInvalidatesStaleRestoreProof(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := releaseTestRegistry(t, releaseTestEntry(1))
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("SetBlacklistedWithChange() changed = false")
	}
	if !registry.SetBlacklistReleaseAt(1, now.Add(-time.Second)) {
		t.Fatal("SetBlacklistReleaseAt() = false")
	}
	staleRef := registry.BlacklistedCredentials()[0]

	if credentials, _ := registry.ReleaseExpiredBlacklists(now); credentials != 1 {
		t.Fatalf("ReleaseExpiredBlacklists() = %d, want 1", credentials)
	}
	if registry.RecoverIfMatch(staleRef) {
		t.Fatal("stale restore proof recovered a released credential")
	}
	// A fresh failure cycle must remain blacklisted against the stale proof.
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("re-blacklist changed = false")
	}
	if registry.RecoverIfMatch(staleRef) {
		t.Fatal("stale restore proof cleared a newer blacklist")
	}
	if !registry.Snapshot()[0].Blacklisted {
		t.Fatal("stale restore proof cleared the current blacklist")
	}
}

// R1: release clears the blacklist but preserves a still-valid Provider
// cooldown deadline.
func TestReleaseExpiredBlacklistsPreservesCredentialCooldown(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := releaseTestRegistry(t, releaseTestEntry(1))
	cooldown := now.Add(30 * time.Minute)
	if _, changed := registry.SetCooldownWithChange(1, cooldown); !changed {
		t.Fatal("SetCooldownWithChange() changed = false")
	}
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("SetBlacklistedWithChange() changed = false")
	}
	if !registry.SetBlacklistReleaseAt(1, now.Add(-time.Second)) {
		t.Fatal("SetBlacklistReleaseAt() = false")
	}

	if credentials, _ := registry.ReleaseExpiredBlacklists(now); credentials != 1 {
		t.Fatalf("ReleaseExpiredBlacklists() = %d, want 1", credentials)
	}
	view := registry.Snapshot()[0]
	if view.Blacklisted {
		t.Fatal("blacklist not released")
	}
	if !view.CooldownUntil.Equal(cooldown) {
		t.Fatalf("cooldown = %v, want preserved %v", view.CooldownUntil, cooldown)
	}
	// The surviving cooldown still governs schedulability.
	if candidates := registry.CollectCredentialCandidates([]uint{1}, nil, now); len(candidates) != 0 {
		t.Fatalf("candidates = %#v, want none while cooldown is active", candidates)
	}
}

// R1: route-entry release is keyed by entry identity, so a sibling entry and the
// credential state are untouched.
func TestReleaseExpiredBlacklistsReleasesExpiredEntryOnly(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := releaseTestRegistry(t, releaseTestEntry(1))
	expired := RouteEntryKey{GroupID: 1, EntryID: "model-a"}
	sibling := RouteEntryKey{GroupID: 1, EntryID: "model-b"}
	if _, changed := registry.SetEntryBlacklistedWithChange(expired); !changed {
		t.Fatal("SetEntryBlacklistedWithChange(expired) changed = false")
	}
	if !registry.SetEntryBlacklistReleaseAt(1, "model-a", now.Add(-time.Second)) {
		t.Fatal("SetEntryBlacklistReleaseAt(expired) = false")
	}
	if _, changed := registry.SetEntryBlacklistedWithChange(sibling); !changed {
		t.Fatal("SetEntryBlacklistedWithChange(sibling) changed = false")
	}
	if !registry.SetEntryBlacklistReleaseAt(1, "model-b", now.Add(time.Hour)) {
		t.Fatal("SetEntryBlacklistReleaseAt(sibling) = false")
	}
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("credential SetBlacklistedWithChange() changed = false")
	}
	if !registry.SetBlacklistReleaseAt(1, now.Add(time.Hour)) {
		t.Fatal("credential SetBlacklistReleaseAt() = false")
	}

	credentials, entries := registry.ReleaseExpiredBlacklists(now)
	if credentials != 0 || entries != 1 {
		t.Fatalf("ReleaseExpiredBlacklists() = %d/%d, want 0/1", credentials, entries)
	}
	released, _ := registry.EntryRuntime(expired, now)
	if released.Blacklisted {
		t.Fatalf("expired entry = %#v, want released", released)
	}
	kept, _ := registry.EntryRuntime(sibling, now)
	if !kept.Blacklisted {
		t.Fatalf("sibling entry = %#v, want still blacklisted", kept)
	}
	if !registry.Snapshot()[0].Blacklisted {
		t.Fatal("entry release changed credential blacklist")
	}
}

// R1: entry release increments the entry failure version so a stale entry view
// is detectable by version mismatch.
func TestReleaseExpiredBlacklistsIncrementsEntryFailureVersion(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := NewCredentialRegistry()
	key := RouteEntryKey{GroupID: 1, EntryID: "model-a"}
	if _, changed := registry.SetEntryBlacklistedWithChange(key); !changed {
		t.Fatal("SetEntryBlacklistedWithChange() changed = false")
	}
	if !registry.SetEntryBlacklistReleaseAt(1, "model-a", now.Add(-time.Second)) {
		t.Fatal("SetEntryBlacklistReleaseAt() = false")
	}
	before, ok := registry.EntryRuntime(key, now)
	if !ok {
		t.Fatal("EntryRuntime() = false")
	}
	if _, entries := registry.ReleaseExpiredBlacklists(now); entries != 1 {
		t.Fatalf("ReleaseExpiredBlacklists() entries = %d, want 1", entries)
	}
	after, ok := registry.EntryRuntime(key, now)
	if !ok {
		t.Fatal("EntryRuntime() after release = false")
	}
	if after.FailureVersion <= before.FailureVersion {
		t.Fatalf("entry failure version = %d, want > %d", after.FailureVersion, before.FailureVersion)
	}
	if after.Blacklisted || after.FailureCount != 0 || !after.BlacklistReleaseAt.IsZero() {
		t.Fatalf("entry after release = %#v, want released with cleared failure count", after)
	}
}

// R1: SetBlacklistReleaseAt never revives a non-blacklisted target and never
// overwrites an already-scheduled deadline (stale writer protection).
func TestSetBlacklistReleaseAtIgnoresUnblacklistedAndExistingDeadline(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := releaseTestRegistry(t, releaseTestEntry(1))
	if registry.SetBlacklistReleaseAt(1, now.Add(time.Hour)) {
		t.Fatal("SetBlacklistReleaseAt() on a non-blacklisted credential = true")
	}
	if _, changed := registry.SetBlacklistedWithChange(1); !changed {
		t.Fatal("SetBlacklistedWithChange() changed = false")
	}
	first := now.Add(time.Hour)
	if !registry.SetBlacklistReleaseAt(1, first) {
		t.Fatal("first SetBlacklistReleaseAt() = false")
	}
	if registry.SetBlacklistReleaseAt(1, now.Add(2*time.Hour)) {
		t.Fatal("SetBlacklistReleaseAt() overwrote an existing deadline")
	}
	if got := registry.Snapshot()[0].BlacklistReleaseAt; !got.Equal(first) {
		t.Fatalf("deadline = %v, want first %v", got, first)
	}
}

// R1: a concurrent release and a new failure/blacklist update must not overwrite
// each other. The entry ends either released or blacklisted with the fresh
// deadline, never blacklisted with the expired deadline.
func TestReleaseExpiredBlacklistsConcurrentWithNewFailure(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	for attempt := 0; attempt < 200; attempt++ {
		registry := NewCredentialRegistry()
		key := RouteEntryKey{GroupID: 1, EntryID: "model-a"}
		if _, changed := registry.SetEntryBlacklistedWithChange(key); !changed {
			t.Fatal("SetEntryBlacklistedWithChange() changed = false")
		}
		if !registry.SetEntryBlacklistReleaseAt(1, "model-a", now.Add(-time.Second)) {
			t.Fatal("SetEntryBlacklistReleaseAt() = false")
		}
		fresh := now.Add(2 * time.Hour)

		var wait sync.WaitGroup
		wait.Add(2)
		go func() {
			defer wait.Done()
			registry.ReleaseExpiredBlacklists(now)
		}()
		go func() {
			defer wait.Done()
			if _, changed := registry.SetEntryBlacklistedWithChange(key); changed {
				registry.SetEntryBlacklistReleaseAt(1, "model-a", fresh)
			}
		}()
		wait.Wait()

		view, _ := registry.EntryRuntime(key, now)
		if view.Blacklisted && !view.BlacklistReleaseAt.After(now) {
			t.Fatalf("attempt %d: blacklisted entry kept expired deadline %v", attempt, view.BlacklistReleaseAt)
		}
	}
}

// The failure mutation is one registry transaction. Whichever side wins the
// lock, a fresh failure must remain represented by a fresh blacklist deadline.
func TestRecordFailureWithBlacklistInterleavesWithReleaseWithoutLoss(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	fresh := now.Add(2 * time.Hour)
	for attempt := 0; attempt < 200; attempt++ {
		registry := releaseTestRegistry(t, releaseTestEntry(1))
		if _, changed := registry.SetBlacklistedWithChange(1); !changed {
			t.Fatal("SetBlacklistedWithChange() changed = false")
		}
		if !registry.SetBlacklistReleaseAt(1, now.Add(-time.Second)) {
			t.Fatal("SetBlacklistReleaseAt() = false")
		}
		before := registry.CaptureActiveCredentialRefs([]uint{1})[0]

		start := make(chan struct{})
		var wait sync.WaitGroup
		wait.Go(func() {
			<-start
			registry.ReleaseExpiredBlacklists(now)
		})
		wait.Go(func() {
			<-start
			count, exists, _ := registry.RecordFailureWithBlacklist(1, 1, fresh)
			if !exists || count != 1 {
				t.Errorf("RecordFailureWithBlacklist() = %d/%t, want 1/true", count, exists)
			}
		})
		close(start)
		wait.Wait()

		view := registry.Snapshot()[0]
		if !view.Blacklisted || view.FailureCount != 1 ||
			!view.BlacklistReleaseAt.Equal(fresh) {
			after := registry.CaptureActiveCredentialRefs([]uint{1})[0]
			if after.FailureGeneration <= before.FailureGeneration {
				t.Fatalf("attempt %d: failure generation = %d, want > %d", attempt, after.FailureGeneration, before.FailureGeneration)
			}
			t.Fatalf("attempt %d: credential state = %#v, want fresh failure and deadline", attempt, view)
		}
		after := registry.CaptureActiveCredentialRefs([]uint{1})[0]
		if after.FailureGeneration <= before.FailureGeneration {
			t.Fatalf("attempt %d: failure generation = %d, want > %d", attempt, after.FailureGeneration, before.FailureGeneration)
		}
	}
}

func TestRecordEntryFailureWithBlacklistAtomicallyRefreshesExpiredDeadline(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	fresh := now.Add(time.Hour)
	registry := NewCredentialRegistry()
	key := RouteEntryKey{GroupID: 1, EntryID: "model-a"}
	if _, changed := registry.SetEntryBlacklistedWithChange(key); !changed {
		t.Fatal("SetEntryBlacklistedWithChange() changed = false")
	}
	if !registry.SetEntryBlacklistReleaseAt(1, key.EntryID, now.Add(-time.Second)) {
		t.Fatal("SetEntryBlacklistReleaseAt() = false")
	}
	before, _ := registry.EntryRuntime(key, now)
	count, exists, becameBlacklisted := registry.RecordEntryFailureWithBlacklist(
		key.GroupID, key.EntryID, 1, fresh,
	)
	if !exists || count != 1 || becameBlacklisted {
		t.Fatalf("RecordEntryFailureWithBlacklist() = %d/%t/%t", count, exists, becameBlacklisted)
	}
	view, _ := registry.EntryRuntime(key, now)
	if !view.Blacklisted || view.FailureCount != 1 ||
		!view.BlacklistReleaseAt.Equal(fresh) || view.FailureVersion <= before.FailureVersion {
		t.Fatalf("entry state = %#v, want failure, refreshed deadline and version", view)
	}
}

// R1: a stale entry view cannot clear a blacklist that a newer failure created
// after the view was captured, while the current view can.
func TestRecoverEntryIfVersionMatchRejectsStaleView(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := NewCredentialRegistry()
	key := RouteEntryKey{GroupID: 1, EntryID: "model-a"}
	if _, changed := registry.SetEntryBlacklistedWithChange(key); !changed {
		t.Fatal("SetEntryBlacklistedWithChange() changed = false")
	}
	stale, ok := registry.EntryRuntime(key, now)
	if !ok {
		t.Fatal("EntryRuntime() = false")
	}
	if _, changed := registry.SetEntryBlacklistedWithChange(key); changed {
		t.Fatal("second SetEntryBlacklistedWithChange() changed = true, want already blacklisted")
	}
	if _, ok := registry.IncrEntryFailureForEntry(1, "model-a"); !ok {
		t.Fatal("IncrEntryFailureForEntry() = false")
	}
	if registry.RecoverEntryIfVersionMatch(1, "model-a", stale.FailureVersion) {
		t.Fatal("stale entry view recovered a newer failure state")
	}
	stillBlacklisted, ok := registry.EntryRuntime(key, now)
	if !ok || !stillBlacklisted.Blacklisted {
		t.Fatalf("entry runtime = %#v/%t, want still blacklisted", stillBlacklisted, ok)
	}
	current, ok := registry.EntryRuntime(key, now)
	if !ok {
		t.Fatal("EntryRuntime() current = false")
	}
	if !registry.RecoverEntryIfVersionMatch(1, "model-a", current.FailureVersion) {
		t.Fatal("current entry view failed to recover")
	}
	recovered, _ := registry.EntryRuntime(key, now)
	if recovered.Blacklisted || recovered.FailureCount != 0 || !recovered.BlacklistReleaseAt.IsZero() {
		t.Fatalf("recovered entry = %#v, want cleared", recovered)
	}
}
