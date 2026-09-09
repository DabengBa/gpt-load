package state

import (
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestCredentialRegistryCollectCandidatesExcludesRuntimeUnavailable(t *testing.T) {
	now := time.Date(2026, time.July, 20, 12, 0, 0, 0, time.UTC)
	registry := NewCredentialRegistry()
	entries := []CredentialEntry{
		testCredential(1, 10),
		testCredential(2, 11),
		testCredential(3, 12),
		testCredential(4, 13),
		testCredential(5, 14),
		testCredential(6, 15),
	}
	entries[1].CooldownUntil = now.Add(time.Second)
	entries[2].Blacklisted = true
	entries[3].CooldownUntil = now
	entries[4].AuthState = CredentialAuthStateRefreshing
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatal(err)
	}
	got := registry.CollectCredentialCandidates([]uint{10, 11, 12, 13, 14, 15}, func(id uint) bool { return id == 6 }, now)
	want := []CredentialMeta{{ID: 1, GroupID: 10, Version: 1, IdentityGeneration: 1}, {ID: 4, GroupID: 13, Version: 1, IdentityGeneration: 1}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("CollectCredentialCandidates() = %#v, want %#v", got, want)
	}
}

func TestCredentialRuntimeViewClassifiesAvailability(t *testing.T) {
	now := time.Date(2026, time.July, 24, 10, 0, 0, 0, time.UTC)
	for _, test := range []struct {
		name string
		view CredentialRuntimeView
		want CredentialRuntimeState
	}{
		{name: "blacklist wins cooldown", view: CredentialRuntimeView{Blacklisted: true, CooldownUntil: now.Add(time.Minute)}, want: CredentialRuntimeBlacklisted},
		{name: "future cooldown", view: CredentialRuntimeView{CooldownUntil: now.Add(time.Nanosecond)}, want: CredentialRuntimeCooldown},
		{name: "cooldown equality is available", view: CredentialRuntimeView{CooldownUntil: now}, want: CredentialRuntimeAvailable},
		{name: "available", view: CredentialRuntimeView{}, want: CredentialRuntimeAvailable},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := test.view.RuntimeState(now); got != test.want {
				t.Fatalf("RuntimeState() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestCollectCandidatesUsesRuntimeViewBoundary(t *testing.T) {
	now := time.Date(2026, time.July, 24, 10, 0, 0, 0, time.UTC)
	registry := NewCredentialRegistry()
	entries := []CredentialEntry{testCredential(1, 10), testCredential(2, 11), testCredential(3, 12), testCredential(4, 13)}
	entries[1].CooldownUntil = now.Add(time.Nanosecond)
	entries[2].Blacklisted = true
	entries[3].CooldownUntil = now
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatal(err)
	}
	views := registry.Snapshot()
	availableIDs := make([]uint, 0)
	for _, view := range views {
		if view.RuntimeState(now) == CredentialRuntimeAvailable {
			availableIDs = append(availableIDs, view.ID)
		}
	}
	candidates := registry.CollectCredentialCandidates([]uint{10, 11, 12, 13}, nil, now)
	candidateIDs := make([]uint, 0, len(candidates))
	for _, candidate := range candidates {
		candidateIDs = append(candidateIDs, candidate.ID)
	}
	if !reflect.DeepEqual(candidateIDs, availableIDs) {
		t.Fatalf("candidate IDs = %v, runtime view IDs = %v", candidateIDs, availableIDs)
	}
}

func TestCredentialRegistrySetCooldownNeverShortensDeadline(t *testing.T) {
	now := time.Date(2026, time.July, 22, 12, 0, 0, 0, time.UTC)
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	if !registry.SetCooldown(1, now.Add(time.Hour)) || !registry.SetCooldown(1, now.Add(time.Minute)) {
		t.Fatal("SetCooldown() failed")
	}
	if got := registry.CollectCredentialCandidates([]uint{10}, nil, now.Add(2*time.Minute)); len(got) != 0 {
		t.Fatalf("candidates before longest deadline = %#v", got)
	}
	if got := registry.CollectCredentialCandidates([]uint{10}, nil, now.Add(time.Hour)); len(got) != 1 {
		t.Fatalf("candidates at deadline = %#v", got)
	}
}

func TestCredentialRegistrySetCooldownConcurrentWritersKeepLatestDeadline(t *testing.T) {
	now := time.Date(2026, time.July, 22, 12, 0, 0, 0, time.UTC)
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	const writers = 32
	start := make(chan struct{})
	var wait sync.WaitGroup
	for offset := 1; offset <= writers; offset++ {
		deadline := now.Add(time.Duration(offset) * time.Minute)
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			if !registry.SetCooldown(1, deadline) {
				t.Errorf("SetCooldown(%v) failed", deadline)
			}
		}()
	}
	close(start)
	wait.Wait()
	latest := now.Add(writers * time.Minute)
	if got := registry.CollectCredentialCandidates([]uint{10}, nil, latest.Add(-time.Nanosecond)); len(got) != 0 {
		t.Fatalf("candidates before latest deadline = %#v", got)
	}
	if got := registry.CollectCredentialCandidates([]uint{10}, nil, latest); len(got) != 1 {
		t.Fatalf("candidates at latest deadline = %#v", got)
	}
}

func TestCredentialRegistryClearFailureAndRecover(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	registry.SetBlacklisted(1)
	for range 3 {
		registry.IncrFailure(1)
	}
	if !registry.ClearFailure(1) || !registry.Snapshot()[0].Blacklisted || registry.Snapshot()[0].FailureCount != 0 {
		t.Fatalf("ClearFailure() did not preserve blacklist: %#v", registry.Snapshot())
	}
	if !registry.Recover(1) || registry.Snapshot()[0].Blacklisted || registry.Snapshot()[0].FailureCount != 0 {
		t.Fatalf("Recover() did not clear runtime health: %#v", registry.Snapshot())
	}
	if registry.ClearFailure(99) || registry.Recover(99) {
		t.Fatal("missing credential runtime mutation succeeded")
	}
}

func TestCredentialRegistryRecoverIfMatchRejectsStaleGeneration(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	registry.SetBlacklisted(1)
	stale := registry.BlacklistedCredentials()[0]
	registry.IncrFailure(1)
	before := registry.Snapshot()[0]
	if registry.RecoverIfMatch(stale) {
		t.Fatal("RecoverIfMatch(stale) = true")
	}
	if got := registry.Snapshot()[0]; !reflect.DeepEqual(got, before) {
		t.Fatalf("stale recovery mutated runtime: %#v -> %#v", before, got)
	}
	fresh := registry.BlacklistedCredentials()[0]
	if !registry.RecoverIfMatch(fresh) || registry.Snapshot()[0].Blacklisted {
		t.Fatal("RecoverIfMatch(fresh) did not recover")
	}
}

func TestCredentialRegistryBlacklistedCredentialsReturnsSortedRefs(t *testing.T) {
	registry := NewCredentialRegistry()
	entries := []CredentialEntry{testCredential(3, 30), testCredential(2, 20), testCredential(1, 10), testCredential(4, 40)}
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatal(err)
	}
	for _, id := range []uint{3, 2, 1} {
		registry.SetBlacklisted(id)
	}
	got := registry.BlacklistedCredentials()
	want := []CredentialRef{
		{ID: 1, GroupID: 10, Version: 1, IdentityGeneration: 1, Fingerprint: "fp", EncryptedValue: "cipher", FailureGeneration: 1},
		{ID: 2, GroupID: 20, Version: 1, IdentityGeneration: 1, Fingerprint: "fp", EncryptedValue: "cipher", FailureGeneration: 1},
		{ID: 3, GroupID: 30, Version: 1, IdentityGeneration: 1, Fingerprint: "fp", EncryptedValue: "cipher", FailureGeneration: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BlacklistedCredentials() = %#v, want %#v", got, want)
	}
}

func TestValidateCredentialEntriesRejectsInvalidAuthState(t *testing.T) {
	entry := testCredential(1, 10)
	entry.AuthState = CredentialAuthState("revoked")
	if err := ValidateCredentialEntries([]CredentialEntry{entry}); err == nil {
		t.Fatal("ValidateCredentialEntries() accepted invalid auth state")
	}
}
