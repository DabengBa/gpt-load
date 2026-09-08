package state

import (
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestEntryRuntimeIsIndependentByGroupAndEntryID(t *testing.T) {
	registry := NewCredentialRegistry()
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	keyA := RouteEntryKey{GroupID: 1, EntryID: "model-a"}
	keyB := RouteEntryKey{GroupID: 1, EntryID: "model-b"}
	keyOtherGroup := RouteEntryKey{GroupID: 2, EntryID: "model-a"}

	if count, ok := registry.IncrEntryFailure(keyA); !ok || count != 1 {
		t.Fatalf("IncrEntryFailure(keyA) = %d/%t", count, ok)
	}
	if count, ok := registry.IncrEntryFailure(keyA); !ok || count != 2 {
		t.Fatalf("second IncrEntryFailure(keyA) = %d/%t", count, ok)
	}
	if got, ok := registry.EntryRuntime(keyB, now); ok || got.FailureCount != 0 {
		t.Fatalf("sibling entry runtime = %#v/%t", got, ok)
	}
	if got, ok := registry.EntryRuntime(keyOtherGroup, now); ok || got.FailureCount != 0 {
		t.Fatalf("other-group entry runtime = %#v/%t", got, ok)
	}

	until := now.Add(time.Minute)
	if exists, changed := registry.SetEntryCooldownWithChange(keyA, until); !exists || !changed {
		t.Fatalf("SetEntryCooldownWithChange() = %t/%t", exists, changed)
	}
	view, ok := registry.EntryRuntime(keyA, now)
	if !ok || view.RuntimeState(now) != EntryRuntimeCooldown || !view.CooldownUntil.Equal(until) {
		t.Fatalf("cooldown view = %#v/%t", view, ok)
	}
	if got, ok := registry.EntryRuntime(keyB, now); ok || got.RuntimeState(now) != EntryRuntimeAvailable {
		t.Fatalf("sibling after cooldown = %#v/%t", got, ok)
	}
	if got, ok := registry.EntryRuntime(keyA, until); !ok || got.RuntimeState(until) != EntryRuntimeAvailable {
		t.Fatalf("expired cooldown = %#v/%t", got, ok)
	}
}

func TestEntryRuntimeBlacklistAndRecoveryAreIndependentFromCredentialState(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{{
		ID: 1, GroupID: 1, Status: CredentialStatusActive,
		Version: 1, IdentityGeneration: 1, Fingerprint: "fingerprint", EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatal(err)
	}
	key := RouteEntryKey{GroupID: 1, EntryID: "model-a"}
	if !registry.SetBlacklisted(1) {
		t.Fatal("SetBlacklisted(credential) = false")
	}
	if _, ok := registry.IncrEntryFailure(key); !ok {
		t.Fatal("IncrEntryFailure() = false")
	}
	if _, changed := registry.SetEntryBlacklistedWithChange(key); !changed {
		t.Fatal("SetEntryBlacklistedWithChange() changed = false")
	}
	credential := registry.Snapshot()[0]
	entry, ok := registry.EntryRuntime(key, time.Time{})
	if !ok || !credential.Blacklisted || !entry.Blacklisted {
		t.Fatalf("credential/entry state = %#v / %#v", credential, entry)
	}
	if !registry.RecoverEntry(key) {
		t.Fatal("RecoverEntry() = false")
	}
	entry, ok = registry.EntryRuntime(key, time.Time{})
	if !ok || entry.Blacklisted || entry.FailureCount != 0 {
		t.Fatalf("recovered entry = %#v/%t", entry, ok)
	}
	if !registry.Snapshot()[0].Blacklisted {
		t.Fatal("entry recovery changed credential blacklist")
	}
}

func TestEntryRuntimeSnapshotIsDetachedSecretFreeAndConcurrent(t *testing.T) {
	registry := NewCredentialRegistry()
	key := RouteEntryKey{GroupID: 7, EntryID: "model-a"}
	if _, ok := registry.IncrEntryFailure(key); !ok {
		t.Fatal("IncrEntryFailure() = false")
	}
	views := registry.EntryRuntimeSnapshot()
	if len(views) != 1 || !reflect.DeepEqual(views[0].Key, key) || views[0].FailureCount != 1 {
		t.Fatalf("EntryRuntimeSnapshot() = %#v", views)
	}
	views[0].FailureCount = 99
	if again := registry.EntryRuntimeSnapshot(); again[0].FailureCount != 1 {
		t.Fatalf("snapshot aliases registry: %#v", again)
	}

	const workers = 8
	const operations = 100
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for operation := 0; operation < operations; operation++ {
				key := RouteEntryKey{GroupID: uint(worker + 1), EntryID: "model"}
				registry.IncrEntryFailure(key)
				registry.EntryRuntime(key, time.Time{})
			}
		}(worker)
	}
	wg.Wait()
	if got := len(registry.EntryRuntimeSnapshot()); got != workers+1 {
		t.Fatalf("entry runtime count = %d, want %d", got, workers+1)
	}
}
