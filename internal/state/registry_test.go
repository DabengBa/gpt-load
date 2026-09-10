package state

import (
	"reflect"
	"testing"
	"time"

	providerobservation "gpt-load/internal/subscription/providers/observation"
)

func intPointer(value int) *int           { return &value }
func floatPointer(value float64) *float64 { return &value }

func testCredential(id, groupID uint) CredentialEntry {
	return CredentialEntry{ID: id, GroupID: groupID, Version: 1, IdentityGeneration: 1, Fingerprint: "fp", AuthState: CredentialAuthStateReady, EncryptedValue: "cipher"}
}

func registryEntry(t *testing.T, registry *CredentialRegistry, credentialID uint) CredentialEntry {
	t.Helper()
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	groupID, ok := registry.credentialGroups[credentialID]
	if !ok {
		t.Fatalf("credential %d missing", credentialID)
	}
	return *registry.buckets[groupID][credentialID]
}

func TestCredentialRegistryReplaceAndEncryptedValue(t *testing.T) {
	registry := NewCredentialRegistry()
	entries := []CredentialEntry{testCredential(1, 10), testCredential(2, 20)}
	entries[0].EncryptedValue = "cipher-one"
	entries[1].EncryptedValue = "cipher-two"
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatal(err)
	}
	if got, ok := registry.EncryptedCredentialData(1); !ok || got != "cipher-one" {
		t.Fatalf("EncryptedCredentialData(1) = %q/%t", got, ok)
	}
	if got, ok := registry.ActiveEncryptedCredentialData(1, 10); !ok || got != "cipher-one" {
		t.Fatalf("ActiveEncryptedCredentialData() = %q/%t", got, ok)
	}
	if _, ok := registry.ActiveEncryptedCredentialData(1, 20); ok {
		t.Fatal("group mismatch returned active credential")
	}
}

func TestCredentialRegistryReplaceSecretIfMatchPreservesRuntimeState(t *testing.T) {
	registry := NewCredentialRegistry()
	entry := testCredential(1, 10)
	entry.CooldownUntil = time.Unix(100, 0)
	entry.Blacklisted = true
	entry.FailureCount = 2
	if err := registry.ReplaceCredentials([]CredentialEntry{entry}); err != nil {
		t.Fatal(err)
	}
	before := registryEntry(t, registry, 1)
	if !registry.ReplaceCredentialSecretIfMatch(1, 1, 4, "new-fingerprint", "new-cipher") {
		t.Fatal("matching secret replacement failed")
	}
	got := registryEntry(t, registry, 1)
	if got.Version != 4 || got.Fingerprint != "new-fingerprint" || got.EncryptedValue != "new-cipher" || !got.CooldownUntil.Equal(before.CooldownUntil) || !got.Blacklisted || got.FailureCount != 2 {
		t.Fatalf("runtime state changed during replacement: %#v", got)
	}
	if registry.ReplaceCredentialSecretIfMatch(1, 1, 5, "stale", "stale") {
		t.Fatal("stale secret replacement succeeded")
	}
}

func TestCredentialRegistryRestoreRuntimeState(t *testing.T) {
	registry := NewCredentialRegistry()
	entry := testCredential(1, 10)
	entry.CooldownUntil = time.Now().Add(time.Hour)
	entry.Blacklisted = true
	entry.FailureCount = 3
	if err := registry.ReplaceCredentials([]CredentialEntry{entry}); err != nil {
		t.Fatal(err)
	}
	if !registry.RestoreRuntimeState(1) {
		t.Fatal("RestoreRuntimeState() failed")
	}
	got := registryEntry(t, registry, 1)
	if !got.CooldownUntil.IsZero() || got.Blacklisted || got.FailureCount != 0 {
		t.Fatalf("restored entry = %#v", got)
	}
	if registry.RestoreRuntimeState(99) {
		t.Fatal("missing credential restore succeeded")
	}
}

func TestCredentialRegistryRestoreRuntimeStateIfMatchRequiresAndClearsCooldown(t *testing.T) {
	registry := NewCredentialRegistry()
	cooldown := time.Now().Add(time.Hour).UTC()
	entry := testCredential(1, 10)
	entry.Blacklisted = true
	entry.CooldownUntil = cooldown
	entry.FailureCount = 3
	if err := registry.ReplaceCredentials([]CredentialEntry{entry}); err != nil {
		t.Fatal(err)
	}
	ref, ok := registry.CredentialRef(1)
	if !ok {
		t.Fatal("CredentialRef() failed")
	}
	if registry.RestoreRuntimeStateIfMatch(ref, cooldown.Add(time.Minute)) {
		t.Fatal("restore with stale cooldown succeeded")
	}
	if got := registryEntry(t, registry, 1); !got.CooldownUntil.Equal(cooldown) || !got.Blacklisted {
		t.Fatalf("stale restore mutated entry: %#v", got)
	}
	if !registry.RestoreRuntimeStateIfMatch(ref, cooldown) {
		t.Fatal("restore with matching cooldown failed")
	}
	got := registryEntry(t, registry, 1)
	if !got.CooldownUntil.IsZero() || got.Blacklisted || got.FailureCount != 0 {
		t.Fatalf("matching restore did not clear runtime state: %#v", got)
	}
}

func TestCredentialRegistryQuotaObservationDoesNotAffectCandidates(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	remaining := 0.0
	if !registry.SetCredentialQuotaObservation(1, &remaining, time.Now().Add(time.Hour)) {
		t.Fatal("SetCredentialQuotaObservation() failed")
	}
	if got := registry.CollectCredentialCandidates([]uint{10}, nil, time.Now()); len(got) != 1 {
		t.Fatalf("quota observation removed candidate: %#v", got)
	}
}

func TestCredentialRegistryApplyQuotaWindowsPublishesTightestAccountBottleneck(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	resetA, resetB := time.Now().Add(time.Hour).UnixMilli(), time.Now().Add(2*time.Hour).UnixMilli()
	if !registry.ApplyQuotaWindows(1, []providerobservation.QuotaWindow{{Scope: "account", Utilization: floatPointer(0.6), ResetAtMS: &resetA}, {Scope: "account", Utilization: floatPointer(0.9), ResetAtMS: &resetB}}) {
		t.Fatal("ApplyQuotaWindows() failed")
	}
	view := registry.Snapshot()[0]
	remaining := view.ObservedQuotaRemaining()
	if remaining == nil || *remaining < 0.099 || *remaining > 0.101 {
		t.Fatalf("quota view = %#v", view)
	}
}

func TestCredentialRegistryBatchMutationsAreAllOrNothing(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	before := registry.Snapshot()
	if err := registry.ApplyCredentialImport(10, []CredentialEntry{testCredential(2, 10), testCredential(3, 10)}); err == nil {
		t.Fatal("multi-credential import succeeded")
	}
	if got := registry.Snapshot(); !reflect.DeepEqual(got, before) {
		t.Fatalf("rejected import mutated registry: %#v -> %#v", before, got)
	}
}

func TestCredentialRegistryCaptureRefsIncludesUnavailableRuntimeEntries(t *testing.T) {
	now := time.Now().UTC().Add(time.Hour)
	registry := NewCredentialRegistry()
	entries := []CredentialEntry{testCredential(1, 10), testCredential(2, 20), testCredential(3, 30)}
	entries[0].Blacklisted = true
	entries[1].CooldownUntil = now
	entries[2].AuthState = CredentialAuthStateRefreshing
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatal(err)
	}
	refs := registry.CaptureActiveCredentialRefs([]uint{10, 20, 30})
	if len(refs) != 2 || refs[0].ID != 1 || refs[1].ID != 2 {
		t.Fatalf("captured refs = %#v", refs)
	}
}

func TestCredentialRegistryActiveEncryptedDataIfMatchRejectsIdentityChanges(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	ref, ok := registry.CredentialRef(1)
	if !ok {
		t.Fatal("CredentialRef() failed")
	}
	if got, ok := registry.ActiveEncryptedCredentialDataIfMatch(ref); !ok || got != "cipher" {
		t.Fatalf("matching ref = %q/%t", got, ok)
	}
	if !registry.ReplaceCredentialSecretIfMatch(1, 1, 2, "changed", "changed-cipher") {
		t.Fatal("secret replacement failed")
	}
	if got, ok := registry.ActiveEncryptedCredentialDataIfMatch(ref); ok || got != "" {
		t.Fatalf("stale ref = %q/%t", got, ok)
	}
}

func TestCredentialRegistryRemoveOperationsClearReverseIndexes(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10), testCredential(2, 20)}); err != nil {
		t.Fatal(err)
	}
	if !registry.RemoveCredential(1) || registry.RemoveCredential(1) {
		t.Fatal("RemoveCredential() result unexpected")
	}
	if _, ok := registry.CredentialRef(1); ok {
		t.Fatal("removed credential still has reverse index")
	}
	if !registry.RemoveGroup(20) || registry.RemoveGroup(20) {
		t.Fatal("RemoveGroup() result unexpected")
	}
	if _, ok := registry.CredentialRef(2); ok {
		t.Fatal("removed group credential still has reverse index")
	}
}

func TestCredentialRegistryReconcilePreservesSatisfiedRuntimeState(t *testing.T) {
	registry := NewCredentialRegistry()
	entry := testCredential(1, 10)
	if err := registry.ReplaceCredentials([]CredentialEntry{entry}); err != nil {
		t.Fatal(err)
	}
	registry.IncrFailure(1)
	changed, err := registry.ReconcileGroup(10, []CredentialEntry{entry})
	if err != nil || changed {
		t.Fatalf("ReconcileGroup() = %t/%v", changed, err)
	}
	if got := registryEntry(t, registry, 1); got.FailureCount != 1 {
		t.Fatalf("reconcile reset runtime state: %#v", got)
	}
}

func TestCredentialRegistryValidateRejectsMalformedEntries(t *testing.T) {
	for _, entries := range [][]CredentialEntry{
		{{GroupID: 10, Version: 1, IdentityGeneration: 1, Fingerprint: "fp", EncryptedValue: "cipher"}},
		{{ID: 1, GroupID: 10, IdentityGeneration: 1, Fingerprint: "fp", EncryptedValue: "cipher"}},
		{{ID: 1, GroupID: 10, Version: 1, IdentityGeneration: 1, Fingerprint: "fp", AuthState: CredentialAuthState("invalid"), EncryptedValue: "cipher"}},
		{{ID: 1, GroupID: 10, Version: 1, IdentityGeneration: 1, Fingerprint: "fp"}},
	} {
		if err := ValidateCredentialEntries(entries); err == nil {
			t.Fatalf("ValidateCredentialEntries(%#v) accepted malformed input", entries)
		}
	}
}

func TestCredentialRegistryApplyImportRejectsExistingIDFromAnotherGroup(t *testing.T) {
	registry := NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]CredentialEntry{testCredential(1, 10)}); err != nil {
		t.Fatal(err)
	}
	if err := registry.ApplyCredentialImport(20, []CredentialEntry{testCredential(1, 20)}); err == nil {
		t.Fatal("cross-group credential identity reuse succeeded")
	}
}
