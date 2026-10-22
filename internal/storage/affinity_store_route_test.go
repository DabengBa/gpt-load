package storage

import (
	"testing"
	"time"

	"gpt-load/internal/affinity"
	"gpt-load/internal/storage/models"
)

const routeAffinityKey affinity.Key = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func insertAffinityRow(t *testing.T, value string) {
	t.Helper()
	db := openAffinityDatabase(t)
	t.Cleanup(func() {
		sqlDB, err := db.DB()
		if err != nil {
			t.Fatal(err)
		}
		_ = sqlDB.Close()
	})
	if err := db.Exec(
		"INSERT INTO system_settings (key, value, updated_at_ms) VALUES (?, ?, ?)",
		affinityBindingSettingPrefix+string(routeAffinityKey), value, time.Now().UnixMilli(),
	).Error; err != nil {
		t.Fatal(err)
	}
	got, found, err := NewAffinityStore(db).Lookup(t.Context(), routeAffinityKey, testDurablePolicy)
	if err != nil || found {
		t.Fatalf("Lookup(obsolete payload) = %#v, found %t, err %v; want a plain miss", got, found, err)
	}
}

// R2: the durable authority round-trips the exact route identity, so a
// recovered binding still names the entry and upstream model that succeeded.
func TestAffinityStoreRoundTripsRouteIdentity(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	target := affinity.Target{
		GroupID: 7, CredentialID: 11, IdentityGeneration: 13,
		EntryID: "e000000000042", UpstreamModelID: "upstream-model",
	}
	if err := store.Upsert(t.Context(), routeAffinityKey, target, testDurablePolicy); err != nil {
		t.Fatalf("Upsert() error = %v", err)
	}
	got, found, err := NewAffinityStore(db).Lookup(t.Context(), routeAffinityKey, testDurablePolicy)
	if err != nil || !found || got != target {
		t.Fatalf("Lookup() = %#v, %t, %v; want %#v, true, nil", got, found, err, target)
	}
}

// R2: a resource route has no upstream model. The empty value must survive the
// durable round trip instead of being mistaken for a missing field.
func TestAffinityStoreRoundTripsEmptyUpstreamModel(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	target := affinity.Target{GroupID: 7, CredentialID: 11, IdentityGeneration: 13, EntryID: "e000000000042"}
	if err := store.Upsert(t.Context(), routeAffinityKey, target, testDurablePolicy); err != nil {
		t.Fatalf("Upsert() error = %v", err)
	}
	got, found, err := store.Lookup(t.Context(), routeAffinityKey, testDurablePolicy)
	if err != nil || !found || got != target {
		t.Fatalf("Lookup() = %#v, %t, %v; want %#v, true, nil", got, found, err, target)
	}
}

// R2: the obsolete credential-only payload is not a compatibility surface. It
// has no route identity, so it is a plain miss that the next success replaces.
func TestAffinityStoreTreatsObsoleteCredentialOnlyPayloadAsMiss(t *testing.T) {
	insertAffinityRow(t, `{"group_id":1,"credential_id":2,"identity_generation":3}`)
}

// The read path never deletes: the obsolete row stays until a success
// overwrites it, so a transient read cannot destroy the only durable record.
func TestAffinityStoreDoesNotDeleteObsoletePayloadOnLookup(t *testing.T) {
	db := openAffinityDatabase(t)
	if err := db.Exec(
		"INSERT INTO system_settings (key, value, updated_at_ms) VALUES (?, ?, ?)",
		affinityBindingSettingPrefix+string(routeAffinityKey),
		`{"group_id":1,"credential_id":2,"identity_generation":3}`,
		time.Now().UnixMilli(),
	).Error; err != nil {
		t.Fatal(err)
	}
	var before int64
	if err := db.Model(&models.SystemSetting{}).
		Where("key = ?", affinityBindingSettingPrefix+string(routeAffinityKey)).
		Count(&before).Error; err != nil {
		t.Fatal(err)
	}
	if _, found, err := NewAffinityStore(db).Lookup(t.Context(), routeAffinityKey, testDurablePolicy); err != nil || found {
		t.Fatalf("Lookup() = found %t, err %v; want a plain miss", found, err)
	}
	var after int64
	if err := db.Model(&models.SystemSetting{}).
		Where("key = ?", affinityBindingSettingPrefix+string(routeAffinityKey)).
		Count(&after).Error; err != nil {
		t.Fatal(err)
	}
	if before != 1 || after != before {
		t.Fatalf("rows before=%d after=%d, want the obsolete row preserved by lookup", before, after)
	}

	// The next success replaces the row in place with the full route identity.
	target := affinity.Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3, EntryID: "e000000000001", UpstreamModelID: "m"}
	if err := NewAffinityStore(db).Upsert(t.Context(), routeAffinityKey, target, testDurablePolicy); err != nil {
		t.Fatalf("Upsert() error = %v", err)
	}
	got, found, err := NewAffinityStore(db).Lookup(t.Context(), routeAffinityKey, testDurablePolicy)
	if err != nil || !found || got != target {
		t.Fatalf("Lookup() after replace = %#v, %t, %v; want %#v", got, found, err, target)
	}
}

// A payload with a route identity but no credential identity is corrupt, not
// obsolete: it still fails closed.
func TestAffinityStoreRejectsPayloadWithoutCredentialIdentity(t *testing.T) {
	db := openAffinityDatabase(t)
	if err := db.Exec(
		"INSERT INTO system_settings (key, value, updated_at_ms) VALUES (?, ?, ?)",
		affinityBindingSettingPrefix+string(routeAffinityKey),
		`{"group_id":1,"credential_id":2,"identity_generation":0,"entry_id":"e1"}`,
		time.Now().UnixMilli(),
	).Error; err != nil {
		t.Fatal(err)
	}
	if _, _, err := NewAffinityStore(db).Lookup(t.Context(), routeAffinityKey, testDurablePolicy); err == nil {
		t.Fatal("Lookup(zero identity generation) error = nil, want decode error")
	}
}
