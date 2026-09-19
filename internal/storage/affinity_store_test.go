package storage

import (
	"bytes"
	"path/filepath"
	"testing"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/affinity"
	"gpt-load/internal/storage/models"
)

var testDurablePolicy = affinity.DurablePolicy{TTL: time.Hour, Capacity: 10_000}

func openAffinityDatabase(t *testing.T) *gorm.DB {
	t.Helper()
	db, err := Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}

	if err := AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	return db
}

func TestAffinityStoreRoundTripAndRestart(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	key := affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	target := affinity.Target{GroupID: 7, CredentialID: 11, IdentityGeneration: 13}
	if err := store.Upsert(t.Context(), key, target, testDurablePolicy); err != nil {
		t.Fatalf("Upsert() error = %v", err)
	}
	got, found, err := NewAffinityStore(db).Lookup(t.Context(), key, testDurablePolicy)
	if err != nil || !found || got != target {
		t.Fatalf("Lookup() = %#v, %t, %v; want %#v, true, nil", got, found, err, target)
	}
}

func TestAffinityStoreRejectsInvalidKeyAndTarget(t *testing.T) {
	store := NewAffinityStore(openAffinityDatabase(t))
	valid := affinity.Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3}
	for _, key := range []affinity.Key{"", "not-a-hmac", "0123456789abcdef"} {
		if err := store.Upsert(t.Context(), key, valid, testDurablePolicy); err == nil {
			t.Fatalf("Upsert(%q) error = nil, want validation error", key)
		}
	}
	if err := store.Upsert(t.Context(), affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"), affinity.Target{}, testDurablePolicy); err == nil {
		t.Fatal("Upsert(invalid target) error = nil, want validation error")
	}
}

func TestAffinityStoreDistinguishesNotFoundAndDecodeError(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	key := affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	if _, found, err := store.Lookup(t.Context(), key, testDurablePolicy); err != nil || found {
		t.Fatalf("missing Lookup() = found %t, err %v; want false, nil", found, err)
	}
	malformed := affinity.Key("1111111111111111111111111111111111111111111111111111111111111111")
	invalidTarget := affinity.Key("2222222222222222222222222222222222222222222222222222222222222222")
	for _, row := range []struct {
		key   affinity.Key
		value string
	}{
		{key: malformed, value: `not-json`},
		{key: invalidTarget, value: `{"group_id":1}`},
	} {
		if err := db.Exec("INSERT INTO system_settings (key, value, updated_at_ms) VALUES (?, ?, ?)", affinityBindingSettingPrefix+string(row.key), row.value, time.Now().UnixMilli()).Error; err != nil {
			t.Fatal(err)
		}
		if _, _, err := store.Lookup(t.Context(), row.key, testDurablePolicy); err == nil {
			t.Fatalf("Lookup(%q) error = nil, want decode error", row.value)
		}
	}
}

func TestAffinityStoreRecoversAfterDatabaseReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "affinity.db")
	key := affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	target := affinity.Target{GroupID: 7, CredentialID: 11, IdentityGeneration: 13}

	first, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := AutoMigrate(first); err != nil {
		t.Fatal(err)
	}
	if err := NewAffinityStore(first).Upsert(t.Context(), key, target, testDurablePolicy); err != nil {
		t.Fatalf("Upsert() error = %v", err)
	}
	firstSQL, err := first.DB()
	if err != nil {
		t.Fatal(err)
	}
	if err := firstSQL.Close(); err != nil {
		t.Fatal(err)
	}

	second, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	secondSQL, err := second.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = secondSQL.Close() })
	got, found, err := NewAffinityStore(second).Lookup(t.Context(), key, testDurablePolicy)
	if err != nil || !found || got != target {
		t.Fatalf("Lookup() after reopen = %#v, %t, %v; want %#v, true, nil", got, found, err, target)
	}
}

func TestAffinityStoreLookupHonorsDurableTTL(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	base := time.Date(2026, time.January, 2, 12, 0, 0, 0, time.UTC)
	store.now = func() time.Time { return base }
	key := affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	if err := db.Exec(
		"INSERT INTO system_settings (key, value, updated_at_ms) VALUES (?, ?, ?)",
		affinityBindingSettingPrefix+string(key),
		`{"group_id":1,"credential_id":2,"identity_generation":3}`,
		base.Add(-time.Hour).UnixMilli(),
	).Error; err != nil {
		t.Fatal(err)
	}
	if _, found, err := store.Lookup(t.Context(), key, testDurablePolicy); err != nil || found {
		t.Fatalf("expired Lookup() = found %t, err %v; want false, nil", found, err)
	}
}

func TestAffinityStoreUpsertsSingleRowAtomically(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	key := affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	first := affinity.Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3}
	second := affinity.Target{GroupID: 4, CredentialID: 5, IdentityGeneration: 6}
	if err := store.Upsert(t.Context(), key, first, testDurablePolicy); err != nil {
		t.Fatalf("first Upsert() error = %v", err)
	}
	if err := store.Upsert(t.Context(), key, second, testDurablePolicy); err != nil {
		t.Fatalf("second Upsert() error = %v", err)
	}
	var count int64
	if err := db.Model(&models.SystemSetting{}).
		Where("key = ?", affinityBindingSettingPrefix+string(key)).Count(&count).Error; err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("stored binding rows = %d, want 1", count)
	}
	got, found, err := store.Lookup(t.Context(), key, testDurablePolicy)
	if err != nil || !found || got != second {
		t.Fatalf("Lookup() = %#v, %t, %v; want latest %#v", got, found, err, second)
	}
}

func TestAffinityStoreSweepRemovesExpiredAndOldestBindings(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	base := time.Date(2026, time.January, 2, 12, 0, 0, 0, time.UTC)
	rows := []struct {
		key       affinity.Key
		updatedAt int64
	}{
		{key: affinity.Key("1111111111111111111111111111111111111111111111111111111111111111"), updatedAt: base.Add(-2 * time.Hour).UnixMilli()},
		{key: affinity.Key("2222222222222222222222222222222222222222222222222222222222222222"), updatedAt: base.Add(-30 * time.Minute).UnixMilli()},
		{key: affinity.Key("3333333333333333333333333333333333333333333333333333333333333333"), updatedAt: base.Add(-10 * time.Minute).UnixMilli()},
		{key: affinity.Key("4444444444444444444444444444444444444444444444444444444444444444"), updatedAt: base.Add(-time.Minute).UnixMilli()},
	}
	for _, row := range rows {
		if err := db.Exec(
			"INSERT INTO system_settings (key, value, updated_at_ms) VALUES (?, ?, ?)",
			affinityBindingSettingPrefix+string(row.key),
			`{"group_id":1,"credential_id":2,"identity_generation":3}`,
			row.updatedAt,
		).Error; err != nil {
			t.Fatal(err)
		}
	}

	if err := store.SweepAffinityBindings(t.Context(), base, affinity.DurablePolicy{TTL: time.Hour, Capacity: 2}); err != nil {
		t.Fatalf("SweepAffinityBindings() error = %v", err)
	}

	var keys []string
	if err := db.Model(&models.SystemSetting{}).
		Where("key LIKE ?", affinityBindingSettingPrefix+"%").
		Order("key ASC").Pluck("key", &keys).Error; err != nil {
		t.Fatal(err)
	}
	want := []string{
		affinityBindingSettingPrefix + string(rows[2].key),
		affinityBindingSettingPrefix + string(rows[3].key),
	}
	if len(keys) != len(want) || keys[0] != want[0] || keys[1] != want[1] {
		t.Fatalf("remaining affinity keys = %#v, want %#v", keys, want)
	}
}

func TestAffinityStoreUpsertEnforcesDurableCapacity(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	target := affinity.Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3}
	for index, digit := range []byte{'1', '2', '3'} {
		key := affinity.Key(string(bytes.Repeat([]byte{digit}, 64)))
		if err := store.Upsert(t.Context(), key, target, affinity.DurablePolicy{TTL: time.Hour, Capacity: 2}); err != nil {
			t.Fatalf("Upsert(%d) error = %v", index, err)
		}
	}
	var count int64
	if err := db.Model(&models.SystemSetting{}).
		Where("key LIKE ?", affinityBindingSettingPrefix+"%").Count(&count).Error; err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("durable affinity rows = %d, want capacity 2", count)
	}
}

func TestAffinityStoreLookupReturnsDatabaseError(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	if err := sqlDB.Close(); err != nil {
		t.Fatal(err)
	}
	_, _, err = store.Lookup(t.Context(), affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"), testDurablePolicy)
	if err == nil {
		t.Fatal("Lookup(closed DB) error = nil, want database error")
	}
}
