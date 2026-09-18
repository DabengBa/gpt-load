package storage

import (
	"path/filepath"
	"testing"

	"gorm.io/gorm"

	"gpt-load/internal/affinity"
	"gpt-load/internal/storage/models"
)

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
	if err := store.Upsert(t.Context(), key, target); err != nil {
		t.Fatalf("Upsert() error = %v", err)
	}
	got, found, err := NewAffinityStore(db).Lookup(t.Context(), key)
	if err != nil || !found || got != target {
		t.Fatalf("Lookup() = %#v, %t, %v; want %#v, true, nil", got, found, err, target)
	}
}

func TestAffinityStoreRejectsInvalidKeyAndTarget(t *testing.T) {
	store := NewAffinityStore(openAffinityDatabase(t))
	valid := affinity.Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3}
	for _, key := range []affinity.Key{"", "not-a-hmac", "0123456789abcdef"} {
		if err := store.Upsert(t.Context(), key, valid); err == nil {
			t.Fatalf("Upsert(%q) error = nil, want validation error", key)
		}
	}
	if err := store.Upsert(t.Context(), affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"), affinity.Target{}); err == nil {
		t.Fatal("Upsert(invalid target) error = nil, want validation error")
	}
}

func TestAffinityStoreDistinguishesNotFoundAndDecodeError(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	key := affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	if _, found, err := store.Lookup(t.Context(), key); err != nil || found {
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
		if err := db.Exec("INSERT INTO system_settings (key, value, updated_at_ms) VALUES (?, ?, ?)", affinityBindingSettingPrefix+string(row.key), row.value, 1).Error; err != nil {
			t.Fatal(err)
		}
		if _, _, err := store.Lookup(t.Context(), row.key); err == nil {
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
	if err := NewAffinityStore(first).Upsert(t.Context(), key, target); err != nil {
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
	got, found, err := NewAffinityStore(second).Lookup(t.Context(), key)
	if err != nil || !found || got != target {
		t.Fatalf("Lookup() after reopen = %#v, %t, %v; want %#v, true, nil", got, found, err, target)
	}
}

func TestAffinityStoreUpsertsSingleRowAtomically(t *testing.T) {
	db := openAffinityDatabase(t)
	store := NewAffinityStore(db)
	key := affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	first := affinity.Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3}
	second := affinity.Target{GroupID: 4, CredentialID: 5, IdentityGeneration: 6}
	if err := store.Upsert(t.Context(), key, first); err != nil {
		t.Fatalf("first Upsert() error = %v", err)
	}
	if err := store.Upsert(t.Context(), key, second); err != nil {
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
	got, found, err := store.Lookup(t.Context(), key)
	if err != nil || !found || got != second {
		t.Fatalf("Lookup() = %#v, %t, %v; want latest %#v", got, found, err, second)
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
	_, _, err = store.Lookup(t.Context(), affinity.Key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"))
	if err == nil {
		t.Fatal("Lookup(closed DB) error = nil, want database error")
	}
}
