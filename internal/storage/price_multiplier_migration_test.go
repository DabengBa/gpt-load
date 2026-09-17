package storage

import (
	"fmt"
	"testing"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	migrationfiles "gpt-load/internal/storage/migrations"
)

func TestPriceMultiplierMigrationAddsConfigurationColumns(t *testing.T) {
	t.Parallel()
	db := openInternalMigrationTestDatabase(t)
	if err := AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"groups", "access_keys"} {
		if !db.Migrator().HasColumn(table, "price_multiplier_micros") {
			t.Fatalf("%s.price_multiplier_micros missing", table)
		}
	}
}

func TestPriceMultiplierMigrationUpgradeAndRecovery(t *testing.T) {
	t.Parallel()
	testPriceMultiplierMigrationContract(t, func(t *testing.T) *gorm.DB { return openInternalMigrationTestDatabase(t) })
}

func testPriceMultiplierMigrationContract(t *testing.T, open func(*testing.T) *gorm.DB) {
	t.Helper()
	for _, mode := range []string{"fresh", "upgrade", "interrupted", "columns_added"} {
		t.Run(mode, func(t *testing.T) {
			db := open(t)
			if db.Dialector.Name() == "sqlite" {
				t.Parallel()
			}
			if mode != "fresh" {
				if err := applyMigrationRegistry(db, migrations[:8]); err != nil {
					t.Fatal(err)
				}
				if err := db.Table("groups").Create(map[string]any{
					"id": 1, "name": "legacy multiplier group", "channel_id": "openai", "connection_type": "api_key",
					"params": "{}", "models": "[]", "enabled": true, "created_at_ms": 1, "updated_at_ms": 1,
				}).Error; err != nil {
					t.Fatal(err)
				}
				if err := db.Exec(`INSERT INTO access_keys (id, name, key_value, key_hash, key_suffix, status, filters, created_at_ms, updated_at_ms) VALUES (1, 'legacy multiplier key', 'encrypted', 'multiplier-hash', 'cafe', 'active', '{}', 1, 1)`).Error; err != nil {
					t.Fatal(err)
				}
			}
			if mode == "interrupted" || mode == "columns_added" {
				entry := migrations[8]
				entry.Up = func(tx *gorm.DB) error {
					if mode == "columns_added" {
						if err := migrationfiles.Up0009(tx); err != nil {
							return err
						}
						return fmt.Errorf("simulated interruption before recording price multiplier migration")
					}
					if err := tx.Exec(`ALTER TABLE ? ADD COLUMN price_multiplier_micros BIGINT NOT NULL DEFAULT 1000000 CONSTRAINT chk_group_price_multiplier CHECK (price_multiplier_micros >= 0 AND price_multiplier_micros <= 1000000000)`, clause.Table{Name: "groups"}).Error; err != nil {
						return err
					}
					return fmt.Errorf("simulated interruption after first price multiplier column")
				}
				entries := append([]migration(nil), migrations[:8]...)
				entries = append(entries, entry)
				if err := applyMigrationRegistry(db, entries); err == nil {
					t.Fatal("interrupted migration succeeded")
				}
				if db.Migrator().HasColumn("groups", "price_multiplier_micros") {
					t.Fatal("transactional driver did not roll back interrupted DDL")
				}
			}
			if err := AutoMigrate(db); err != nil {
				t.Fatal(err)
			}
			if err := AutoMigrate(db); err != nil {
				t.Fatal(err)
			}
			if err := migrationfiles.Validate0009(db); err != nil {
				t.Fatal(err)
			}
			assertInternalMigrationComplete(t, db, registeredMigrationIDs())
			if mode == "fresh" {
				return
			}
			for _, table := range []string{"groups", "access_keys"} {
				var value int64
				if err := db.Table(table).Select("price_multiplier_micros").Where("id = 1").Scan(&value).Error; err != nil {
					t.Fatal(err)
				}
				if value != 1_000_000 {
					t.Fatalf("%s legacy multiplier = %d", table, value)
				}
				for _, valid := range []int64{0, 125_000, 1_000_000_000} {
					if err := db.Table(table).Where("id = 1").Update("price_multiplier_micros", valid).Error; err != nil {
						t.Fatalf("%s rejects %d: %v", table, valid, err)
					}
				}
				for _, invalid := range []any{int64(-1), int64(1_000_000_001), nil} {
					if err := db.Table(table).Where("id = 1").Update("price_multiplier_micros", invalid).Error; err == nil {
						t.Fatalf("%s accepted invalid multiplier %v", table, invalid)
					}
				}
			}
		})
	}
}

// Shared SQLite migration test helpers preserved from the removed external
// database migration contract test.

func openInternalMigrationTestDatabase(t *testing.T) *gorm.DB {
	t.Helper()
	db, err := Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	return db
}

func assertInternalMigrationComplete(t *testing.T, db *gorm.DB, wantIDs []string) {
	t.Helper()
	for _, table := range migrationfiles.TableNames0001() {
		if !db.Migrator().HasTable(table) {
			t.Errorf("table %q is missing", table)
		}
	}
	if len(wantIDs) >= 2 {
		for _, table := range migrationfiles.TableNames0002() {
			if !db.Migrator().HasTable(table) {
				t.Errorf("table %q is missing", table)
			}
		}
	}
	if len(wantIDs) >= 3 && db.Migrator().HasColumn("credential_observations", "fresh_until_ms") {
		t.Error("credential_observations.fresh_until_ms remains after migration 0003")
	}
	if len(wantIDs) >= 4 && !db.Migrator().HasIndex("usage_stats", "idx_usage_stats_group_bucket") {
		t.Error("usage_stats group activity index is missing after migration 0004")
	}
	if len(wantIDs) >= 6 {
		for _, column := range []string{
			"failure_origin", "failure_scope", "retry_directive", "effect", "rule_id",
		} {
			if !db.Migrator().HasColumn("request_log_attempts", column) {
				t.Errorf("request_log_attempts.%s is missing after migration 0006", column)
			}
		}
	}
	if len(wantIDs) >= 7 && !db.Migrator().HasColumn("access_keys", "expires_at_ms") {
		t.Error("access_keys.expires_at_ms is missing after migration 0007")
	}
	var ids []string
	if err := db.Table(migrationLedgerTable).Order("id").Pluck("id", &ids).Error; err != nil {
		t.Fatal(err)
	}
	if len(ids) != len(wantIDs) {
		t.Fatalf("migration IDs = %v, want %v", ids, wantIDs)
	}
	for index := range wantIDs {
		if ids[index] != wantIDs[index] {
			t.Fatalf("migration IDs = %v, want %v", ids, wantIDs)
		}
	}
}

func registeredMigrationIDs() []string {
	result := make([]string, 0, len(migrations))
	for _, entry := range migrations {
		result = append(result, entry.ID)
	}
	return result
}
