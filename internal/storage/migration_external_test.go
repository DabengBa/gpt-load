package storage

import (
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/platform/config"
	migrationfiles "gpt-load/internal/storage/migrations"
)

// TestExternalDatabaseIncrementalMigrations is opt-in so ordinary unit tests
// remain hermetic. It exercises interrupted migration recovery on PostgreSQL,
// whose advisory lock protects concurrent application startup.
func TestExternalDatabaseIncrementalMigrations(t *testing.T) {
	rawDSN := strings.TrimSpace(os.Getenv("GPT_LOAD_DATABASE_TEST_DSN"))
	if rawDSN == "" {
		t.Skip("GPT_LOAD_DATABASE_TEST_DSN is not set")
	}
	db := openExternalIncrementalMigrationDatabase(t, rawDSN)
	if err := db.AutoMigrate(&schemaMigration{}); err != nil {
		t.Fatalf("create migration ledger: %v", err)
	}
	if err := migrations[0].Up(db); err != nil {
		t.Fatalf("create 0001 schema: %v", err)
	}
	if err := migrations[0].Validate(db); err != nil {
		t.Fatalf("validate 0001 schema: %v", err)
	}
	if err := db.Create(&schemaMigration{ID: migrations[0].ID}).Error; err != nil {
		t.Fatalf("record 0001 migration: %v", err)
	}

	if err := AutoMigrate(db); err != nil {
		t.Fatalf("apply pending incremental migrations: %v", err)
	}
	if err := AutoMigrate(db); err != nil {
		t.Fatalf("repeat migrated schema validation: %v", err)
	}
	assertInternalMigrationComplete(t, db, registeredMigrationIDs())
}

func openExternalIncrementalMigrationDatabase(t *testing.T, rawDSN string) *gorm.DB {
	t.Helper()
	parsed, err := url.Parse(rawDSN)
	if err != nil {
		t.Fatalf("parse external database DSN: %v", err)
	}
	if !strings.EqualFold(parsed.Scheme, "postgres") && !strings.EqualFold(parsed.Scheme, "postgresql") {
		t.Fatalf("unsupported external database driver %q; expected PostgreSQL", parsed.Scheme)
	}

	admin, err := OpenWithSource(rawDSN, config.DatabaseSourceExternal)
	if err != nil {
		t.Fatalf("open PostgreSQL admin connection: %v", err)
	}
	adminSQL, err := admin.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = adminSQL.Close() })

	name := fmt.Sprintf("gpt_load_migration_%d", time.Now().UnixNano())
	if err := admin.Exec(`CREATE SCHEMA "` + name + `"`).Error; err != nil {
		t.Fatalf("create PostgreSQL migration schema: %v", err)
	}
	t.Cleanup(func() {
		if dropErr := admin.Exec(`DROP SCHEMA IF EXISTS "` + name + `" CASCADE`).Error; dropErr != nil {
			t.Errorf("drop PostgreSQL migration schema: %v", dropErr)
		}
	})

	query := parsed.Query()
	query.Set("search_path", name)
	parsed.RawQuery = query.Encode()
	db, err := OpenWithSource(parsed.String(), config.DatabaseSourceExternal)
	if err != nil {
		t.Fatalf("open isolated incremental migration database: %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	return db
}

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
