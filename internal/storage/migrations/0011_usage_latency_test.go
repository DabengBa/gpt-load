package migrations_test

import (
	"testing"

	"gorm.io/gorm"

	"gpt-load/internal/storage/migrations"
)

func TestUsageLatencyMigrationAddsValidatedColumnsAndIsIdempotent(t *testing.T) {
	db := openInitialTestDatabase(t)
	for _, up := range []func(*gorm.DB) error{
		migrations.Up0001, migrations.Up0002, migrations.Up0003, migrations.Up0004,
		migrations.Up0005, migrations.Up0006, migrations.Up0007, migrations.Up0008,
		migrations.Up0009, migrations.Up0010,
	} {
		if err := up(db); err != nil {
			t.Fatal(err)
		}
	}
	if err := migrations.Up0011(db); err != nil {
		t.Fatalf("Up0011() error = %v", err)
	}
	for _, table := range []string{"usage_aggregation_journal", "usage_stats"} {
		for _, column := range []string{"duration_ms_total", "duration_sample_count"} {
			if !db.Migrator().HasColumn(table, column) {
				t.Fatalf("%s.%s is missing", table, column)
			}
		}
	}
	if err := migrations.Validate0011(db); err != nil {
		t.Fatalf("Validate0011() error = %v", err)
	}
	if err := migrations.Up0011(db); err != nil {
		t.Fatalf("repeated Up0011() error = %v", err)
	}

	invalidInsert := `
		INSERT INTO usage_stats (
			bucket_start_ms, access_key_id, channel_id, group_id, credential_id, model,
			request_count, success_count, failure_count, duration_sample_count
		) VALUES (1, 1, '', 1, 0, 'invalid-latency', 1, 1, 0, 2)
	`
	if err := db.Exec(invalidInsert).Error; err == nil {
		t.Fatal("usage_stats CHECK accepted duration_sample_count greater than request_count")
	}
	if err := db.Exec(`PRAGMA ignore_check_constraints = ON`).Error; err != nil {
		t.Fatalf("disable SQLite CHECK constraints: %v", err)
	}
	defer func() {
		if err := db.Exec(`PRAGMA ignore_check_constraints = OFF`).Error; err != nil {
			t.Errorf("restore SQLite CHECK constraints: %v", err)
		}
	}()
	if err := db.Exec(invalidInsert).Error; err != nil {
		t.Fatalf("insert invalid duration sample count: %v", err)
	}
	if err := migrations.Validate0011(db); err == nil {
		t.Fatal("Validate0011() accepted duration_sample_count greater than request_count")
	}
}
