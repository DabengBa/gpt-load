package migrations_test

import (
	"testing"

	"gpt-load/internal/storage/migrations"
)

func TestUsageFirstResponseMigrationInstallsAndValidatesAggregateColumns(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatalf("Up0001() error = %v", err)
	}
	if err := migrations.Up0011(db); err != nil {
		t.Fatalf("Up0011() error = %v", err)
	}
	if err := migrations.Up0022(db); err != nil {
		t.Fatalf("first Up0022() error = %v", err)
	}
	if err := migrations.Validate0022(db); err != nil {
		t.Fatalf("first Validate0022() error = %v", err)
	}
	for _, table := range []string{"usage_aggregation_journal", "usage_stats"} {
		for _, column := range []string{"first_response_ms_total", "first_response_sample_count"} {
			if !db.Migrator().HasColumn(table, column) {
				t.Fatalf("%s.%s is missing", table, column)
			}
		}
	}
	if err := migrations.Up0022(db); err != nil {
		t.Fatalf("repeat Up0022() error = %v", err)
	}
	if err := migrations.Validate0022(db); err != nil {
		t.Fatalf("repeat Validate0022() error = %v", err)
	}
}
