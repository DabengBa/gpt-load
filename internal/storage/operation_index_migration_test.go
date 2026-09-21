package storage

import (
	"fmt"
	"testing"

	"gorm.io/gorm"

	migrationfiles "gpt-load/internal/storage/migrations"
)

func TestOperationIndexMigrationContract(t *testing.T) {
	t.Parallel()
	testOperationIndexMigration(t, openInternalMigrationTestDatabase)
}

func migrationIndexOf(t *testing.T, id string) int {
	t.Helper()
	for index, entry := range migrations {
		if entry.ID == id {
			return index
		}
	}
	t.Fatalf("migration %q is not registered", id)
	return -1
}

func testOperationIndexMigration(t *testing.T, open func(*testing.T) *gorm.DB) {
	for _, scenario := range []string{"fresh", "existing", "interrupted"} {
		t.Run(scenario, func(t *testing.T) {
			db := open(t)
			operationIndex := migrationIndexOf(t, migrationfiles.ID0019)
			if scenario != "fresh" {
				if err := applyMigrationRegistry(db, migrations[:operationIndex]); err != nil {
					t.Fatal(err)
				}
				if err := db.Table("request_logs").Create(map[string]any{
					"id": "existing", "completed_at_ms": 1, "access_key_id": 1,
					"protocol": "openai-responses", "operation": "responses_retrieve",
					"client_model": "test", "upstream_model": "test", "status": "error",
					"status_code": 500, "duration_ms": 1, "error_summary": "existing diagnostic",
				}).Error; err != nil {
					t.Fatal(err)
				}
				if scenario == "interrupted" {
					registry := append([]migration(nil), migrations[:operationIndex+1]...)
					up := registry[operationIndex].Up
					registry[len(registry)-1].Up = func(tx *gorm.DB) error {
						if err := up(tx); err != nil {
							return err
						}
						return fmt.Errorf("interrupt after operation index DDL")
					}
					if err := applyMigrationRegistry(db, registry); err == nil {
						t.Fatal("expected migration interruption")
					}
					if db.Migrator().HasIndex("request_logs", "idx_request_logs_operation_completed_id") {
						t.Fatal("transactional driver did not roll back interrupted DDL")
					}
				}
			}
			for range 2 {
				if err := AutoMigrate(db); err != nil {
					t.Fatal(err)
				}
				if !db.Migrator().HasIndex("request_logs", "idx_request_logs_operation_completed_id") {
					t.Fatal("operation cursor index is missing")
				}
				if scenario != "fresh" {
					var operation string
					if err := db.Table("request_logs").
						Where("id = ?", "existing").
						Pluck("operation", &operation).Error; err != nil ||
						operation != "responses_retrieve" {
						t.Fatalf("existing operation = %q, error = %v", operation, err)
					}
				}
			}
		})
	}
	t.Run("unexpected index definition", func(t *testing.T) {
		db := open(t)
		operationIndex := migrationIndexOf(t, migrationfiles.ID0019)
		if err := applyMigrationRegistry(db, migrations[:operationIndex]); err != nil {
			t.Fatal(err)
		}
		if err := db.Exec(
			"CREATE INDEX idx_request_logs_operation_completed_id ON request_logs (status)",
		).Error; err != nil {
			t.Fatal(err)
		}
		if err := AutoMigrate(db); err == nil {
			t.Fatal("unexpected operation index definition accepted")
		}
	})
}
