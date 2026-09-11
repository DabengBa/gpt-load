package migrations

import (
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/storage/models"
)

const ID0013 = "0013_usage_attempt_stats"

var usageAttemptStatsTables0013 = []struct {
	name  string
	model any
}{
	{"usage_attempt_aggregation_journals", &models.UsageAttemptAggregationJournal{}},
	{"usage_attempt_stats", &models.UsageAttemptStat{}},
}

// Up0013 adds durable route-attempt aggregates without changing request-level
// usage statistics.
func Up0013(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("usage attempt stats migration: database is nil")
	}
	for _, table := range usageAttemptStatsTables0013 {
		if err := db.AutoMigrate(table.model); err != nil {
			return fmt.Errorf("create %s: %w", table.name, err)
		}
	}
	return Validate0013(db)
}

var requiredColumns0013 = map[string][]string{
	"usage_attempt_aggregation_journals": {
		"request_id", "sequence", "bucket_start_ms", "access_key_id", "group_id", "channel_id",
		"credential_id", "model", "attempt_count", "failure_count", "applied",
	},
	"usage_attempt_stats": {
		"id", "bucket_start_ms", "access_key_id", "channel_id", "group_id", "credential_id",
		"model", "attempt_count", "failure_count",
	},
}

// Validate0013 verifies that both durable route-attempt tables and their
// required columns are present.
func Validate0013(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate usage attempt stats: database is nil")
	}
	for _, table := range usageAttemptStatsTables0013 {
		if !db.Migrator().HasTable(table.name) {
			return fmt.Errorf("usage attempt stats table %q is missing", table.name)
		}
		for _, column := range requiredColumns0013[table.name] {
			if !db.Migrator().HasColumn(table.name, column) {
				return fmt.Errorf("usage attempt stats column %s.%s is missing", table.name, column)
			}
		}
	}
	return nil
}
