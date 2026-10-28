package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0027 = "0027_success_usage_efficiency"

func Up0027(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("success usage efficiency migration: unsupported or nil database")
	}
	for _, table := range []string{"usage_aggregation_journal", "usage_stats"} {
		if !db.Migrator().HasTable(table) {
			return fmt.Errorf("success usage efficiency migration: %s is missing", table)
		}
		for _, column := range []struct{ name, definition string }{
			{"success_output_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK (success_output_tokens >= 0)"},
			{"success_duration_ms_total", "INTEGER NOT NULL DEFAULT 0 CHECK (success_duration_ms_total >= 0)"},
			{"success_efficiency_sample_count", "INTEGER NOT NULL DEFAULT 0 CHECK (success_efficiency_sample_count >= 0 AND success_efficiency_sample_count <= success_count)"},
		} {
			if !db.Migrator().HasColumn(table, column.name) {
				if err := db.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column.name, column.definition)).Error; err != nil {
					return fmt.Errorf("add %s.%s: %w", table, column.name, err)
				}
			}
		}
	}
	return Validate0027(db)
}

func Validate0027(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("validate success usage efficiency migration: unsupported or nil database")
	}
	for _, table := range []string{"usage_aggregation_journal", "usage_stats"} {
		if !db.Migrator().HasTable(table) {
			return fmt.Errorf("validate success usage efficiency migration: %s is missing", table)
		}
		for _, column := range []string{"success_output_tokens", "success_duration_ms_total", "success_efficiency_sample_count"} {
			if !db.Migrator().HasColumn(table, column) {
				return fmt.Errorf("validate success usage efficiency migration: %s.%s is missing", table, column)
			}
		}
	}
	return nil
}
