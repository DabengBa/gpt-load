package migrations

import (
	"fmt"
	"gorm.io/gorm"
	"strings"
)

const ID0025 = "0025_affinity_diagnostics"

var affinityDiagnosticColumns0025 = []struct{ name, definition string }{
	{"resolved_affinity_credential_id", "BIGINT NOT NULL DEFAULT 0"},
	{"first_attempt_credential_id", "BIGINT NOT NULL DEFAULT 0"},
	{"affinity_hit_assignment_reached", "BOOLEAN NOT NULL DEFAULT false"},
}

func Up0025(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("affinity diagnostics migration: unsupported or nil database")
	}
	if !db.Migrator().HasTable("request_logs") {
		return fmt.Errorf("affinity diagnostics migration: request_logs is missing")
	}
	for _, column := range affinityDiagnosticColumns0025 {
		if !db.Migrator().HasColumn("request_logs", column.name) {
			if err := db.Exec("ALTER TABLE request_logs ADD COLUMN " + column.name + " " + column.definition).Error; err != nil {
				return fmt.Errorf("add request_logs.%s: %w", column.name, err)
			}
		}
	}
	return Validate0025(db)
}

func Validate0025(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("validate affinity diagnostics migration: unsupported or nil database")
	}
	for _, column := range affinityDiagnosticColumns0025 {
		if !db.Migrator().HasColumn("request_logs", column.name) {
			return fmt.Errorf("validate affinity diagnostics migration: request_logs.%s is missing", column.name)
		}
	}
	return nil
}
