package migrations

import (
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/storage/models"
)

const ID0010 = "0010_debug_captures"

var debugCaptureTables0010 = []any{
	&models.DebugCapture{},
	&models.DebugCaptureAttempt{},
	&models.DebugCaptureChunk{},
}

// Up0010 installs the database-only raw capture store. AutoMigrate is used
// here so SQLite, MySQL, and PostgreSQL receive their native binary type
// (BLOB, VARBINARY/BLOB, and bytea respectively) from the GORM dialector.
func Up0010(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("debug capture migration: database is nil")
	}
	if err := db.AutoMigrate(debugCaptureTables0010...); err != nil {
		return fmt.Errorf("create debug capture schema: %w", err)
	}
	return Validate0010(db)
}

// ValidateRecoverable0010 permits an interrupted AutoMigrate to be retried,
// while rejecting a same-named table that cannot be completed safely.
func ValidateRecoverable0010(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate recoverable debug capture schema: database is nil")
	}
	for _, table := range debugCaptureTables0010 {
		name := tableName0010(table)
		if !db.Migrator().HasTable(name) {
			continue
		}
		if err := validateColumns0010(db, name, requiredColumns0010[name]); err != nil {
			return err
		}
	}
	return nil
}

func Validate0010(db *gorm.DB) error {
	if err := ValidateRecoverable0010(db); err != nil {
		return err
	}
	for _, table := range debugCaptureTables0010 {
		name := tableName0010(table)
		if !db.Migrator().HasTable(name) {
			return fmt.Errorf("debug capture table %q is missing", name)
		}
		if err := validateColumns0010(db, name, requiredColumns0010[name]); err != nil {
			return err
		}
	}
	for _, index := range []struct{ table, name string }{
		{"debug_captures", "idx_debug_captures_request_id"},
		{"debug_captures", "idx_debug_captures_access_key_id"},
		{"debug_captures", "idx_debug_captures_protocol"},
		{"debug_captures", "idx_debug_captures_operation"},
		{"debug_captures", "idx_debug_captures_created_at_ms"},
		{"debug_captures", "idx_debug_captures_expires_at_ms"},
		{"debug_captures", "idx_debug_captures_state"},
		{"debug_capture_chunks", "idx_debug_capture_chunks_attempt_part_direction_id"},
	} {
		if !db.Migrator().HasIndex(index.table, index.name) {
			return fmt.Errorf("debug capture index %q on %q is missing", index.name, index.table)
		}
	}
	if !db.Migrator().HasConstraint("debug_capture_attempts", "fk_debug_captures_attempts") {
		return fmt.Errorf("debug capture attempt foreign key is missing")
	}
	if !db.Migrator().HasConstraint("debug_capture_chunks", "fk_debug_capture_chunks_capture") {
		return fmt.Errorf("debug capture chunk capture foreign key is missing")
	}
	if !db.Migrator().HasConstraint("debug_capture_chunks", "fk_debug_capture_attempts_chunks") {
		return fmt.Errorf("debug capture composite capture/attempt foreign key is missing")
	}
	return nil
}

var requiredColumns0010 = map[string][]string{
	"debug_captures":         {"id", "request_id", "access_key_id", "protocol", "operation", "metadata", "created_at_ms", "expires_at_ms", "state", "error", "terminal_at_ms"},
	"debug_capture_attempts": {"id", "capture_id", "sequence", "metadata", "started_at_ms", "completed_at_ms", "state", "error"},
	"debug_capture_chunks":   {"id", "capture_id", "attempt_id", "part", "direction", "data", "created_at_ms"},
}

func validateColumns0010(db *gorm.DB, table string, columns []string) error {
	for _, column := range columns {
		if !db.Migrator().HasColumn(table, column) {
			return fmt.Errorf("debug capture column %s.%s is missing", table, column)
		}
	}
	return nil
}

func tableName0010(value any) string {
	switch value.(type) {
	case *models.DebugCapture:
		return "debug_captures"
	case *models.DebugCaptureAttempt:
		return "debug_capture_attempts"
	default:
		return "debug_capture_chunks"
	}
}
