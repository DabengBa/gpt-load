package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0026 = "0026_debug_capture_chunk_compression"

func Up0026(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("debug capture compression migration: unsupported or nil database")
	}
	if !db.Migrator().HasTable("debug_capture_chunks") {
		return fmt.Errorf("debug capture compression migration: debug_capture_chunks is missing")
	}
	if !db.Migrator().HasColumn("debug_capture_chunks", "encoding") {
		if err := db.Exec("ALTER TABLE debug_capture_chunks ADD COLUMN encoding VARCHAR(16) NOT NULL DEFAULT 'identity'").Error; err != nil {
			return fmt.Errorf("add debug_capture_chunks.encoding: %w", err)
		}
	}
	return Validate0026(db)
}

func Validate0026(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("validate debug capture compression migration: unsupported or nil database")
	}
	if !db.Migrator().HasTable("debug_capture_chunks") || !db.Migrator().HasColumn("debug_capture_chunks", "encoding") {
		return fmt.Errorf("validate debug capture compression migration: debug_capture_chunks.encoding is missing")
	}
	return nil
}
