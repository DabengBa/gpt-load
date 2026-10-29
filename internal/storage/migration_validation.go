package storage

import (
	"fmt"

	"gorm.io/gorm"
)

// validateMigrationForeignKeys performs SQLite's integrity check for the
// migration that just rewrote schema. It only runs while applying pending
// migrations inside the migration transaction; a start that only revalidates
// the ledger leaves existing rows to SQLite's per-write enforcement.
func validateMigrationForeignKeys(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || db.Dialector.Name() != "sqlite" {
		return nil
	}
	var violations []struct {
		Table string
		RowID int64 `gorm:"column:rowid"`
	}
	if err := db.Raw("PRAGMA foreign_key_check").Scan(&violations).Error; err != nil {
		return fmt.Errorf("validate migration foreign keys: %w", err)
	}
	if len(violations) != 0 {
		return fmt.Errorf("validate migration foreign keys: %d violation(s)", len(violations))
	}
	return nil
}
