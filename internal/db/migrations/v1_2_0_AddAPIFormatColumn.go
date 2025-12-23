package db

import (
	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
)

// V1_2_0_AddAPIFormatColumn adds api_format column to groups table
// This migration supports the multi-format API transformer feature
func V1_2_0_AddAPIFormatColumn(db *gorm.DB) error {
	// Check if column already exists
	var columnExists bool

	switch db.Dialector.Name() {
	case "mysql":
		var count int64
		db.Raw(`
			SELECT COUNT(*)
			FROM information_schema.COLUMNS
			WHERE TABLE_SCHEMA = DATABASE()
			AND TABLE_NAME = 'groups'
			AND COLUMN_NAME = 'api_format'
		`).Count(&count)
		columnExists = count > 0
	case "sqlite":
		// For SQLite, check pragma table_info
		type ColumnInfo struct {
			Name string
		}
		var columns []ColumnInfo
		db.Raw("PRAGMA table_info(groups)").Scan(&columns)
		for _, col := range columns {
			if col.Name == "api_format" {
				columnExists = true
				break
			}
		}
	default:
		// For PostgreSQL and others
		var count int64
		db.Raw(`
			SELECT COUNT(*)
			FROM information_schema.columns
			WHERE table_name = 'groups'
			AND column_name = 'api_format'
		`).Count(&count)
		columnExists = count > 0
	}

	if columnExists {
		logrus.Info("Column api_format already exists in groups table, skipping v1.2.0...")
		return nil
	}

	logrus.Info("Adding api_format column to groups table...")

	// Add the column
	if err := db.Exec("ALTER TABLE groups ADD COLUMN api_format VARCHAR(50) DEFAULT ''").Error; err != nil {
		return err
	}

	logrus.Info("Migration v1.2.0 completed successfully")
	return nil
}
