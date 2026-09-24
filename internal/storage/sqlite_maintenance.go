package storage

import (
	"context"
	"fmt"

	"gorm.io/gorm"
)

// SQLiteMaintenanceStatus is an observation, not a claim that DELETE returns disk space.
// VACUUM is deliberately never invoked by InspectSQLite or request retention.
type SQLiteMaintenanceStatus struct {
	JournalMode     string `json:"journal_mode"`
	FreelistPages   int64  `json:"freelist_pages"`
	MaintenanceMode string `json:"maintenance_mode"`
}

func InspectSQLite(ctx context.Context, db *gorm.DB) (SQLiteMaintenanceStatus, error) {
	if db == nil || db.Dialector == nil || db.Dialector.Name() != "sqlite" {
		return SQLiteMaintenanceStatus{}, fmt.Errorf("inspect SQLite: SQLite database required")
	}
	if ctx == nil {
		return SQLiteMaintenanceStatus{}, fmt.Errorf("inspect SQLite: context required")
	}
	status := SQLiteMaintenanceStatus{MaintenanceMode: "offline_only"}
	if err := db.WithContext(ctx).Raw("PRAGMA journal_mode").Scan(&status.JournalMode).Error; err != nil {
		return SQLiteMaintenanceStatus{}, fmt.Errorf("inspect SQLite journal mode: %w", err)
	}
	if err := db.WithContext(ctx).Raw("PRAGMA freelist_count").Scan(&status.FreelistPages).Error; err != nil {
		return SQLiteMaintenanceStatus{}, fmt.Errorf("inspect SQLite freelist: %w", err)
	}
	return status, nil
}
