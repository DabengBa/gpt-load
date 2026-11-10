package migrations

import (
	"fmt"
	"gorm.io/gorm"
)

const ID0030 = "0030_announcement_source_status"

func Up0030(db *gorm.DB) error {
	existing := db.Migrator().HasColumn("group_site_configs", "announcement_status")
	for _, column := range []struct{ name, definition string }{
		{"announcement_status", "varchar(32) NOT NULL DEFAULT 'pending'"},
		{"notice_status", "varchar(32) NOT NULL DEFAULT 'pending'"},
		{"announcement_fetched_at_ms", "integer"},
		{"notice_fetched_at_ms", "integer"},
	} {
		if !db.Migrator().HasColumn("group_site_configs", column.name) {
			if err := db.Exec("ALTER TABLE group_site_configs ADD COLUMN " + column.name + " " + column.definition).Error; err != nil {
				return err
			}
		}
	}
	if !existing {
		if err := db.Exec("UPDATE group_site_configs SET announcement_status = 'unknown', notice_status = 'unknown' WHERE last_attempt_at_ms IS NOT NULL OR fetched_at_ms IS NOT NULL OR EXISTS (SELECT 1 FROM site_announcements WHERE site_announcements.group_id = group_site_configs.group_id)").Error; err != nil {
			return err
		}
	}
	return Validate0030(db)
}

func Validate0030(db *gorm.DB) error {
	for _, name := range []string{"announcement_status", "notice_status", "announcement_fetched_at_ms", "notice_fetched_at_ms"} {
		if !db.Migrator().HasColumn("group_site_configs", name) {
			return fmt.Errorf("announcement source column %s missing", name)
		}
	}
	return nil
}
