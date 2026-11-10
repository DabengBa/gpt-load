package migrations

import (
	"fmt"

	"gorm.io/gorm"
)

const ID0031 = "0031_announcement_source_errors"

func Up0031(db *gorm.DB) error {
	for _, name := range []string{"announcement_last_error", "notice_last_error"} {
		if !db.Migrator().HasColumn("group_site_configs", name) {
			if err := db.Exec("ALTER TABLE group_site_configs ADD COLUMN " + name + " text NOT NULL DEFAULT ''").Error; err != nil {
				return err
			}
		}
	}
	return Validate0031(db)
}

func Validate0031(db *gorm.DB) error {
	for _, name := range []string{"announcement_last_error", "notice_last_error"} {
		if !db.Migrator().HasColumn("group_site_configs", name) {
			return fmt.Errorf("announcement source column %s missing", name)
		}
	}
	return nil
}
