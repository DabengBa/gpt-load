package migrations

import (
	"fmt"

	"gorm.io/gorm"
)

const ID0017 = "0017_remove_validation_interval"

const (
	validationIntervalKey0017 = "validation_interval"
	systemSettingTable0017    = "system_settings"
)

type systemSetting0017 struct {
	Key string `gorm:"column:key;primaryKey"`
}

func (systemSetting0017) TableName() string { return systemSettingTable0017 }

// Up0017 removes the retired automatic validation interval from existing databases.
func Up0017(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("remove validation interval: database is nil")
	}
	if !db.Migrator().HasTable(&systemSetting0017{}) {
		return fmt.Errorf("remove validation interval: table %q is missing", systemSettingTable0017)
	}
	if err := db.Where("key = ?", validationIntervalKey0017).Delete(&systemSetting0017{}).Error; err != nil {
		return fmt.Errorf("delete %s system setting: %w", validationIntervalKey0017, err)
	}
	return Validate0017(db)
}

// Validate0017 verifies that no retired validation interval remains.
func Validate0017(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate validation interval removal: database is nil")
	}
	if !db.Migrator().HasTable(&systemSetting0017{}) {
		return fmt.Errorf("validate validation interval removal: table %q is missing", systemSettingTable0017)
	}
	var count int64
	if err := db.Model(&systemSetting0017{}).
		Where("key = ?", validationIntervalKey0017).
		Count(&count).Error; err != nil {
		return fmt.Errorf("count %s system settings: %w", validationIntervalKey0017, err)
	}
	if count != 0 {
		return fmt.Errorf("retired system setting %q remains", validationIntervalKey0017)
	}
	return nil
}
