package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0015 = "0015_provider_url"

type groupProviderURL0015 struct {
	ProviderURL *string `gorm:"column:provider_url;type:text"`
}

func (groupProviderURL0015) TableName() string { return "groups" }

// Up0015 adds optional provider metadata storage without changing existing rows.
func Up0015(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("provider url migration: database is nil")
	}
	if db.Migrator().HasColumn(&groupProviderURL0015{}, "provider_url") {
		return Validate0015(db)
	}
	if err := db.Migrator().AddColumn(&groupProviderURL0015{}, "ProviderURL"); err != nil {
		return fmt.Errorf("add groups.provider_url: %w", err)
	}
	return Validate0015(db)
}

func Validate0015(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate provider url schema: database is nil")
	}
	if !db.Migrator().HasColumn("groups", "provider_url") {
		return fmt.Errorf("validate provider url: column groups.provider_url is missing")
	}
	columns, err := db.Migrator().ColumnTypes("groups")
	if err != nil {
		return fmt.Errorf("inspect groups.provider_url: %w", err)
	}
	for _, column := range columns {
		if !strings.EqualFold(column.Name(), "provider_url") {
			continue
		}
		typeName := strings.ToLower(column.DatabaseTypeName())
		if !strings.Contains(typeName, "text") && !strings.Contains(typeName, "char") && typeName != "clob" {
			return fmt.Errorf("validate provider url: column groups.provider_url is not text")
		}
		if nullable, known := column.Nullable(); known && !nullable {
			return fmt.Errorf("validate provider url: column groups.provider_url is not nullable")
		}
		return nil
	}
	return fmt.Errorf("validate provider url: column groups.provider_url is missing")
}
