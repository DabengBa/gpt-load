package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0013 = "0013_provider_url"

type groupProviderURL0013 struct {
	ProviderURL *string `gorm:"column:provider_url;type:text"`
}

func (groupProviderURL0013) TableName() string { return "groups" }

// Up0013 adds optional provider metadata storage without changing existing rows.
func Up0013(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("provider url migration: database is nil")
	}
	if db.Migrator().HasColumn(&groupProviderURL0013{}, "provider_url") {
		return Validate0013(db)
	}
	if err := db.Migrator().AddColumn(&groupProviderURL0013{}, "ProviderURL"); err != nil {
		return fmt.Errorf("add groups.provider_url: %w", err)
	}
	return Validate0013(db)
}

func Validate0013(db *gorm.DB) error {
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
