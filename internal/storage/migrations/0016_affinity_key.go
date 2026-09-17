package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0016 = "0016_affinity_key"

const (
	affinityKeyTable0016  = "request_logs"
	affinityKeyColumn0016 = "affinity_key"
	affinityKeyIndex0016  = "idx_request_logs_affinity_completed_id"
)

// Up0016 adds the canonical, non-sensitive affinity scope projection used by
// request-log filtering. Existing rows receive the empty bounded value.
func Up0016(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("affinity key migration: database is nil")
	}
	if !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("affinity key migration: unsupported database driver %q", db.Dialector.Name())
	}
	if err := ValidateRecoverable0016(db); err != nil {
		return err
	}
	if !db.Migrator().HasColumn(affinityKeyTable0016, affinityKeyColumn0016) {
		if err := db.Exec(
			"ALTER TABLE request_logs ADD COLUMN affinity_key VARCHAR(36) NOT NULL DEFAULT ''",
		).Error; err != nil {
			return fmt.Errorf("add request_logs.affinity_key: %w", err)
		}
	}
	if !db.Migrator().HasIndex(affinityKeyTable0016, affinityKeyIndex0016) {
		if err := db.Exec(
			"CREATE INDEX idx_request_logs_affinity_completed_id " +
				"ON request_logs (affinity_key, completed_at_ms DESC, id DESC)",
		).Error; err != nil {
			return fmt.Errorf("create request log affinity index: %w", err)
		}
	}
	return Validate0016(db)
}

// ValidateRecoverable0016 validates any pieces already present so an
// interrupted migration can safely resume without accepting an incompatible
// column or index.
func ValidateRecoverable0016(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate affinity key migration: database is nil")
	}
	if !db.Migrator().HasTable(affinityKeyTable0016) {
		return fmt.Errorf("affinity key table %q is missing", affinityKeyTable0016)
	}
	if db.Migrator().HasColumn(affinityKeyTable0016, affinityKeyColumn0016) {
		if err := validateAffinityKeyColumn0016(db); err != nil {
			return err
		}
	}
	if db.Migrator().HasIndex(affinityKeyTable0016, affinityKeyIndex0016) {
		if err := validateAffinityKeyIndex0016(db); err != nil {
			return err
		}
	}
	return nil
}

// Validate0016 confirms the complete affinity key schema contract.
func Validate0016(db *gorm.DB) error {
	if err := ValidateRecoverable0016(db); err != nil {
		return err
	}
	if !db.Migrator().HasColumn(affinityKeyTable0016, affinityKeyColumn0016) {
		return fmt.Errorf("request_logs.%s is missing", affinityKeyColumn0016)
	}
	if !db.Migrator().HasIndex(affinityKeyTable0016, affinityKeyIndex0016) {
		return fmt.Errorf("request_logs.%s is missing", affinityKeyIndex0016)
	}
	if err := validateAffinityKeyColumn0016(db); err != nil {
		return err
	}
	return validateAffinityKeyIndex0016(db)
}

func validateAffinityKeyColumn0016(db *gorm.DB) error {
	columns, err := db.Migrator().ColumnTypes(affinityKeyTable0016)
	if err != nil {
		return fmt.Errorf("inspect request_logs.%s: %w", affinityKeyColumn0016, err)
	}
	for _, column := range columns {
		if !strings.EqualFold(column.Name(), affinityKeyColumn0016) {
			continue
		}
		if nullable, known := column.Nullable(); !known || nullable {
			return fmt.Errorf("request_logs.%s is nullable", affinityKeyColumn0016)
		}
		dbType := strings.ToLower(column.DatabaseTypeName())
		if !strings.Contains(dbType, "char") && !strings.Contains(dbType, "text") {
			return fmt.Errorf("request_logs.%s type %q is not textual", affinityKeyColumn0016, dbType)
		}
		value, known := affinityKeyColumnDefault0016(db)
		if !known || value != "" {
			return fmt.Errorf("request_logs.%s default %q is not empty", affinityKeyColumn0016, value)
		}
		return nil
	}
	return fmt.Errorf("request_logs.%s is missing", affinityKeyColumn0016)
}

func affinityKeyColumnDefault0016(db *gorm.DB) (string, bool) {
	if !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return "", false
	}
	var defaultValue string
	if err := db.Raw(
		"SELECT dflt_value FROM pragma_table_info(?) WHERE name = ?",
		affinityKeyTable0016,
		affinityKeyColumn0016,
	).Scan(&defaultValue).Error; err != nil {
		return "", false
	}
	return normalizeAffinityKeyDefault0016(defaultValue), defaultValue != ""
}

func normalizeAffinityKeyDefault0016(value string) string {
	value = strings.Split(value, "::")[0]
	return strings.ToLower(strings.TrimSpace(strings.Trim(value, "()' \"`")))
}

func validateAffinityKeyIndex0016(db *gorm.DB) error {
	switch strings.ToLower(db.Dialector.Name()) {
	case "sqlite":
		return validateAffinityKeySQLiteIndex0016(db)
	default:
		return fmt.Errorf("validate affinity key index: unsupported database driver %q", db.Dialector.Name())
	}
}

func validateAffinityKeySQLiteIndex0016(db *gorm.DB) error {
	var indexes []struct {
		Name   string
		Unique int
	}
	if err := db.Raw("PRAGMA index_list('request_logs')").Scan(&indexes).Error; err != nil {
		return fmt.Errorf("inspect request log affinity indexes: %w", err)
	}
	for _, index := range indexes {
		if index.Name != affinityKeyIndex0016 {
			continue
		}
		if index.Unique != 0 {
			return fmt.Errorf("request log affinity index is unique")
		}
		var columns []struct {
			Name string
			Key  int
			Desc int
		}
		if err := db.Raw("PRAGMA index_xinfo('idx_request_logs_affinity_completed_id')").Scan(&columns).Error; err != nil {
			return fmt.Errorf("inspect request log affinity index columns: %w", err)
		}
		var names []string
		for _, column := range columns {
			if column.Key == 1 {
				names = append(names, column.Name)
				if len(names) > 1 && column.Desc != 1 {
					return fmt.Errorf("request log affinity index column %q is not descending", column.Name)
				}
			}
		}
		want := []string{"affinity_key", "completed_at_ms", "id"}
		if len(names) != len(want) {
			return fmt.Errorf("request log affinity index columns = %v, want %v", names, want)
		}
		for index := range want {
			if !strings.EqualFold(names[index], want[index]) {
				return fmt.Errorf("request log affinity index columns = %v, want %v", names, want)
			}
		}
		return nil
	}
	return fmt.Errorf("request log affinity index %q is missing", affinityKeyIndex0016)
}
