package migrations

import (
	"fmt"
	"strconv"
	"strings"

	"gorm.io/gorm"
)

const ID0022 = "0022_usage_first_response"

var usageFirstResponseTables0022 = []struct {
	table, totalConstraint, sampleConstraint string
}{
	{"usage_aggregation_journal", "chk_usage_journal_first_response_total", "chk_usage_journal_first_response_samples"},
	{"usage_stats", "chk_usage_stat_first_response_total", "chk_usage_stat_first_response_samples"},
}

// Up0022 adds durable first-response aggregates to the usage persistence tables.
func Up0022(db *gorm.DB) error {
	if db == nil || db.Dialector == nil {
		return fmt.Errorf("usage first response migration: database is nil")
	}
	if !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("usage first response migration: unsupported database driver %q", db.Dialector.Name())
	}
	for _, table := range usageFirstResponseTables0022 {
		if !db.Migrator().HasTable(table.table) {
			return fmt.Errorf("usage first response migration: table %q is missing", table.table)
		}
		if err := ensureUsageFirstResponseColumn0022(db, table.table, "first_response_ms_total", table.totalConstraint); err != nil {
			return err
		}
		if err := ensureUsageFirstResponseColumn0022(db, table.table, "first_response_sample_count", table.sampleConstraint); err != nil {
			return err
		}
	}
	return Validate0022(db)
}

func ensureUsageFirstResponseColumn0022(db *gorm.DB, table, column, constraint string) error {
	if !db.Migrator().HasColumn(table, column) {
		expression := quoteUsageFirstResponseIdentifier0022(column) + " >= 0"
		if column == "first_response_sample_count" {
			expression += " AND " + quoteUsageFirstResponseIdentifier0022(column) + " <= " + quoteUsageFirstResponseIdentifier0022("request_count")
		}
		statement := fmt.Sprintf(
			"ALTER TABLE %s ADD COLUMN %s BIGINT NOT NULL DEFAULT 0 CONSTRAINT %s CHECK (%s)",
			quoteUsageFirstResponseIdentifier0022(table),
			quoteUsageFirstResponseIdentifier0022(column),
			quoteUsageFirstResponseIdentifier0022(constraint),
			expression,
		)
		if err := db.Exec(statement).Error; err != nil {
			return fmt.Errorf("add %s.%s: %w", table, column, err)
		}
	}
	if db.Migrator().HasConstraint(table, constraint) {
		return nil
	}
	return fmt.Errorf("%s.%s constraint %q is missing", table, column, constraint)
}

// Validate0022 verifies both first-response aggregates are complete and bounded.
func Validate0022(db *gorm.DB) error {
	if db == nil || db.Dialector == nil {
		return fmt.Errorf("validate usage first response migration: database is nil")
	}
	for _, table := range usageFirstResponseTables0022 {
		if !db.Migrator().HasTable(table.table) {
			return fmt.Errorf("validate usage first response migration: table %q is missing", table.table)
		}
		for _, definition := range []struct {
			column, constraint string
		}{
			{"first_response_ms_total", table.totalConstraint},
			{"first_response_sample_count", table.sampleConstraint},
		} {
			if err := validateUsageFirstResponseColumn0022(db, table.table, definition.column, definition.constraint); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateUsageFirstResponseColumn0022(db *gorm.DB, table, column, constraint string) error {
	columns, err := db.Migrator().ColumnTypes(table)
	if err != nil {
		return fmt.Errorf("inspect %s.%s: %w", table, column, err)
	}
	found := false
	for _, definition := range columns {
		if !strings.EqualFold(definition.Name(), column) {
			continue
		}
		found = true
		if !strings.Contains(strings.ToLower(definition.DatabaseTypeName()), "int") {
			return fmt.Errorf("%s.%s is not integer", table, column)
		}
		if nullable, known := definition.Nullable(); !known || nullable {
			return fmt.Errorf("%s.%s is nullable", table, column)
		}
		defaultValue, known := definition.DefaultValue()
		if strings.EqualFold(db.Dialector.Name(), "sqlite") {
			if err := db.Raw("SELECT dflt_value FROM pragma_table_info(?) WHERE name = ?", table, column).Scan(&defaultValue).Error; err != nil {
				return fmt.Errorf("inspect %s.%s default: %w", table, column, err)
			}
			known = defaultValue != ""
		}
		defaultValue = strings.Trim(strings.Split(defaultValue, "::")[0], "()' \"")
		value, parseErr := strconv.ParseInt(defaultValue, 10, 64)
		if !known || parseErr != nil || value != 0 {
			return fmt.Errorf("%s.%s has invalid default %q (known %t)", table, column, defaultValue, known)
		}
	}
	if !found {
		return fmt.Errorf("%s.%s is missing", table, column)
	}
	if !db.Migrator().HasConstraint(table, constraint) {
		return fmt.Errorf("%s.%s constraint %q is missing", table, column, constraint)
	}
	definition, err := usageFirstResponseConstraintDefinition0022(db, table)
	if err != nil {
		return err
	}
	normalized := strings.NewReplacer(" ", "", "\n", "", "\t", "", "(", "", ")", "", "`", "", `"`, "", "::bigint", "").Replace(strings.ToLower(definition))
	expectedBounds := strings.ToLower(column) + ">=0"
	if column == "first_response_sample_count" {
		expectedBounds += "and" + strings.ToLower(column) + "<=request_count"
	}
	if !strings.Contains(normalized, expectedBounds) {
		return fmt.Errorf("%s.%s constraint has invalid bounds", table, column)
	}
	invalidWhere := column + " IS NULL OR " + column + " < 0"
	if column == "first_response_sample_count" {
		invalidWhere += " OR " + column + " > request_count"
	}
	var invalid int64
	if err := db.Table(table).Where(invalidWhere).Count(&invalid).Error; err != nil {
		return fmt.Errorf("validate %s.%s values: %w", table, column, err)
	}
	if invalid != 0 {
		return fmt.Errorf("%s contains invalid %s values", table, column)
	}
	return nil
}

func usageFirstResponseConstraintDefinition0022(db *gorm.DB, table string) (string, error) {
	var definition string
	if err := db.Raw("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", table).Scan(&definition).Error; err != nil {
		return "", fmt.Errorf("inspect %s first response constraints: %w", table, err)
	}
	return definition, nil
}

func quoteUsageFirstResponseIdentifier0022(value string) string {
	return `"` + value + `"`
}
