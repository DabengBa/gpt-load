package migrations

import (
	"fmt"
	"strconv"
	"strings"

	"gorm.io/gorm"
)

const ID0011 = "0011_usage_latency"

var usageLatencyTables0011 = []struct {
	table, totalConstraint, sampleConstraint string
}{
	{"usage_aggregation_journal", "chk_usage_journal_duration_total", "chk_usage_journal_duration_samples"},
	{"usage_stats", "chk_usage_stat_duration_total", "chk_usage_stat_duration_samples"},
}

// Up0011 adds durable latency aggregates to both usage persistence tables.
func Up0011(db *gorm.DB) error {
	if err := ValidateRecoverable0011(db); err != nil {
		return err
	}
	for _, table := range usageLatencyTables0011 {
		if err := ensureUsageLatencyColumn0011(db, table.table, "duration_ms_total", table.totalConstraint); err != nil {
			return err
		}
		if err := ensureUsageLatencyColumn0011(db, table.table, "duration_sample_count", table.sampleConstraint); err != nil {
			return err
		}
	}
	return Validate0011(db)
}

func ensureUsageLatencyColumn0011(db *gorm.DB, table, column, constraint string) error {
	if !db.Migrator().HasColumn(table, column) {
		statement := fmt.Sprintf(
			"ALTER TABLE %s ADD COLUMN %s BIGINT NOT NULL DEFAULT 0",
			quoteUsageLatencyIdentifier0011(db, table),
			quoteUsageLatencyIdentifier0011(db, column),
		)
		if strings.EqualFold(db.Dialector.Name(), "sqlite") {
			statement += fmt.Sprintf(
				" CONSTRAINT %s CHECK (%s)",
				quoteUsageLatencyIdentifier0011(db, constraint),
				usageLatencyConstraintExpression0011(db, column),
			)
		}
		if err := db.Exec(statement).Error; err != nil {
			return fmt.Errorf("add %s.%s: %w", table, column, err)
		}
	}
	if db.Migrator().HasConstraint(table, constraint) {
		return nil
	}
	if strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("%s.%s constraint %q is missing", table, column, constraint)
	}
	statement := fmt.Sprintf(
		"ALTER TABLE %s ADD CONSTRAINT %s CHECK (%s)",
		quoteUsageLatencyIdentifier0011(db, table),
		quoteUsageLatencyIdentifier0011(db, constraint),
		usageLatencyConstraintExpression0011(db, column),
	)
	if err := db.Exec(statement).Error; err != nil {
		return fmt.Errorf("add %s.%s constraint: %w", table, column, err)
	}
	return nil
}

func usageLatencyConstraintExpression0011(db *gorm.DB, column string) string {
	expression := quoteUsageLatencyIdentifier0011(db, column) + " >= 0"
	if column == "duration_sample_count" {
		expression += " AND " + quoteUsageLatencyIdentifier0011(db, column) + " <= " + quoteUsageLatencyIdentifier0011(db, "request_count")
	}
	return expression
}

// ValidateRecoverable0011 accepts the initial schema and each completed column
// addition, allowing a retry after a non-transactional DDL interruption.
func ValidateRecoverable0011(db *gorm.DB) error {
	for _, table := range usageLatencyTables0011 {
		if !db.Migrator().HasTable(table.table) {
			return fmt.Errorf("validate recoverable usage latency: table %q is missing", table.table)
		}
		for _, column := range []string{"duration_ms_total", "duration_sample_count"} {
			if !db.Migrator().HasColumn(table.table, column) {
				continue
			}
			constraint := table.totalConstraint
			if column == "duration_sample_count" {
				constraint = table.sampleConstraint
			}
			requireConstraint := strings.EqualFold(db.Dialector.Name(), "sqlite")
			if err := validateUsageLatencyColumn0011(db, table.table, column, constraint, requireConstraint); err != nil {
				return err
			}
		}
	}
	return nil
}

// Validate0011 verifies both non-negative latency aggregates are complete.
func Validate0011(db *gorm.DB) error {
	if err := ValidateRecoverable0011(db); err != nil {
		return err
	}
	for _, table := range usageLatencyTables0011 {
		for _, definition := range []struct{ column, constraint string }{
			{"duration_ms_total", table.totalConstraint},
			{"duration_sample_count", table.sampleConstraint},
		} {
			if !db.Migrator().HasColumn(table.table, definition.column) {
				return fmt.Errorf("%s.%s is missing", table.table, definition.column)
			}
			if err := validateUsageLatencyColumn0011(db, table.table, definition.column, definition.constraint, true); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateUsageLatencyColumn0011(db *gorm.DB, table, column, constraint string, requireConstraint bool) error {
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
		if requireConstraint {
			return fmt.Errorf("%s.%s constraint %q is missing", table, column, constraint)
		}
		return nil
	}
	definition, err := usageLatencyConstraintDefinition0011(db, table, constraint)
	if err != nil {
		return err
	}
	normalized := strings.NewReplacer(" ", "", "\n", "", "\t", "", "(", "", ")", "", "`", "", `"`, "", "::bigint", "").Replace(strings.ToLower(definition))
	expectedBounds := strings.ToLower(column) + ">=0"
	if column == "duration_sample_count" {
		expectedBounds += "and" + strings.ToLower(column) + "<=request_count"
	}
	if !strings.Contains(normalized, expectedBounds) {
		return fmt.Errorf("%s.%s constraint has invalid bounds", table, column)
	}
	invalidWhere := column + " IS NULL OR " + column + " < 0"
	if column == "duration_sample_count" {
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

func usageLatencyConstraintDefinition0011(db *gorm.DB, table, constraint string) (string, error) {
	var definition string
	var err error
	switch strings.ToLower(db.Dialector.Name()) {
	case "sqlite":
		err = db.Raw("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", table).Scan(&definition).Error
	case "mysql":
		err = db.Raw("SELECT CHECK_CLAUSE FROM information_schema.check_constraints WHERE constraint_schema = DATABASE() AND constraint_name = ?", constraint).Scan(&definition).Error
	case "postgres", "postgresql":
		err = db.Raw("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = ? AND conrelid = ?::regclass", constraint, table).Scan(&definition).Error
	default:
		return "", fmt.Errorf("unsupported usage latency migration driver %q", db.Dialector.Name())
	}
	if err != nil {
		return "", fmt.Errorf("inspect %s constraint %q: %w", table, constraint, err)
	}
	return definition, nil
}

func quoteUsageLatencyIdentifier0011(db *gorm.DB, value string) string {
	if strings.EqualFold(db.Dialector.Name(), "mysql") {
		return "`" + value + "`"
	}
	return `"` + value + `"`
}
