package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0014 = "0014_affinity_observability"

const (
	affinityObservabilityTable0014 = "request_logs"

	affinityContinuityColumn0014 = "continuity_hit"
	affinitySourceColumn0014     = "affinity_source"
	affinityStateColumn0014      = "affinity_state"

	// affinitySourceZero0014 和 affinityStateZero0014 是在此迁移之前创建的行以及
	// 未评估软亲和的请求所持久化的 bounded 零值。它们永远不会携带 prompt 缓存键、
	// 派生 key 或 HMAC 输入。
	affinitySourceZero0014 = "none"
	affinityStateZero0014  = "no_signal"
)

type affinityObservabilityColumn0014 struct {
	name          string
	definition    string
	typeFragment  string
	defaultValues []string
}

func affinityObservabilityColumns0014() []affinityObservabilityColumn0014 {
	return []affinityObservabilityColumn0014{
		{
			name:          affinityContinuityColumn0014,
			definition:    "BOOLEAN NOT NULL DEFAULT FALSE",
			typeFragment:  "bool",
			defaultValues: []string{"false", "0"},
		},
		{
			name:          affinitySourceColumn0014,
			definition:    "VARCHAR(32) NOT NULL DEFAULT '" + affinitySourceZero0014 + "'",
			typeFragment:  "char",
			defaultValues: []string{affinitySourceZero0014},
		},
		{
			name:          affinityStateColumn0014,
			definition:    "VARCHAR(32) NOT NULL DEFAULT '" + affinityStateZero0014 + "'",
			typeFragment:  "char",
			defaultValues: []string{affinityStateZero0014},
		},
	}
}

// Up0014 向请求日志表添加携带 bounded 软亲和观测的列。每列都是增量的且幂等的，
// 因此可以安全地恢复中断的运行，并且每列都有 bounded 零值默认值，使得历史行
// 永远不会暴露未设置或原始的亲和信号。
func Up0014(db *gorm.DB) error {
	if err := ValidateRecoverable0014(db); err != nil {
		return err
	}
	for _, column := range affinityObservabilityColumns0014() {
		if db.Migrator().HasColumn(affinityObservabilityTable0014, column.name) {
			continue
		}
		statement := fmt.Sprintf(
			"ALTER TABLE %s ADD COLUMN %s %s",
			affinityObservabilityTable0014,
			column.name,
			column.definition,
		)
		if err := db.Exec(statement).Error; err != nil {
			return fmt.Errorf("add request_logs.%s: %w", column.name, err)
		}
	}
	return Validate0014(db)
}

// ValidateRecoverable0014 确认迁移可以应用或恢复。
func ValidateRecoverable0014(db *gorm.DB) error {
	if !db.Migrator().HasTable(affinityObservabilityTable0014) {
		return fmt.Errorf("affinity observability table %q is missing", affinityObservabilityTable0014)
	}
	for _, column := range affinityObservabilityColumns0014() {
		if !db.Migrator().HasColumn(affinityObservabilityTable0014, column.name) {
			continue
		}
		if err := validateAffinityObservabilityColumn0014(db, column); err != nil {
			return err
		}
	}
	return nil
}

// Validate0014 确认每个亲和观测列都存在且带有 bounded 零值默认值。
func Validate0014(db *gorm.DB) error {
	for _, column := range affinityObservabilityColumns0014() {
		if !db.Migrator().HasColumn(affinityObservabilityTable0014, column.name) {
			return fmt.Errorf("request_logs.%s is missing", column.name)
		}
		if err := validateAffinityObservabilityColumn0014(db, column); err != nil {
			return err
		}
	}
	return ValidateRecoverable0014(db)
}

func validateAffinityObservabilityColumn0014(db *gorm.DB, column affinityObservabilityColumn0014) error {
	columns, err := db.Migrator().ColumnTypes(affinityObservabilityTable0014)
	if err != nil {
		return fmt.Errorf("inspect request_logs.%s: %w", column.name, err)
	}
	for _, candidate := range columns {
		if !strings.EqualFold(candidate.Name(), column.name) {
			continue
		}
		if nullable, known := candidate.Nullable(); !known || nullable {
			return fmt.Errorf("request_logs.%s is nullable", column.name)
		}
		dbType := strings.ToLower(candidate.DatabaseTypeName())
		switch column.typeFragment {
		case "bool":
			if !strings.Contains(dbType, "bool") &&
				dbType != "numeric" && dbType != "integer" && dbType != "tinyint" {
				return fmt.Errorf("request_logs.%s type %q is not boolean", column.name, dbType)
			}
		default:
			if !strings.Contains(dbType, "char") && !strings.Contains(dbType, "text") {
				return fmt.Errorf("request_logs.%s type %q is not textual", column.name, dbType)
			}
		}
		value, known := affinityColumnDefault0014(db, candidate, column.name)
		if !known {
			return fmt.Errorf("request_logs.%s default is unknown", column.name)
		}
		if !affinityDefaultAllowed0014(value, column.defaultValues) {
			return fmt.Errorf("request_logs.%s default %q is not a bounded zero value", column.name, value)
		}
		return nil
	}
	return fmt.Errorf("request_logs.%s is missing", column.name)
}

func affinityColumnDefault0014(db *gorm.DB, columnType gorm.ColumnType, name string) (string, bool) {
	if strings.EqualFold(db.Dialector.Name(), "sqlite") {
		// SQLite 通过 pragma 元数据报告默认值，而非 ColumnType。
		var defaultValue string
		if err := db.Raw(
			"SELECT dflt_value FROM pragma_table_info(?) WHERE name = ?",
			affinityObservabilityTable0014,
			name,
		).Scan(&defaultValue).Error; err != nil {
			return "", false
		}
		return affinityNormalizeDefault0014(defaultValue), defaultValue != ""
	}
	defaultValue, known := columnType.DefaultValue()
	return affinityNormalizeDefault0014(defaultValue), known
}

func affinityNormalizeDefault0014(value string) string {
	value = strings.Split(value, "::")[0]
	value = strings.Trim(value, "()' \"`")
	return strings.ToLower(strings.TrimSpace(value))
}

func affinityDefaultAllowed0014(value string, allowed []string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}
