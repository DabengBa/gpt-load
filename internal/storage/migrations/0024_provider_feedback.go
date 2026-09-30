package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0024 = "0024_provider_feedback"

type feedbackColumn0024 struct {
	table, column, definition, constraint string
}

var feedbackColumns0024 = []feedbackColumn0024{
	{
		table: "request_log_attempts", column: "provider_first_response_ms",
		definition: "BIGINT CONSTRAINT chk_request_log_attempt_provider_first_response CHECK (provider_first_response_ms IS NULL OR provider_first_response_ms >= 0)",
		constraint: "chk_request_log_attempt_provider_first_response",
	},
	{
		table: "request_log_attempts", column: "provider_tokens_per_second",
		definition: "DOUBLE PRECISION CONSTRAINT chk_request_log_attempt_provider_tokens_per_second CHECK (provider_tokens_per_second IS NULL OR (provider_tokens_per_second >= 0 AND provider_tokens_per_second < 1e308))",
		constraint: "chk_request_log_attempt_provider_tokens_per_second",
	},
	{
		table: "request_log_attempts", column: "feedback_reason",
		definition: "VARCHAR(32) NOT NULL DEFAULT '' CONSTRAINT chk_request_log_attempt_feedback_reason CHECK (feedback_reason IN ('','upstream_failure','first_response_slow','output_rate_faulty','output_rate_slow'))",
		constraint: "chk_request_log_attempt_feedback_reason",
	},
	{
		table: "request_log_attempts", column: "feedback_status",
		definition: "VARCHAR(16) NOT NULL DEFAULT '' CONSTRAINT chk_request_log_attempt_feedback_status CHECK (feedback_status IN ('','normal','slow','faulty') AND ((feedback_status IN ('','normal') AND feedback_reason = '') OR (feedback_status = 'slow' AND feedback_reason = 'output_rate_slow') OR (feedback_status = 'faulty' AND feedback_reason IN ('upstream_failure','first_response_slow','output_rate_faulty'))))",
		constraint: "chk_request_log_attempt_feedback_status",
	},
	{
		table: "usage_attempt_aggregation_journals", column: "normal_attempt_count",
		definition: "BIGINT NOT NULL DEFAULT 0 CONSTRAINT chk_usage_attempt_journal_normal_count CHECK (normal_attempt_count >= 0 AND normal_attempt_count <= attempt_count)",
		constraint: "chk_usage_attempt_journal_normal_count",
	},
	{
		table: "usage_attempt_aggregation_journals", column: "slow_attempt_count",
		definition: "BIGINT NOT NULL DEFAULT 0 CONSTRAINT chk_usage_attempt_journal_slow_count CHECK (slow_attempt_count >= 0 AND slow_attempt_count <= attempt_count)",
		constraint: "chk_usage_attempt_journal_slow_count",
	},
	{
		table: "usage_attempt_aggregation_journals", column: "faulty_attempt_count",
		definition: "BIGINT NOT NULL DEFAULT 0 CONSTRAINT chk_usage_attempt_journal_feedback_count CHECK (faulty_attempt_count >= 0 AND normal_attempt_count + slow_attempt_count + faulty_attempt_count <= attempt_count)",
		constraint: "chk_usage_attempt_journal_feedback_count",
	},
	{
		table: "usage_attempt_stats", column: "normal_attempt_count",
		definition: "BIGINT NOT NULL DEFAULT 0 CONSTRAINT chk_usage_attempt_stat_normal_count CHECK (normal_attempt_count >= 0 AND normal_attempt_count <= attempt_count)",
		constraint: "chk_usage_attempt_stat_normal_count",
	},
	{
		table: "usage_attempt_stats", column: "slow_attempt_count",
		definition: "BIGINT NOT NULL DEFAULT 0 CONSTRAINT chk_usage_attempt_stat_slow_count CHECK (slow_attempt_count >= 0 AND slow_attempt_count <= attempt_count)",
		constraint: "chk_usage_attempt_stat_slow_count",
	},
	{
		table: "usage_attempt_stats", column: "faulty_attempt_count",
		definition: "BIGINT NOT NULL DEFAULT 0 CONSTRAINT chk_usage_attempt_stat_feedback_count CHECK (faulty_attempt_count >= 0 AND normal_attempt_count + slow_attempt_count + faulty_attempt_count <= attempt_count)",
		constraint: "chk_usage_attempt_stat_feedback_count",
	},
}

// Up0024 adds bounded provider feedback to attempt logs and attempt aggregates.
func Up0024(db *gorm.DB) error {
	if db == nil || db.Dialector == nil {
		return fmt.Errorf("provider feedback migration: database is nil")
	}
	if !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("provider feedback migration: unsupported database driver %q", db.Dialector.Name())
	}
	for _, table := range []string{
		"request_log_attempts", "usage_attempt_aggregation_journals", "usage_attempt_stats",
	} {
		if !db.Migrator().HasTable(table) {
			return fmt.Errorf("provider feedback migration: table %q is missing", table)
		}
	}
	for _, column := range feedbackColumns0024 {
		if !db.Migrator().HasColumn(column.table, column.column) {
			statement := fmt.Sprintf(
				"ALTER TABLE %s ADD COLUMN %s %s",
				quoteProviderFeedbackIdentifier0024(column.table),
				quoteProviderFeedbackIdentifier0024(column.column),
				column.definition,
			)
			if err := db.Exec(statement).Error; err != nil {
				return fmt.Errorf("add %s.%s: %w", column.table, column.column, err)
			}
		}
	}
	return Validate0024(db)
}

// Validate0024 verifies the new schema and the bounded values it stores.
func Validate0024(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("validate provider feedback migration: unsupported or nil database")
	}
	for _, column := range feedbackColumns0024 {
		if !db.Migrator().HasColumn(column.table, column.column) {
			return fmt.Errorf("validate provider feedback migration: %s.%s is missing", column.table, column.column)
		}
		if !db.Migrator().HasConstraint(column.table, column.constraint) {
			return fmt.Errorf("validate provider feedback migration: constraint %q is missing", column.constraint)
		}
	}
	var invalidAttempts int64
	if err := db.Table("request_log_attempts").Where(
		"provider_first_response_ms < 0 OR provider_tokens_per_second < 0 OR provider_tokens_per_second >= 1e308 OR " +
			"feedback_status NOT IN ('','normal','slow','faulty') OR " +
			"feedback_reason NOT IN ('','upstream_failure','first_response_slow','output_rate_faulty','output_rate_slow') OR " +
			"(feedback_status IN ('','normal') AND feedback_reason <> '') OR " +
			"(feedback_status = 'slow' AND feedback_reason <> 'output_rate_slow') OR " +
			"(feedback_status = 'faulty' AND feedback_reason NOT IN ('upstream_failure','first_response_slow','output_rate_faulty'))",
	).Count(&invalidAttempts).Error; err != nil {
		return fmt.Errorf("validate provider feedback attempt values: %w", err)
	}
	if invalidAttempts != 0 {
		return fmt.Errorf("validate provider feedback attempt values: invalid rows found")
	}
	for _, table := range []string{"usage_attempt_aggregation_journals", "usage_attempt_stats"} {
		var invalidCounts int64
		if err := db.Table(table).Where(
			"normal_attempt_count < 0 OR slow_attempt_count < 0 OR faulty_attempt_count < 0 OR " +
				"normal_attempt_count + slow_attempt_count + faulty_attempt_count > attempt_count",
		).Count(&invalidCounts).Error; err != nil {
			return fmt.Errorf("validate %s feedback counts: %w", table, err)
		}
		if invalidCounts != 0 {
			return fmt.Errorf("validate %s feedback counts: invalid rows found", table)
		}
	}
	return nil
}

func quoteProviderFeedbackIdentifier0024(value string) string {
	return `"` + value + `"`
}
