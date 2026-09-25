package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0023 = "0023_billing_failure_category"

const failureCategoryExpression0023 = "failure_category IN ('ok','rate_limited','model_unavailable','invalid_key','billing','upstream_host_error','client_error','conversion_unsupported','downstream_cancel','authentication_required','ambiguous')"

// Up0023 extends the request-log attempt failure category constraint with the
// billing class so insufficient-balance attempts persist instead of being
// rejected by the CHECK.
func Up0023(db *gorm.DB) error {
	if !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("add billing failure category: unsupported database driver %q", db.Dialector.Name())
	}
	if !db.Migrator().HasTable(requestLogAttemptTable0006) {
		return fmt.Errorf("add billing failure category: table %q is missing", requestLogAttemptTable0006)
	}
	if Validate0023(db) == nil {
		return nil
	}
	return rebuildSQLiteRequestLogAttempts0023(db)
}

func rebuildSQLiteRequestLogAttempts0023(db *gorm.DB) error {
	statements := []string{
		`CREATE TABLE request_log_attempts__0023 (
			request_id varchar(36) NOT NULL,
			sequence integer NOT NULL,
			completed_at_ms integer NOT NULL,
			group_id integer NOT NULL,
			group_name varchar(255) NOT NULL,
			channel_id varchar(64) NOT NULL DEFAULT '',
			credential_id integer NOT NULL,
			operation varchar(64) NOT NULL DEFAULT '',
			route_mode varchar(32) NOT NULL DEFAULT '',
			upstream_model varchar(255) NOT NULL DEFAULT '',
			upstream_request_id varchar(255) NOT NULL DEFAULT '',
			dispatch_state varchar(32) NOT NULL DEFAULT '',
			response_started numeric NOT NULL DEFAULT false,
			upstream_protocol varchar(32) NOT NULL DEFAULT '',
			reasoning_mode varchar(64) NOT NULL DEFAULT '',
			reasoning_effort varchar(64) NOT NULL DEFAULT '',
			reasoning_budget_tokens integer,
			status_code integer NOT NULL,
			duration_ms integer NOT NULL,
			failure_category varchar(32) NOT NULL,
			failure_origin varchar(16) NOT NULL DEFAULT '',
			failure_scope varchar(16) NOT NULL DEFAULT '',
			retry_directive varchar(32) NOT NULL DEFAULT '',
			effect varchar(32) NOT NULL DEFAULT '',
			rule_id varchar(128) NOT NULL DEFAULT '',
			action varchar(32) NOT NULL,
			will_retry numeric NOT NULL DEFAULT false,
			error_code varchar(64) NOT NULL DEFAULT '',
			error_summary text NOT NULL,
			committed numeric NOT NULL DEFAULT false,
			pricing_receipt json,
			PRIMARY KEY (request_id, sequence),
			CONSTRAINT fk_request_log_attempts_request_log FOREIGN KEY (request_id)
				REFERENCES request_logs(id) ON DELETE CASCADE ON UPDATE CASCADE,
			CONSTRAINT chk_request_log_attempt_sequence CHECK (sequence > 0),
			CONSTRAINT chk_request_log_attempt_completed_at CHECK (completed_at_ms >= 0),
			CONSTRAINT chk_request_log_attempt_group CHECK (group_id > 0),
			CONSTRAINT chk_request_log_attempt_credential CHECK (credential_id > 0),
			CONSTRAINT chk_request_log_attempt_duration CHECK (duration_ms >= 0),
			CONSTRAINT chk_request_log_attempt_failure_category CHECK (` + failureCategoryExpression0023 + `),
			CONSTRAINT chk_request_log_attempt_failure_origin CHECK (failure_origin IN ('','client','upstream','downstream','internal')),
			CONSTRAINT chk_request_log_attempt_failure_scope CHECK (failure_scope IN ('','request','model','credential','group')),
			CONSTRAINT chk_request_log_attempt_retry_directive CHECK (retry_directive IN ('','none','refresh_credential','next_candidate')),
			CONSTRAINT chk_request_log_attempt_effect CHECK (effect IN ('','none','cooldown_credential','record_credential_failure','skip_group')),
			CONSTRAINT chk_request_log_attempt_action CHECK (action IN ('terminate','retry','cooldown_credential','fail_credential','skip_group'))
		)`,
		`INSERT INTO request_log_attempts__0023 (
			request_id, sequence, completed_at_ms, group_id, group_name, channel_id,
			credential_id, operation, route_mode, upstream_model, upstream_request_id,
			dispatch_state, response_started, upstream_protocol, reasoning_mode,
			reasoning_effort, reasoning_budget_tokens, status_code, duration_ms,
			failure_category, failure_origin, failure_scope, retry_directive, effect,
			rule_id, action, will_retry, error_code, error_summary, committed, pricing_receipt
		) SELECT
			request_id, sequence, completed_at_ms, group_id, group_name, channel_id,
			credential_id, operation, route_mode, upstream_model, upstream_request_id,
			dispatch_state, response_started, upstream_protocol, reasoning_mode,
			reasoning_effort, reasoning_budget_tokens, status_code, duration_ms,
			failure_category, failure_origin, failure_scope, retry_directive, effect,
			rule_id, action, will_retry, error_code, error_summary, committed, pricing_receipt
		FROM request_log_attempts`,
		`DROP TABLE request_log_attempts`,
		`ALTER TABLE request_log_attempts__0023 RENAME TO request_log_attempts`,
		`CREATE INDEX idx_request_log_attempts_group_completed_request ON request_log_attempts(group_id, completed_at_ms DESC, request_id)`,
		`CREATE INDEX idx_request_log_attempts_channel_completed_request ON request_log_attempts(channel_id, completed_at_ms DESC, request_id)`,
		`CREATE INDEX idx_request_log_attempts_credential_completed_request ON request_log_attempts(credential_id, completed_at_ms DESC, request_id)`,
		`CREATE INDEX idx_request_log_attempts_model_completed_request ON request_log_attempts(upstream_model, completed_at_ms DESC, request_id)`,
		`CREATE INDEX idx_request_log_attempts_status_completed_request ON request_log_attempts(status_code, completed_at_ms DESC, request_id)`,
		`CREATE INDEX idx_request_log_attempts_failure_completed_request ON request_log_attempts(failure_category, completed_at_ms DESC, request_id)`,
		`CREATE INDEX idx_request_log_attempts_error_completed_request ON request_log_attempts(error_code, completed_at_ms DESC, request_id)`,
	}
	for _, statement := range statements {
		if err := db.Exec(statement).Error; err != nil {
			return fmt.Errorf("rebuild SQLite request log attempts: %w", err)
		}
	}
	return nil
}

// Validate0023 verifies the failure category constraint accepts billing while
// retaining the retained decision columns and indexes.
func Validate0023(db *gorm.DB) error {
	if !db.Migrator().HasTable(requestLogAttemptTable0006) {
		return fmt.Errorf("validate billing failure category: table %q is missing", requestLogAttemptTable0006)
	}
	for _, name := range append([]string{failureCategoryConstraint0006}, decisionConstraintNames0006()...) {
		if !db.Migrator().HasConstraint(requestLogAttemptTable0006, name) {
			return fmt.Errorf("validate billing failure category: constraint %q is missing", name)
		}
	}
	for _, index := range requestLogAttemptIndexes0006 {
		if !db.Migrator().HasIndex(requestLogAttemptTable0006, index) {
			return fmt.Errorf("validate billing failure category: index %q is missing", index)
		}
	}
	definition, err := failureCategoryConstraintDefinition0006(db)
	if err != nil {
		return err
	}
	if !strings.Contains(strings.ToLower(definition), "billing") {
		return fmt.Errorf("validate billing failure category: failure category constraint is stale")
	}
	return nil
}
