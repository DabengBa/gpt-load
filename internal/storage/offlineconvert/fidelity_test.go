package offlineconvert_test

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"gorm.io/gorm"

	"gpt-load/internal/storage/offlineconvert"
)

// Every retained row must arrive with byte-identical primary identities,
// ciphertext and cost-limit checkpoints. The converter copies raw SQLite values,
// so this compares the stored values directly instead of going through models.
func TestConvertPreservesIdentitiesCiphertextAndCheckpoints(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("e1", 32)))

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	if _, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	}); err != nil {
		t.Fatalf("convert: %v", err)
	}

	source := openReadOnly(t, sourcePath)
	target := openReadOnly(t, targetDSN)
	for _, table := range []struct {
		name    string
		order   string
		columns []string
	}{
		{"credentials", "id", []string{
			"id", "group_id", "data", "fingerprint", "identity_fingerprint",
			"secret_version", "status", "weight_manual", "proxy_config",
			"created_at_ms", "updated_at_ms",
		}},
		{"access_keys", "id", []string{
			"id", "name", "key_value", "key_hash", "key_suffix", "status", "filters",
			"rpm_limit", "daily_cost_limit_nano_usd", "monthly_cost_limit_nano_usd",
			"expires_at_ms", "created_at_ms", "updated_at_ms",
		}},
		{"access_key_cost_limit_rules", "id", []string{
			"id", "access_key_id", "kind", "limit_nano_usd", "period_seconds",
			"rule_revision", "created_at_ms", "updated_at_ms",
		}},
		{"access_key_cost_limit_states", "rule_id", []string{
			"rule_id", "rule_revision", "used_nano_usd", "window_started_at_ms",
			"window_ends_at_ms", "window_generation", "snapshot_version", "updated_at_ms",
		}},
		{"model_prices", "id", []string{
			"id", "channel_id", "model_id", "input_price_nano_usd_per_million_tokens",
			"output_price_nano_usd_per_million_tokens",
			"cache_read_price_nano_usd_per_million_tokens",
			"cache_write_price_nano_usd_per_million_tokens", "context_price_tiers",
			"mode_price_schedules", "is_manual", "created_at_ms", "updated_at_ms",
		}},
		{"system_settings", "key", []string{"key", "value", "updated_at_ms"}},
		{"control_operations", "commit_sequence", []string{
			"commit_sequence", "operation_id", "idempotency_key", "digest_version",
			"request_digest", "operation_kind", "resource_identity", "proposal_id",
			"canonical_result", "required_stages", "last_completed_stage", "failed_stage",
			"completed_at_ms", "compacted_at_ms", "created_at_ms", "updated_at_ms",
		}},
		{"jobs", "id", []string{
			"id", "type", "status", "payload", "result", "error", "created_at_ms",
			"started_at_ms", "finished_at_ms",
		}},
		{"agent_credentials", "id", []string{
			"id", "name", "secret_hash", "scopes", "status", "expires_at_ms",
			"disabled_at_ms", "created_at_ms", "updated_at_ms",
		}},
		{"agent_change_proposals", "id", []string{
			"id", "created_at_ms", "updated_at_ms", "base_snapshot_revision",
			"creator_credential_id", "state", "updates", "approved_runtime_epoch",
			"approved_by", "approved_at_ms",
		}},
	} {
		want := selectAll(t, source, table.name, table.order, table.columns)
		got := selectAll(t, target, table.name, table.order, table.columns)
		if len(want) != len(got) {
			t.Fatalf("%s rows = %d, want %d", table.name, len(got), len(want))
		}
		for index := range want {
			if !slices.Equal(want[index], got[index]) {
				t.Fatalf("%s row %d changed: %v -> %v", table.name, index, want[index], got[index])
			}
		}
	}
}

// selectAll reads the given columns of a table in a stable order and renders each
// value with its SQLite type, so a value that changed type is visible too.
func selectAll(t *testing.T, db *gorm.DB, table, order string, columns []string) [][]string {
	t.Helper()
	quoted := make([]string, 0, len(columns))
	for _, column := range columns {
		quoted = append(quoted, `"`+column+`"`)
	}
	query := fmt.Sprintf("SELECT %s FROM %q ORDER BY %q", strings.Join(quoted, ", "), table, order)
	rows, err := db.Raw(query).Rows()
	if err != nil {
		t.Fatalf("read %s: %v", table, err)
	}
	defer func() { _ = rows.Close() }()
	var collected [][]string
	for rows.Next() {
		values := make([]any, len(columns))
		pointers := make([]any, len(columns))
		for index := range values {
			pointers[index] = &values[index]
		}
		if err := rows.Scan(pointers...); err != nil {
			t.Fatalf("scan %s: %v", table, err)
		}
		rendered := make([]string, len(values))
		for index, value := range values {
			rendered[index] = fmt.Sprintf("%T:%v", value, value)
		}
		collected = append(collected, rendered)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read %s rows: %v", table, err)
	}
	return collected
}
