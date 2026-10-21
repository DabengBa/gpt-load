package offlineconvert_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gpt-load/internal/storage/offlineconvert"
)

// dropColumn rebuilds a table without one of its columns, which is the only way
// SQLite lets an indexed or constrained column disappear. The rebuilt table
// keeps every other column so the source still looks like a legacy database
// that simply lacks one stored value.
func dropColumn(t *testing.T, path, table, column string) {
	t.Helper()
	source := openReadOnly(t, path)
	var declared []string
	if err := source.Raw("SELECT name FROM pragma_table_info(?)", table).Scan(&declared).Error; err != nil {
		t.Fatalf("read %s columns: %v", table, err)
	}
	kept := make([]string, 0, len(declared))
	for _, name := range declared {
		if name != column {
			kept = append(kept, name)
		}
	}
	if len(kept) == len(declared) {
		t.Fatalf("%s has no column %s", table, column)
	}
	var ddl string
	if err := source.Raw("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", table).Scan(&ddl).Error; err != nil {
		t.Fatalf("read %s ddl: %v", table, err)
	}
	rebuild := rebuildDDLWithout(ddl, column)
	writeSource(t, path, "PRAGMA foreign_keys=OFF")
	writeSource(t, path, "ALTER TABLE "+table+" RENAME TO "+table+"_rebuild_source")
	writeSource(t, path, rebuild)
	writeSource(t, path, "INSERT INTO "+table+" ("+quoteList(kept)+") SELECT "+quoteList(kept)+" FROM "+table+"_rebuild_source")
	writeSource(t, path, "DROP TABLE "+table+"_rebuild_source")
	writeSource(t, path, "PRAGMA foreign_keys=ON")
}

func quoteList(names []string) string {
	quoted := make([]string, 0, len(names))
	for _, name := range names {
		quoted = append(quoted, `"`+name+`"`)
	}
	return strings.Join(quoted, ", ")
}

// rebuildDDLWithout strips one column definition (and every index/constraint
// clause that mentions it) from a CREATE TABLE statement.
func rebuildDDLWithout(ddl, column string) string {
	open := strings.Index(ddl, "(")
	closing := strings.LastIndex(ddl, ")")
	if open < 0 || closing < open {
		return ddl
	}
	body := ddl[open+1 : closing]
	definitions := splitTopLevel(body)
	kept := make([]string, 0, len(definitions))
	for _, definition := range definitions {
		if mentionsColumn(definition, column) {
			continue
		}
		kept = append(kept, definition)
	}
	return ddl[:open+1] + strings.Join(kept, ",") + ddl[closing:]
}

// mentionsColumn reports whether a DDL clause defines, indexes or constrains the
// given column.
func mentionsColumn(definition, column string) bool {
	fields := strings.FieldsFunc(definition, func(r rune) bool {
		return r == ' ' || r == '(' || r == ')' || r == ',' || r == '`' || r == '"' || r == '[' || r == ']'
	})
	for _, field := range fields {
		if strings.EqualFold(field, column) {
			return true
		}
	}
	return false
}

// splitTopLevel splits a CREATE TABLE body on commas that are outside any
// parentheses, so composite index clauses stay intact.
func splitTopLevel(body string) []string {
	var parts []string
	depth := 0
	start := 0
	for index, char := range body {
		switch char {
		case '(':
			depth++
		case ')':
			depth--
		case ',':
			if depth == 0 {
				parts = append(parts, body[start:index])
				start = index + 1
			}
		}
	}
	parts = append(parts, body[start:])
	kept := make([]string, 0, len(parts))
	for _, part := range parts {
		if strings.TrimSpace(part) != "" {
			kept = append(kept, part)
		}
	}
	return kept
}

// The copy must never fall back to "whatever columns happen to share a name".
// Every retained table has an explicit expected column set for the supported
// source versions, so a source that drops a column the operator's data lives in
// is refused, including columns that are nullable and would silently become
// NULL in the target.
func TestConvertRejectsMissingNullableSourceColumn(t *testing.T) {
	for _, test := range []struct {
		table  string
		column string
	}{
		{"groups", "validation_model"},
		{"groups", "params"},
		{"credentials", "fingerprint"},
		{"credentials", "identity_fingerprint"},
		{"credentials", "secret_version"},
		{"credentials", "proxy_config"},
		{"access_keys", "filters"},
		{"control_operations", "idempotency_key"},
		{"control_operations", "proposal_id"},
		{"agent_change_proposals", "approved_by"},
		{"agent_credentials", "expires_at_ms"},
		{"jobs", "error"},
		{"access_key_cost_limit_states", "snapshot_version"},
		{"model_prices", "context_price_tiers"},
		{"system_settings", "updated_at_ms"},
	} {
		t.Run(test.table+"."+test.column, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("b1", 32)))
			dropColumn(t, sourcePath, test.table, test.column)
			before := snapshotSource(t, sourcePath)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			want := test.table + "." + test.column
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("source missing %s accepted: %v", want, err)
			}
			if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
				t.Fatalf("rejected conversion published a target: %v", statErr)
			}
			assertSourceUnchanged(t, sourcePath, before)
		})
	}
}

// A retained table that is missing entirely must be refused: the target would
// otherwise be published with an empty table and the operator's rows silently
// gone.
func TestConvertRejectsMissingRetainedSourceTable(t *testing.T) {
	for _, table := range []string{
		"access_key_cost_limit_rules",
		"access_key_cost_limit_states",
		"access_keys",
		"agent_change_proposals",
		"agent_credentials",
		"control_operations",
		"credentials",
		"groups",
		"jobs",
		"model_prices",
		"system_settings",
	} {
		t.Run(table, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("b2", 32)))
			writeSource(t, sourcePath, "PRAGMA foreign_keys=OFF")
			writeSource(t, sourcePath, "DROP TABLE "+table)
			before := snapshotSource(t, sourcePath)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			if err == nil || !strings.Contains(err.Error(), table) {
				t.Fatalf("source missing table %s accepted: %v", table, err)
			}
			if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
				t.Fatalf("rejected conversion published a target: %v", statErr)
			}
			assertSourceUnchanged(t, sourcePath, before)
		})
	}
}

// A legacy source in WAL mode is snapshotted from one consistent read
// transaction and published as a single delete-journal file.
func TestConvertSnapshotIsIndependentOfLaterSourceWrites(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("b3", 32)))
	writeSource(t, sourcePath, "PRAGMA journal_mode=WAL")
	writeSource(t, sourcePath, "PRAGMA wal_checkpoint(TRUNCATE)")

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	if _, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	}); err != nil {
		t.Fatalf("convert stopped WAL source: %v", err)
	}
	target := openReadOnly(t, targetDSN)
	var journalMode string
	if err := target.Raw("PRAGMA journal_mode").Scan(&journalMode).Error; err != nil {
		t.Fatalf("read converted journal mode: %v", err)
	}
	if !strings.EqualFold(journalMode, "delete") {
		t.Fatalf("converted journal mode = %q, want delete", journalMode)
	}
	assertSingleFileTarget(t, targetDSN)
}
