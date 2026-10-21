package offlineconvert_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"gpt-load/internal/storage/offlineconvert"
)

// addRetiredPriceMultiplierColumns reproduces the schema of a legacy database
// that applied the retired `0009_price_multipliers` migration: both retired
// columns exist and every stored row carries the historical 1.0 value, because
// no operator ever changed the multiplier.
func addRetiredPriceMultiplierColumns(t *testing.T, path string) {
	t.Helper()
	for _, statement := range []string{
		"ALTER TABLE groups ADD COLUMN price_multiplier_micros INTEGER NOT NULL DEFAULT 1000000",
		"ALTER TABLE access_keys ADD COLUMN price_multiplier_micros INTEGER NOT NULL DEFAULT 1000000",
		"UPDATE groups SET price_multiplier_micros = 1000000",
		"UPDATE access_keys SET price_multiplier_micros = 1000000",
	} {
		writeSource(t, path, statement)
	}
}

// A legacy database that applied the retired price-multiplier migration keeps
// both `price_multiplier_micros` columns forever. They are retired columns, not
// unknown ones: the conversion must accept the source, report the columns it
// skipped, and publish a target without them while preserving the retained rows
// and their cumulative cost-limit checkpoints.
func TestConvertAcceptsRetiredPriceMultiplierColumns(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	cipher := newTestCipher(t, strings.Repeat("61", 32))
	want := buildLegacyFixture(t, sourcePath, cipher)
	addRetiredPriceMultiplierColumns(t, sourcePath)

	ledger := append(snapshotSource(t, sourcePath).ledger, retiredPriceMultiplierMigrationID)
	slices.Sort(ledger)
	setLedger(t, sourcePath, ledger)
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err != nil {
		t.Fatalf("source with retired price multiplier columns rejected: %v", err)
	}
	for _, column := range []string{"groups.price_multiplier_micros", "access_keys.price_multiplier_micros"} {
		if !slices.Contains(result.RetiredColumns, column) {
			t.Errorf("retired columns = %v, want %s", result.RetiredColumns, column)
		}
	}
	if result.Tables["groups"] != int64(len(want.groups)) {
		t.Errorf("groups copied = %d, want %d", result.Tables["groups"], len(want.groups))
	}
	assertSourceUnchanged(t, sourcePath, before)
	assertSingleFileTarget(t, targetDSN)

	target := openReadOnly(t, targetDSN)
	for _, table := range []string{"groups", "access_keys"} {
		if target.Migrator().HasColumn(table, "price_multiplier_micros") {
			t.Errorf("target still declares %s.price_multiplier_micros", table)
		}
	}
	// Cumulative usage and configured limits survive the drop: the conversion
	// re-prices nothing and resets no counter.
	assertTargetContent(t, target, cipher, want)
	assertTargetLoadable(t, targetDSN, cipher, want)
}

// The retired columns are named per table, not as a bare column-name allowlist.
// A column named `price_multiplier_micros` on a table that never declared it, a
// retired credential column on another table, and an ordinary unknown column are
// all refused, while the legitimate retired columns stay accepted.
func TestConvertRejectsRetiredColumnOwnerMismatch(t *testing.T) {
	for _, test := range []struct {
		name      string
		legacy    bool
		statement string
		want      string
	}{
		{
			name:      "credentials declares the group retired column",
			statement: "ALTER TABLE credentials ADD COLUMN price_multiplier_micros INTEGER NOT NULL DEFAULT 1000000",
			want:      "credentials.price_multiplier_micros",
		},
		{
			name:      "retired columns do not license the next column",
			legacy:    true,
			statement: "ALTER TABLE access_keys ADD COLUMN legacy_factor TEXT NOT NULL DEFAULT ''",
			want:      "access_keys.legacy_factor",
		},
		{
			name:      "groups declares a retired credential column",
			statement: "ALTER TABLE groups ADD COLUMN auth_state varchar(32) NOT NULL DEFAULT 'ready'",
			want:      "groups.auth_state",
		},
		{
			name:      "access keys declare a retired credential column",
			statement: "ALTER TABLE access_keys ADD COLUMN auth_error_code varchar(64) NOT NULL DEFAULT ''",
			want:      "access_keys.auth_error_code",
		},
		{
			name:      "unclassified column next to a retired column",
			legacy:    true,
			statement: "ALTER TABLE access_keys ADD COLUMN legacy_factor TEXT NOT NULL DEFAULT ''",
			want:      "access_keys.legacy_factor",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("62", 32)))
			if test.legacy {
				// The legitimate retired columns are present, so a refusal here
				// proves the exclusion is one table/column pair at a time rather
				// than a table-wide or column-name-wide allowance.
				addRetiredPriceMultiplierColumns(t, sourcePath)
			}
			writeSource(t, sourcePath, test.statement)
			before := snapshotSource(t, sourcePath)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("source column %s accepted: %v", test.want, err)
			}
			if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
				t.Fatalf("rejected conversion published a target: %v", statErr)
			}
			assertSourceUnchanged(t, sourcePath, before)
		})
	}
}

// The reported retired columns are the source's declared retired columns, not a
// copy of the declaration: a source without them reports none, and the
// subscription auth columns keep their own attribution next to the pricing ones.
func TestConvertReportsDeclaredRetiredColumnsOnly(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	cipher := newTestCipher(t, strings.Repeat("64", 32))
	buildLegacyFixture(t, sourcePath, cipher)
	addRetiredPriceMultiplierColumns(t, sourcePath)
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err != nil {
		t.Fatalf("convert source with retired columns: %v", err)
	}
	want := []string{
		"access_keys.price_multiplier_micros",
		"credentials.auth_error_code",
		"credentials.auth_state",
		"groups.price_multiplier_micros",
	}
	if !slices.Equal(result.RetiredColumns, want) {
		t.Fatalf("retired columns = %v, want %v", result.RetiredColumns, want)
	}
	assertSourceUnchanged(t, sourcePath, before)
}

// A source that carries only one of the two retired columns is accepted: the
// retirement is per column, not an all-or-nothing pair.
func TestConvertAcceptsASingleRetiredPriceMultiplierColumn(t *testing.T) {
	for _, table := range []string{"groups", "access_keys"} {
		t.Run(table, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			cipher := newTestCipher(t, strings.Repeat("63", 32))
			want := buildLegacyFixture(t, sourcePath, cipher)
			writeSource(t, sourcePath,
				"ALTER TABLE "+table+" ADD COLUMN price_multiplier_micros INTEGER NOT NULL DEFAULT 1000000")
			// The legacy fixture also declares the retired subscription credential
			// columns; drop them so this source carries exactly one retired column.
			writeSource(t, sourcePath, "ALTER TABLE credentials DROP COLUMN auth_state")
			writeSource(t, sourcePath, "ALTER TABLE credentials DROP COLUMN auth_error_code")
			before := snapshotSource(t, sourcePath)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			if err != nil {
				t.Fatalf("source with only %s.price_multiplier_micros rejected: %v", table, err)
			}
			reported := []string{table + ".price_multiplier_micros"}
			if !slices.Equal(result.RetiredColumns, reported) {
				t.Fatalf("retired columns = %v, want %v", result.RetiredColumns, reported)
			}
			assertSourceUnchanged(t, sourcePath, before)
			assertTargetLoadable(t, targetDSN, cipher, want)
		})
	}
}
