package offlineconvert

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// The snapshot connection must be read-only at the SQLite level, so a bug in the
// converter cannot write to the operator's legacy database.
func TestOpenSourceForSnapshotIsReadOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "source.db")
	prepareSource(t, path)

	db, err := openSourceForSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("open source for snapshot: %v", err)
	}
	defer closeDatabase(db)

	if err := db.Exec("CREATE TABLE write_probe (id integer)").Error; err == nil {
		t.Fatal("snapshot connection accepted a write")
	}
	if _, err := os.Lstat(path + "-journal"); err == nil {
		t.Fatal("snapshot connection created a rollback journal on the source")
	}
}

// The snapshot must carry committed write-ahead-log content and be a single file
// the copy can read without a sidecar journal.
func TestSnapshotSourceIncludesCommittedWALPages(t *testing.T) {
	path := filepath.Join(t.TempDir(), "source.db")
	prepareSource(t, path)

	snapshotPath, cleanup, err := snapshotSource(context.Background(), path)
	if err != nil {
		t.Fatalf("snapshot source: %v", err)
	}
	defer cleanup()

	snapshot, err := openSnapshot(context.Background(), snapshotPath)
	if err != nil {
		t.Fatalf("open snapshot: %v", err)
	}
	defer closeDatabase(snapshot)
	var value string
	if err := snapshot.Raw("SELECT value FROM system_settings WHERE key = 'probe'").Scan(&value).Error; err != nil {
		t.Fatalf("read snapshot: %v", err)
	}
	if value != "committed" {
		t.Fatalf("snapshot value = %q, want the committed value", value)
	}
	var journalMode string
	if err := snapshot.Raw("PRAGMA journal_mode").Scan(&journalMode).Error; err != nil {
		t.Fatalf("read snapshot journal mode: %v", err)
	}
	if !strings.EqualFold(journalMode, "delete") {
		t.Fatalf("snapshot journal mode = %q, want a single-file delete journal", journalMode)
	}
}

// prepareSource writes a WAL-mode database with one committed row.
func prepareSource(t *testing.T, path string) {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open writable source: %v", err)
	}
	defer closeDatabase(db)
	for _, statement := range []string{
		"PRAGMA journal_mode=WAL",
		"CREATE TABLE system_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
		"INSERT INTO system_settings (key, value) VALUES ('probe', 'committed')",
	} {
		if err := db.Exec(statement).Error; err != nil {
			t.Fatalf("prepare source: %v", err)
		}
	}
}
