package offlineconvert_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"gpt-load/internal/storage/offlineconvert"
)

// A source that another process still holds open in WAL mode is exactly the
// "gateway was not stopped" case. It must be refused, because the converter
// cannot prove the snapshot is a state the writer ever committed alone, and the
// operator has to be told to stop the gateway.
func TestConvertRefusesSourceHeldOpenByAWriter(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("d1", 32)))
	writeSource(t, sourcePath, "PRAGMA journal_mode=WAL")
	writeSource(t, sourcePath, "PRAGMA wal_checkpoint(TRUNCATE)")

	// The live gateway: a persisted WAL database with an open connection.
	live, err := gorm.Open(sqlite.Open(sourcePath), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open live source: %v", err)
	}
	livePool, err := live.DB()
	if err != nil {
		t.Fatalf("live source pool: %v", err)
	}
	defer func() { _ = livePool.Close() }()
	if err := live.Exec(`INSERT INTO system_settings (key, value, updated_at_ms)
		VALUES ('gateway_heartbeat', '1', 1700000300000)`).Error; err != nil {
		t.Fatalf("write from the live gateway: %v", err)
	}

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err = offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "stop") {
		t.Fatalf("source held open by a writer was not refused with the stop-writers remedy: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("refused conversion published a target: %v", statErr)
	}
}

// The reader's own footprint — an empty -wal and SQLite's -shm — is not a
// writer, so a stopped WAL-mode source must still convert.
func TestConvertAcceptsStoppedSourceThatGainsEmptySideFiles(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("d2", 32)))
	writeSource(t, sourcePath, "PRAGMA journal_mode=WAL")
	writeSource(t, sourcePath, "PRAGMA wal_checkpoint(TRUNCATE)")
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	if _, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	}); err != nil {
		t.Fatalf("stopped WAL source refused: %v", err)
	}
	assertSourceUnchanged(t, sourcePath, before)
	if _, statErr := os.Lstat(targetDSN); statErr != nil {
		t.Fatalf("conversion published no target: %v", statErr)
	}
}
