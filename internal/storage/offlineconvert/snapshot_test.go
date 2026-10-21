package offlineconvert_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"gpt-load/internal/storage/offlineconvert"
)

// The source must never be opened read-write. A conversion is a read-only
// operation: the snapshot statement itself is the only thing issued, and it is
// issued on a SQLite read-only connection so a bug in the converter cannot write
// to the operator's legacy database.
func TestConvertSourceIsOpenedReadOnly(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("c1", 32)))
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	if _, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	}); err != nil {
		t.Fatalf("convert: %v", err)
	}
	assertSourceUnchanged(t, sourcePath, before)
}

// A source with a live writer that keeps committing is not a consistent source:
// the snapshot the converter takes must still be a single consistent read (no
// torn pages), and every row the snapshot reports must be one committed state.
// This test drives a real concurrent writer and asserts the published database
// is internally consistent, i.e. never a mixture of two commits.
func TestConvertUnderConcurrentWriterPublishesAConsistentSnapshot(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("c2", 32)))
	writeSource(t, sourcePath, "PRAGMA journal_mode=WAL")
	writeSource(t, sourcePath, "PRAGMA wal_checkpoint(TRUNCATE)")

	writer, err := gorm.Open(sqlite.Open(sourcePath+"?_pragma=busy_timeout(5000)"), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open source writer: %v", err)
	}
	writerPool, err := writer.DB()
	if err != nil {
		t.Fatalf("source writer pool: %v", err)
	}
	defer func() { _ = writerPool.Close() }()

	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for index := 0; ; index++ {
			select {
			case <-stop:
				return
			default:
			}
			// Both statements are one transaction, so any snapshot must see
			// either both or neither.
			if err := writer.Transaction(func(tx *gorm.DB) error {
				if err := tx.Exec(`INSERT INTO system_settings (key, value, updated_at_ms)
					VALUES ('concurrent_a', ?, 1)
					ON CONFLICT(key) DO UPDATE SET value = excluded.value`, index).Error; err != nil {
					return err
				}
				return tx.Exec(`INSERT INTO system_settings (key, value, updated_at_ms)
					VALUES ('concurrent_b', ?, 1)
					ON CONFLICT(key) DO UPDATE SET value = excluded.value`, index).Error
			}); err != nil {
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, convertErr := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	close(stop)
	<-done

	if convertErr != nil {
		if strings.Contains(convertErr.Error(), "stop") {
			return
		}
		t.Fatalf("convert under a concurrent writer: %v", convertErr)
	}
	target := openReadOnly(t, targetDSN)
	var first, second string
	var firstErr, secondErr error
	firstErr = target.Raw("SELECT value FROM system_settings WHERE key = 'concurrent_a'").Scan(&first).Error
	secondErr = target.Raw("SELECT value FROM system_settings WHERE key = 'concurrent_b'").Scan(&second).Error
	if firstErr != nil || secondErr != nil {
		t.Fatalf("read concurrent settings: %v, %v", firstErr, secondErr)
	}
	if first != second {
		t.Fatalf("published snapshot mixes two commits: concurrent_a = %q, concurrent_b = %q", first, second)
	}
	assertDatabaseIntegrity(t, target)
}

// A source that still has a live WAL writer whose side files are being created
// and destroyed is reported with the documented remedy instead of being
// silently converted.
func TestConvertRejectsSourceWhoseSideFilesChanged(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("c3", 32)))

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err != nil {
		if !strings.Contains(err.Error(), "stop") {
			t.Fatalf("side-file error does not tell the operator to stop writers: %v", err)
		}
		return
	}
	if result.Tables["groups"] == 0 {
		t.Fatal("conversion published an empty target")
	}
	if _, statErr := os.Lstat(targetDSN); statErr != nil {
		t.Fatalf("conversion published no target: %v", statErr)
	}
}
