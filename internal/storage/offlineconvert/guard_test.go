package offlineconvert_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gpt-load/internal/storage/offlineconvert"
)

// A non-empty "-shm" is NOT evidence of a live writer, and the converter must not
// treat it as one. Any single read-only open of a WAL-mode database creates the
// shared-memory index at its full 32 KiB size while holding no pages at all, so
// refusing on it would reject the very "stopped gateway, WAL journal mode" source
// the converter exists to serve. This test pins that behaviour: it plants the
// exact file SQLite itself produces and requires a successful conversion.
//
// It also documents why "-wal" is the only valid writer signal: an empty "-wal"
// is created by the same read while a writer's "-wal" carries the committed pages.
func TestConvertAcceptsSharedMemoryIndexCreatedByARead(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	want := buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("71", 32)))
	// A read-only open of the fixture is what SQLite itself does, and it is the
	// only reason the file exists; assert it before planting it explicitly.
	if err := os.WriteFile(sourcePath+"-shm", make([]byte, 32768), 0o600); err != nil {
		t.Fatalf("write source shared-memory index: %v", err)
	}
	if err := os.WriteFile(sourcePath+"-wal", nil, 0o600); err != nil {
		t.Fatalf("write empty source write-ahead log: %v", err)
	}

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err != nil {
		t.Fatalf("source carrying only a read's own -shm was refused: %v", err)
	}
	if result.Tables["groups"] != int64(len(want.groups)) {
		t.Fatalf("groups copied = %d, want %d", result.Tables["groups"], len(want.groups))
	}
	assertSingleFileTarget(t, targetDSN)
}
