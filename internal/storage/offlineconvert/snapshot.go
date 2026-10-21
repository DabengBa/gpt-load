package offlineconvert

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// withSourceDatabase opens a consistent, single-file snapshot of the source and
// hands it to run.
//
// The snapshot is taken with SQLite's own "VACUUM INTO", which opens a read
// transaction on the source, so committed pages (including committed WAL pages)
// are included and concurrent readers cannot see a torn database. The result is
// a standalone file in journal_mode=DELETE inside a private scratch directory;
// the original file is only ever opened read-only for that statement.
func withSourceDatabase(ctx context.Context, path string, run func(*gorm.DB) error) error {
	snapshotPath, cleanup, err := snapshotSource(ctx, path)
	if err != nil {
		return err
	}
	defer cleanup()

	snapshot, err := openSnapshot(ctx, snapshotPath)
	if err != nil {
		return err
	}
	defer closeDatabase(snapshot)
	return run(snapshot)
}

func snapshotSource(ctx context.Context, path string) (string, func(), error) {
	directory, err := os.MkdirTemp("", "gpt-load-convert-source-")
	if err != nil {
		return "", nil, fmt.Errorf("offline conversion: create source scratch directory: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(directory) }
	snapshotPath := filepath.Join(directory, filepath.Base(path))

	source, err := openSourceForSnapshot(ctx, path)
	if err != nil {
		cleanup()
		return "", nil, err
	}
	// VACUUM INTO takes the value as a bound parameter, so the destination is
	// never interpolated into SQL.
	vacuumErr := source.WithContext(ctx).Exec("VACUUM INTO ?", snapshotPath).Error
	closeDatabase(source)
	// Opening a WAL-mode database leaves SQLite's -wal/-shm next to it. They were
	// not there before the conversion and they hold no committed source content
	// once every connection is closed, so the source directory is restored to the
	// state the operator left it in.
	removeReaderFootprint(path)
	if vacuumErr != nil {
		cleanup()
		return "", nil, fmt.Errorf("offline conversion: snapshot source database: %w", vacuumErr)
	}
	// A snapshot must be one file: anything else would make publication
	// non-atomic or depend on a sidecar journal.
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		if _, err := os.Lstat(snapshotPath + suffix); err == nil {
			cleanup()
			return "", nil, fmt.Errorf(
				"offline conversion: source snapshot produced %s; refusing a multi-file snapshot",
				filepath.Base(snapshotPath)+suffix,
			)
		}
	}
	return snapshotPath, cleanup, nil
}

// removeReaderFootprint deletes the empty write-ahead log and shared-memory index
// that reading a WAL-mode database creates. A non-empty WAL is never removed: it
// holds committed pages that belong to the source.
func removeReaderFootprint(path string) {
	if info, err := os.Stat(path + "-wal"); err == nil && info.Size() == 0 {
		_ = os.Remove(path + "-wal")
	}
	_ = os.Remove(path + "-shm")
}

// openSourceForSnapshot opens the original database read-only for the snapshot
// statement.
//
// mode=ro makes the read-only intent a property of the SQLite connection instead
// of a property of the converter's good behaviour: a bug elsewhere cannot write
// to the operator's legacy database. immutable=1 must not be used, because it
// tells SQLite to ignore the -wal file and a source in WAL mode would then be
// read without its committed but not yet checkpointed pages. VACUUM INTO is one
// statement, so it observes a single consistent read.
func openSourceForSnapshot(ctx context.Context, path string) (*gorm.DB, error) {
	db, err := gorm.Open(sqlite.Open("file:"+path+"?mode=ro"), &gorm.Config{
		Logger: logger.Discard,
	})
	if err != nil {
		return nil, fmt.Errorf("offline conversion: open source database: %w", err)
	}
	var tables int64
	if err := db.WithContext(ctx).Raw("SELECT count(*) FROM sqlite_master").Scan(&tables).Error; err != nil {
		closeDatabase(db)
		return nil, fmt.Errorf("offline conversion: read source schema: %w", err)
	}
	return db, nil
}

func openSnapshot(ctx context.Context, path string) (*gorm.DB, error) {
	db, err := gorm.Open(sqlite.Open("file:"+path+"?mode=ro"), &gorm.Config{
		Logger: logger.Discard,
	})
	if err != nil {
		return nil, fmt.Errorf("offline conversion: open source snapshot: %w", err)
	}
	var tables int64
	if err := db.WithContext(ctx).Raw("SELECT count(*) FROM sqlite_master").Scan(&tables).Error; err != nil {
		closeDatabase(db)
		return nil, fmt.Errorf("offline conversion: read source snapshot schema: %w", err)
	}
	return db, nil
}

// requireSourceIntegrity rejects a damaged source before any table is read.
func requireSourceIntegrity(source *gorm.DB) error {
	var integrity string
	if err := source.Raw("PRAGMA integrity_check").Scan(&integrity).Error; err != nil {
		return fmt.Errorf("offline conversion: check source integrity: %w", err)
	}
	if integrity != "ok" {
		return fmt.Errorf("offline conversion: source integrity check = %q", integrity)
	}
	var violations []struct {
		Table  string `gorm:"column:table"`
		RowID  int64  `gorm:"column:rowid"`
		Parent string `gorm:"column:parent"`
	}
	if err := source.Raw("PRAGMA foreign_key_check").Scan(&violations).Error; err != nil {
		return fmt.Errorf("offline conversion: check source foreign keys: %w", err)
	}
	if len(violations) != 0 {
		return fmt.Errorf("offline conversion: source has %d foreign key violations", len(violations))
	}
	return nil
}
