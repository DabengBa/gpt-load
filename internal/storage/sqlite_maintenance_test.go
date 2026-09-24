package storage_test

import (
	"path/filepath"
	"testing"

	"gpt-load/internal/storage"
)

func TestSQLiteMaintenanceReportsJournalAndFreelistWithoutCompactOnInspection(t *testing.T) {
	db, err := storage.Open(filepath.Join(t.TempDir(), "maintenance.db"))
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	if err := db.Exec("CREATE TABLE scratch (payload TEXT)").Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("INSERT INTO scratch(payload) VALUES (zeroblob(100000))").Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("DELETE FROM scratch").Error; err != nil {
		t.Fatal(err)
	}
	before, err := storage.InspectSQLite(t.Context(), db)
	if err != nil {
		t.Fatal(err)
	}
	if before.JournalMode != "wal" || before.FreelistPages <= 0 || before.MaintenanceMode != "offline_only" {
		t.Fatalf("InspectSQLite() = %+v, want WAL, freelist and offline-only maintenance", before)
	}
	after, err := storage.InspectSQLite(t.Context(), db)
	if err != nil || after.FreelistPages != before.FreelistPages {
		t.Fatalf("read-only inspection changed freelist: before=%+v after=%+v err=%v", before, after, err)
	}
}

func TestSQLiteOrdinaryModeMemoryDSNRemainsFileBacked(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ordinary.db")
	db, err := storage.Open(path + "?mode=memory")
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	status, err := storage.InspectSQLite(t.Context(), db)
	if err != nil || status.JournalMode != "wal" {
		t.Fatalf("ordinary mode=memory inspection = %+v, %v, want WAL", status, err)
	}
}
