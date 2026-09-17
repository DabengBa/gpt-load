package storage

import (
	"testing"

	"gpt-load/internal/platform/config"
)

func TestNewDatabaseDialectorSelectsSQLite(t *testing.T) {
	database, err := config.ParseDatabaseDSN(":memory:")
	if err != nil {
		t.Fatalf("ParseDatabaseDSN() error = %v", err)
	}
	dialector, err := newDatabaseDialector(database)
	if err != nil {
		t.Fatalf("newDatabaseDialector() error = %v", err)
	}
	if got, want := dialector.Name(), string(config.DatabaseDriverSQLite); got != want {
		t.Fatalf("dialector.Name() = %q, want %q", got, want)
	}
}

func TestOpenSQLiteURLUsesCommonLifecycleAndSQLiteRuntime(t *testing.T) {
	db, err := Open("sqlite:///:memory:")
	if err != nil {
		t.Fatalf("Open(sqlite URL) error = %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	t.Cleanup(func() {
		if err := sqlDB.Close(); err != nil {
			t.Errorf("close database: %v", err)
		}
	})

	if !db.Config.TranslateError {
		t.Fatal("TranslateError = false, want true")
	}
	if err := sqlDB.Ping(); err != nil {
		t.Fatalf("Ping() error = %v", err)
	}
	stats := sqlDB.Stats()
	if stats.MaxOpenConnections != 1 {
		t.Fatalf("SQLite pool max open connections = %d, want 1", stats.MaxOpenConnections)
	}
}
