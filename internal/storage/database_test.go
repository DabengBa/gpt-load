package storage

import (
	"strings"
	"testing"

	"gpt-load/internal/platform/config"
)

func TestDatabasePoolLimitsUseConfiguredValuesForPostgreSQL(t *testing.T) {
	pool := config.DatabasePoolConfig{
		MaxOpenConnections: 24,
		MaxIdleConnections: 12,
	}

	maxOpen, maxIdle := databasePoolLimits(config.DatabaseDriverPostgreSQL, pool)
	if maxOpen != 24 || maxIdle != 12 {
		t.Fatalf("databasePoolLimits(%q) = %d/%d, want 24/12", config.DatabaseDriverPostgreSQL, maxOpen, maxIdle)
	}
}

func TestDatabasePoolLimitsForceSQLiteSingleConnection(t *testing.T) {
	maxOpen, maxIdle := databasePoolLimits(config.DatabaseDriverSQLite, config.DatabasePoolConfig{
		MaxOpenConnections: 24,
		MaxIdleConnections: 12,
	})
	if maxOpen != 1 || maxIdle != 1 {
		t.Fatalf("databasePoolLimits(SQLite) = %d/%d, want 1/1", maxOpen, maxIdle)
	}
}

func TestNewDatabaseDialectorSelectsSupportedDrivers(t *testing.T) {
	for _, test := range []struct {
		name       string
		dsn        string
		wantDriver config.DatabaseDriver
	}{
		{name: "sqlite", dsn: ":memory:", wantDriver: config.DatabaseDriverSQLite},
		{name: "postgres", dsn: "postgres://user:password@db.example:5432/gpt_load", wantDriver: config.DatabaseDriverPostgreSQL},
	} {
		t.Run(test.name, func(t *testing.T) {
			database, err := config.ParseDatabaseDSN(test.dsn)
			if err != nil {
				t.Fatalf("ParseDatabaseDSN() error = %v", err)
			}
			dialector, err := newDatabaseDialector(database)
			if err != nil {
				t.Fatalf("newDatabaseDialector() error = %v", err)
			}
			if got := dialector.Name(); got != string(test.wantDriver) {
				t.Fatalf("dialector.Name() = %q, want %q", got, test.wantDriver)
			}
		})
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

func TestOpenNetworkDatabaseRejectsManagedSourceBeforeConnecting(t *testing.T) {
	for _, dsn := range []string{
		"postgres://user:password@db.example:5432/gpt_load",
	} {
		_, err := OpenWithSource(dsn, config.DatabaseSourceManaged)
		if err == nil {
			t.Fatalf("OpenWithSource(%q, managed) error = nil, want source validation error", dsn)
		}
		if !strings.Contains(err.Error(), "managed source") {
			t.Fatalf("OpenWithSource(%q, managed) error = %v, want managed-source error", dsn, err)
		}
	}
}
