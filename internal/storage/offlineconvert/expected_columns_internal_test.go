package offlineconvert

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"gpt-load/internal/storage"
)

// The expected column sets are hand-written, so they can drift away from the
// schema the application actually creates. A column the current schema gained but
// currentColumns does not list would be reported as an unexpected source column
// and block every conversion; a column it lost would keep the converter waiting
// for a value that no longer exists. This pins both maps to the real schema so
// such drift fails here, in one place, instead of on an operator's database.
func TestExpectedColumnSetsMatchCurrentSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "current.db")
	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open current schema database: %v", err)
	}
	defer closeDatabase(db)
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatalf("apply current schema: %v", err)
	}

	for _, table := range migratedTables {
		expected, known := currentColumns[table]
		if !known {
			t.Fatalf("migrated table %q has no expected column set", table)
		}
		var declared []string
		if err := db.Raw("SELECT name FROM pragma_table_info(?)", table).Scan(&declared).Error; err != nil {
			t.Fatalf("read current %s columns: %v", table, err)
		}
		if len(declared) == 0 {
			t.Fatalf("current schema declares no columns for %q; the comparison would pass vacuously", table)
		}
		if !sameSet(expected, declared) {
			t.Fatalf("%s expected columns drifted from the current schema:\n  expected-only: %v\n  schema-only:   %v",
				table, difference(expected, declared), difference(declared, expected))
		}
		// The copy reads exactly the expected list, so a duplicate there would
		// produce a duplicate projection and a broken INSERT.
		if duplicates := duplicateValues(expected); len(duplicates) != 0 {
			t.Fatalf("%s expected column set repeats %v", table, duplicates)
		}
	}
}

// Every table the current schema declares is either copied with an explicit
// column set or deliberately dropped. A new table that is neither would be
// refused by the converter at run time; catching it here names the omission.
func TestCurrentSchemaTablesAreClassified(t *testing.T) {
	path := filepath.Join(t.TempDir(), "current.db")
	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open current schema database: %v", err)
	}
	defer closeDatabase(db)
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatalf("apply current schema: %v", err)
	}

	var tables []string
	if err := db.Raw(
		"SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
	).Scan(&tables).Error; err != nil {
		t.Fatalf("list current schema tables: %v", err)
	}
	if len(tables) == 0 {
		t.Fatal("current schema declares no tables")
	}
	for _, table := range tables {
		switch {
		case slices.Contains(migratedTables, table):
		case slices.Contains(droppedTables, table):
		default:
			t.Errorf("current schema table %q is neither migrated nor dropped", table)
		}
	}
}

// The target is built by the application's own migration chain, so a retired
// column listed as "present in legacy sources" must really be gone from it.
// Otherwise every conversion would skip a column the target still needs.
func TestRetiredColumnsAreAbsentFromCurrentSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "current.db")
	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open current schema database: %v", err)
	}
	defer closeDatabase(db)
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatalf("apply current schema: %v", err)
	}

	retired := retiredColumns()
	if len(retired) == 0 {
		t.Fatal("no retired columns are declared")
	}
	for _, table := range migratedTables {
		for _, column := range retired {
			if db.Migrator().HasColumn(table, column) {
				t.Errorf("current schema still declares retired column %s.%s", table, column)
			}
		}
	}
}

func sameSet(left, right []string) bool {
	return len(difference(left, right)) == 0 && len(difference(right, left)) == 0
}

func difference(left, right []string) []string {
	var only []string
	for _, value := range left {
		if !slices.Contains(right, value) {
			only = append(only, value)
		}
	}
	slices.Sort(only)
	return only
}

func duplicateValues(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	var repeated []string
	for _, value := range values {
		if _, ok := seen[value]; ok {
			repeated = append(repeated, value)
			continue
		}
		seen[value] = struct{}{}
	}
	return repeated
}
