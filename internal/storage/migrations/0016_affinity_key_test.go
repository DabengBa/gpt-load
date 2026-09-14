package migrations_test

import (
	"reflect"
	"strings"
	"testing"

	"gorm.io/gorm"
	"gpt-load/internal/storage/migrations"
)

func Test0016AffinityKeyAddsSafeDefaultAndOrderedNonUniqueIndex(t *testing.T) {
	db := openInitialTestDatabase(t)
	applyMigrationsThrough0015(t, db)
	legacy := legacyAffinityRequestLog("00000000-0000-4000-8000-000000000016")
	if err := db.Omit("AffinityKey").Create(&legacy).Error; err != nil {
		t.Fatalf("create legacy request log: %v", err)
	}

	if err := migrations.Up0016(db); err != nil {
		t.Fatalf("Up0016() error = %v", err)
	}
	if err := migrations.Validate0016(db); err != nil {
		t.Fatalf("Validate0016() error = %v", err)
	}
	columns := initialColumns(t, db, "request_logs")
	column, ok := columns["affinity_key"]
	if !ok || column.NotNull != 1 || column.DefaultValue == nil ||
		strings.Trim(*column.DefaultValue, "'\"") != "" {
		t.Fatalf("affinity_key column = %#v, want NOT NULL DEFAULT ''", column)
	}
	var value string
	if err := db.Table("request_logs").Select("affinity_key").Where("id = ?", legacy.ID).Scan(&value).Error; err != nil {
		t.Fatalf("read legacy affinity_key: %v", err)
	}
	if value != "" {
		t.Fatalf("legacy affinity_key = %q, want empty", value)
	}

	var indexes []struct {
		Name   string
		Unique int
	}
	if err := db.Raw("PRAGMA index_list('request_logs')").Scan(&indexes).Error; err != nil {
		t.Fatalf("inspect request log indexes: %v", err)
	}
	var found bool
	for _, index := range indexes {
		if index.Name != "idx_request_logs_affinity_completed_id" {
			continue
		}
		found = true
		if index.Unique != 0 {
			t.Fatalf("affinity index unique = %d, want non-unique", index.Unique)
		}
		var indexed []struct {
			Name string
			Key  int
			Desc int
		}
		if err := db.Raw("PRAGMA index_xinfo('idx_request_logs_affinity_completed_id')").Scan(&indexed).Error; err != nil {
			t.Fatalf("inspect affinity index columns: %v", err)
		}
		var names []string
		var directions []int
		for _, column := range indexed {
			if column.Key == 1 {
				names = append(names, column.Name)
				directions = append(directions, column.Desc)
			}
		}
		if !reflect.DeepEqual(names, []string{"affinity_key", "completed_at_ms", "id"}) ||
			!reflect.DeepEqual(directions, []int{0, 1, 1}) {
			t.Fatalf("affinity index = %v directions %v, want ordered projection", names, directions)
		}
	}
	if !found {
		t.Fatal("affinity index is missing")
	}

	if err := migrations.Up0016(db); err != nil {
		t.Fatalf("repeated Up0016() error = %v", err)
	}
}

func Test0016AffinityKeyResumesAfterColumnWasAdded(t *testing.T) {
	db := openInitialTestDatabase(t)
	applyMigrationsThrough0015(t, db)
	if err := db.Exec("ALTER TABLE request_logs ADD COLUMN affinity_key VARCHAR(36) NOT NULL DEFAULT ''").Error; err != nil {
		t.Fatalf("pre-add affinity_key: %v", err)
	}
	if err := migrations.Up0016(db); err != nil {
		t.Fatalf("Up0016() after pre-added column error = %v", err)
	}
	if !db.Migrator().HasIndex("request_logs", "idx_request_logs_affinity_completed_id") {
		t.Fatal("affinity index missing after recoverable migration")
	}
}

func applyMigrationsThrough0015(t *testing.T, db *gorm.DB) {
	t.Helper()
	for _, migrate := range []func(*gorm.DB) error{
		migrations.Up0001,
		migrations.Up0002,
		migrations.Up0003,
		migrations.Up0004,
		migrations.Up0005,
		migrations.Up0006,
		migrations.Up0007,
		migrations.Up0008,
		migrations.Up0009,
		migrations.Up0010,
		migrations.Up0011,
		migrations.Up0012,
		migrations.Up0013,
		migrations.Up0014,
		migrations.Up0015,
	} {
		if err := migrate(db); err != nil {
			t.Fatalf("prepare pre-0016 schema: %v", err)
		}
	}
}
