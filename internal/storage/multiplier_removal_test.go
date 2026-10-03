package storage

import (
	"reflect"
	"strings"
	"testing"

	"gorm.io/gorm"
)

func TestMigrationNumericPrefixContract(t *testing.T) {
	for _, test := range []struct {
		name  string
		ids   []string
		valid bool
	}{
		{"gap", []string{"0001_initial", "0008_previous", "0010_next"}, true},
		{"duplicate numeric", []string{"0001_initial", "0001_other"}, false},
		{"zero", []string{"0000_initial"}, false},
		{"reverse", []string{"0001_initial", "0010_next", "0008_previous"}, false},
		{"malformed", []string{"0001_initial", "10_next"}, false},
		{"first not initial", []string{"0008_previous"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			var entries []migration
			for _, id := range test.ids {
				entries = append(entries, migration{ID: id, Up: func(*gorm.DB) error { return nil }, Validate: func(*gorm.DB) error { return nil }})
			}
			if err := validateMigrationRegistry(entries); (err == nil) != test.valid {
				t.Fatalf("validate %v = %v, want valid %t", test.ids, err, test.valid)
			}
		})
	}
}

func TestMultiplierFreshSchemaAndGapContinuation(t *testing.T) {
	db := openInternalMigrationTestDatabase(t)
	if err := applyMigrationRegistry(db, migrations[:8]); err != nil {
		t.Fatal(err)
	}
	if migrations[8].ID != "0010_single_credential_per_group" {
		t.Fatalf("next migration = %s", migrations[8].ID)
	}
	if err := AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	if err := AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"groups", "access_keys"} {
		if db.Migrator().HasColumn(table, "price_multiplier_micros") {
			t.Fatalf("%s retains multiplier column", table)
		}
	}
}

func TestMultiplierRetiredLedgerUpgradeAndRepeatStart(t *testing.T) {
	for _, test := range []struct {
		name    string
		applied int
	}{
		{"pending migrations", 8},
		{"fully applied", len(migrations)},
	} {
		t.Run(test.name, func(t *testing.T) {
			db := openInternalMigrationTestDatabase(t)
			if err := applyMigrationRegistry(db, migrations[:test.applied]); err != nil {
				t.Fatal(err)
			}
			if err := db.Create(&schemaMigration{ID: "0009_price_multipliers"}).Error; err != nil {
				t.Fatal(err)
			}
			for start := 0; start < 2; start++ {
				if err := AutoMigrate(db); err != nil {
					t.Fatalf("start %d with retired ledger entry: %v", start, err)
				}
			}
			var ids []string
			if err := db.Table(migrationLedgerTable).Order("id").Pluck("id", &ids).Error; err != nil {
				t.Fatal(err)
			}
			var want []string
			for index, entry := range migrations {
				if index == 8 {
					want = append(want, "0009_price_multipliers")
				}
				want = append(want, entry.ID)
			}
			if !reflect.DeepEqual(ids, want) {
				t.Fatalf("ledger = %v, want %v", ids, want)
			}
			assertFeedbackSchema0024(t, db)
		})
	}
}

func TestMultiplierLedgerRejectionDoesNotMutate(t *testing.T) {
	for _, test := range []struct {
		name string
		ids  []string
	}{
		{"retired with missing active migrations", []string{"0001_initial", "0009_price_multipliers", "0010_single_credential_per_group"}},
		{"unknown at retired number", []string{"0001_initial", "0009_unknown"}},
		{"unknown", []string{"0001_initial", "0002_unknown"}},
		{"missing", []string{"0001_initial", "0003_remove_observation_fresh_until"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			db := openInternalMigrationTestDatabase(t)
			if err := db.Exec("CREATE TABLE schema_migrations (id TEXT PRIMARY KEY NOT NULL)").Error; err != nil {
				t.Fatal(err)
			}
			if err := db.Exec("CREATE TABLE business_data (value TEXT NOT NULL)").Error; err != nil {
				t.Fatal(err)
			}
			if err := db.Exec("INSERT INTO business_data VALUES ('preserved')").Error; err != nil {
				t.Fatal(err)
			}
			for _, id := range test.ids {
				if err := db.Exec("INSERT INTO schema_migrations VALUES (?)", id).Error; err != nil {
					t.Fatal(err)
				}
			}
			var before, after []string
			if err := db.Raw("SELECT COALESCE(sql, '') FROM sqlite_master ORDER BY name").Scan(&before).Error; err != nil {
				t.Fatal(err)
			}
			err := AutoMigrate(db)
			if err == nil || !strings.Contains(err.Error(), "unknown or non-contiguous") {
				t.Fatalf("ledger rejection = %v", err)
			}
			if err := db.Raw("SELECT COALESCE(sql, '') FROM sqlite_master ORDER BY name").Scan(&after).Error; err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before, after) {
				t.Fatalf("schema changed: %v -> %v", before, after)
			}
			var ids []string
			if err := db.Table("schema_migrations").Order("id").Pluck("id", &ids).Error; err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(ids, test.ids) {
				t.Fatalf("ledger changed: %v", ids)
			}
			var values []string
			if err := db.Table("business_data").Pluck("value", &values).Error; err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(values, []string{"preserved"}) {
				t.Fatalf("business data changed: %v", values)
			}
		})
	}
}
