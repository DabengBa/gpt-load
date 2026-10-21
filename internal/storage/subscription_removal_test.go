package storage

import (
	"fmt"
	"reflect"
	"slices"

	"testing"

	"gorm.io/gorm"
)

func subscriptionSchemaSnapshot(t *testing.T, db *gorm.DB) []string {
	t.Helper()
	var schema []string
	if err := db.Raw(`SELECT type || ':' || name || ':' || coalesce(sql,'') FROM sqlite_master ORDER BY type,name`).Scan(&schema).Error; err != nil {
		t.Fatal(err)
	}
	var ledger []string
	if err := db.Table(migrationLedgerTable).Order("id").Pluck("id", &ledger).Error; err != nil {
		t.Fatal(err)
	}
	return append(schema, ledger...)
}

// credentialColumnNames returns the credential columns the database really
// declares, so a removal assertion can never pass by simply checking nothing.
func credentialColumnNames(t *testing.T, db *gorm.DB) []string {
	t.Helper()
	var declared []string
	if err := db.Raw("SELECT name FROM pragma_table_info('credentials') ORDER BY cid").Scan(&declared).Error; err != nil {
		t.Fatal(err)
	}
	return declared
}
func TestSubscriptionSchemaRemoved(t *testing.T) {
	db := openInternalMigrationTestDatabase(t)
	if err := applyMigrations(db); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"credential_stages", "credential_observations", "credential_reset_operations"} {
		if db.Migrator().HasTable(table) {
			t.Errorf("retired table %s remains", table)
		}
	}
	declared := credentialColumnNames(t, db)
	if len(declared) == 0 {
		t.Fatal("credentials schema exposes no columns; removal assertions would pass vacuously")
	}
	for _, retained := range []string{"secret_version", "identity_fingerprint", "fingerprint"} {
		if !slices.Contains(declared, retained) {
			t.Fatalf("credentials schema lost retained column %s: %v", retained, declared)
		}
	}
	for _, retired := range []string{"auth_state", "auth_error_code"} {
		if slices.Contains(declared, retired) {
			t.Fatalf("credentials schema still declares retired subscription column %s: %v", retired, declared)
		}
	}
	if err := applyMigrations(db); err != nil {
		t.Fatalf("reopen fresh schema: %v", err)
	}
}

func TestRejectSubscriptionSchemaBeforePendingMigrations(t *testing.T) {
	obsoleteColumns := []string{"auth_state", "auth_error_code"}
	for _, prefix := range []int{1, 2} {
		for _, obsolete := range append([]string{"credential_stages", "credential_observations", "credential_reset_operations"}, obsoleteColumns...) {
			t.Run(fmt.Sprintf("%d/%s", prefix, obsolete), func(t *testing.T) {
				db := openInternalMigrationTestDatabase(t)
				if err := applyMigrationRegistry(db, migrations[:prefix]); err != nil {
					t.Fatal(err)
				}
				if slices.Contains(obsoleteColumns, obsolete) {
					if !db.Migrator().HasColumn("credentials", obsolete) {
						if err := db.Exec("ALTER TABLE credentials ADD COLUMN " + obsolete + " TEXT").Error; err != nil {
							t.Fatal(err)
						}
					}
				} else {
					if !db.Migrator().HasTable(obsolete) {
						if err := db.Exec("CREATE TABLE " + obsolete + " (evidence TEXT)").Error; err != nil {
							t.Fatal(err)
						}
					}
				}
				if err := db.Exec("CREATE TABLE preserved_evidence (value TEXT)").Error; err != nil {
					t.Fatal(err)
				}
				if err := db.Exec("INSERT INTO preserved_evidence VALUES ('unchanged')").Error; err != nil {
					t.Fatal(err)
				}
				before := subscriptionSchemaSnapshot(t, db)
				if err := applyMigrations(db); err == nil {
					t.Fatal("obsolete subscription schema accepted")
				}
				if after := subscriptionSchemaSnapshot(t, db); !reflect.DeepEqual(before, after) {
					t.Fatal("rejection changed schema or ledger")
				}
				var value string
				if err := db.Raw("SELECT value FROM preserved_evidence").Scan(&value).Error; err != nil || value != "unchanged" {
					t.Fatalf("rejection changed data: %q, %v", value, err)
				}
			})
		}
	}
}

func TestRejectRetiredSubscriptionMigrationLedger(t *testing.T) {
	db := openInternalMigrationTestDatabase(t)
	if err := applyMigrationRegistry(db, migrations[:2]); err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&schemaMigration{ID: "0003_remove_observation_fresh_until"}).Error; err != nil {
		t.Fatal(err)
	}
	before := subscriptionSchemaSnapshot(t, db)
	if err := applyMigrations(db); err == nil {
		t.Fatal("retired 0003 ledger accepted")
	}
	if !reflect.DeepEqual(before, subscriptionSchemaSnapshot(t, db)) {
		t.Fatal("rejection changed schema or ledger")
	}
}
