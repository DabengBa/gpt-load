package migrations_test

import (
	"gpt-load/internal/storage/migrations"
	"testing"
)

func TestSiteInfoMigrationRejectsInvalidDedupIndex(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := db.Exec("CREATE TABLE groups (id integer primary key)").Error; err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0029(db); err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("DROP INDEX idx_site_announcements_group_dedup").Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("CREATE INDEX idx_site_announcements_group_dedup ON site_announcements(group_id, dedup_key)").Error; err != nil {
		t.Fatal(err)
	}
	if err := migrations.Validate0029(db); err == nil {
		t.Fatal("non-unique dedup index accepted")
	}
}
