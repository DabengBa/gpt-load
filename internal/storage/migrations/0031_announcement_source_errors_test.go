package migrations_test

import (
	"testing"

	"gpt-load/internal/storage/migrations"
)

func TestAnnouncementSourceErrorsMigrationPreservesRowsAndIsIdempotent(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := db.Exec("CREATE TABLE groups (id integer primary key)").Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("INSERT INTO groups(id) VALUES (1)").Error; err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0029(db); err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0030(db); err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("INSERT INTO group_site_configs(group_id,updated_at_ms,last_error,announcement_status) VALUES (1,1,'old error','error')").Error; err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if err := migrations.Up0031(db); err != nil {
			t.Fatal(err)
		}
	}
	var row struct{ LastError, AnnouncementLastError, NoticeLastError, AnnouncementStatus string }
	if err := db.Table("group_site_configs").Where("group_id = 1").Take(&row).Error; err != nil {
		t.Fatal(err)
	}
	if row.LastError != "old error" || row.AnnouncementStatus != "error" || row.AnnouncementLastError != "" || row.NoticeLastError != "" {
		t.Fatalf("migration changed legacy state: %+v", row)
	}
}
