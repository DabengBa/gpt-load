package migrations_test

import (
	"gpt-load/internal/storage/migrations"
	"testing"
)

func TestAnnouncementSourceMigrationPreservesLegacyData(t *testing.T) {
	db := openInitialTestDatabase(t)
	for _, statement := range []string{
		"CREATE TABLE groups (id integer primary key)",
		"INSERT INTO groups(id) VALUES (1), (2)",
	} {
		if err := db.Exec(statement).Error; err != nil {
			t.Fatal(err)
		}
	}
	if err := migrations.Up0029(db); err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("INSERT INTO group_site_configs(group_id, updated_at_ms, last_attempt_at_ms) VALUES (1, 1, 1), (2, 1, NULL)").Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("INSERT INTO site_announcements(group_id,site_root_url,dedup_key,content,event_at_ms,fetched_at_ms) VALUES (1,'https://example.com','x','old',1,1)").Error; err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0030(db); err != nil {
		t.Fatal(err)
	}
	var rows []struct {
		GroupID            int
		AnnouncementStatus string
		NoticeStatus       string
	}
	if err := db.Table("group_site_configs").Order("group_id").Scan(&rows).Error; err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0].AnnouncementStatus != "unknown" || rows[0].NoticeStatus != "unknown" || rows[1].AnnouncementStatus != "pending" {
		t.Fatalf("status: %+v", rows)
	}
	var count int64
	if err := db.Table("site_announcements").Where("content = ?", "old").Count(&count).Error; err != nil || count != 1 {
		t.Fatalf("lost announcement: count=%d err=%v", count, err)
	}
}
