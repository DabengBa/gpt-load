package control

import (
	"strings"
	"sync"
	"testing"

	"gorm.io/gorm"
	"gpt-load/internal/storage/models"
)

func TestAnnouncementsReadConsistentSnapshotDuringSync(t *testing.T) {
	f, _ := newFileServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-snapshot")
	now := f.service.now().UnixMilli()
	if err := f.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: "cipher", SiteRootURL: "https://site.example", AnnouncementStatus: "error", UpdatedAtMS: now}).Error; err != nil {
		t.Fatal(err)
	}
	if err := f.db.Create(&models.SiteAnnouncement{GroupID: id, SiteRootURL: "https://site.example", DedupKey: "old", Content: "old", Source: "announcement", EventAtMS: now, FetchedAtMS: now}).Error; err != nil {
		t.Fatal(err)
	}
	var once sync.Once
	var writeErr error
	interleaved := false
	if err := f.db.Callback().Row().Before("gorm:row").Register("test:interleave_announcement_sync", func(tx *gorm.DB) {
		if !strings.Contains(tx.Statement.Table, "group_site_configs") {
			return
		}
		once.Do(func() {
			interleaved = true
			writeErr = f.db.Transaction(func(w *gorm.DB) error {
				if err := w.Model(&models.GroupSiteConfig{}).Where("group_id = ?", id).Updates(map[string]any{"announcement_status": "ok", "announcement_fetched_at_ms": now}).Error; err != nil {
					return err
				}
				return w.Model(&models.SiteAnnouncement{}).Where("group_id = ?", id).Update("content", "new").Error
			})
		})
	}); err != nil {
		t.Fatal(err)
	}
	defer f.db.Callback().Row().Remove("test:interleave_announcement_sync")
	got, err := f.service.GetAnnouncements(t.Context(), "", "")
	if writeErr != nil {
		t.Fatalf("interleaved commit: %v", writeErr)
	}
	if !interleaved {
		t.Fatal("sync did not commit between the two reads")
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Items) != 1 || len(got.Sources) != 1 || got.Items[0].Content != "old" || got.Sources[0].Announcement.Status != "error" {
		t.Fatalf("mixed snapshot: %+v", got)
	}
	next, err := f.service.GetAnnouncements(t.Context(), "", "")
	if err != nil || len(next.Items) != 1 || len(next.Sources) != 1 || next.Items[0].Content != "new" || next.Sources[0].Announcement.Status != "ok" {
		t.Fatalf("committed result: %+v err=%v", next, err)
	}
}
