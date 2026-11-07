package control

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"gpt-load/internal/platform/httpclient"
	"gpt-load/internal/storage/models"
)

func TestSiteInfoStatusAnnouncementProtocolIsolation(t *testing.T) {
	for _, tc := range []struct {
		name, announcements string
		wantCount           int
		keepOld             bool
	}{
		{"mixed extra types", `[{"content":"object","extra":{"link":"https://example.com"}},{"content":"number","extra":42},{"content":"text","extra":"details"},{"content":123},{"content":"last","extra":null}]`, 4, false},
		{"malformed announcement collection", `{"unexpected":true}`, 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprintf(w, `{"success":true,"data":{"system_name":"Example","quota_per_unit":100000,"announcements":%s}}`, tc.announcements)
				case "/api/user/self":
					fmt.Fprint(w, `{"success":true,"data":{"quota":1000000}}`)
				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			f := newServiceFixture(t)
			id := createGroupWithCredentials(t, f, "sk-announcement-protocol")
			enc, err := f.encryption.Encrypt("token")
			if err != nil {
				t.Fatal(err)
			}
			cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}
			if err := f.db.Create(&cfg).Error; err != nil {
				t.Fatal(err)
			}
			old := models.SiteAnnouncement{GroupID: id, SiteRootURL: server.URL, Source: "announcement", DedupKey: "old", Content: "old"}
			if err := f.db.Create(&old).Error; err != nil {
				t.Fatal(err)
			}
			c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
			if err := c.syncGroup(t.Context(), id); err != nil {
				t.Fatal(err)
			}
			cfg = models.GroupSiteConfig{}
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
				t.Fatal(err)
			}
			if cfg.RemainingAmount == nil || *cfg.RemainingAmount != 10 || cfg.SiteName != "Example" {
				t.Fatalf("announcement decoding broke balance: %+v", cfg)
			}
			var rows []models.SiteAnnouncement
			if err := f.db.Where("group_id = ? AND source = ?", id, "announcement").Find(&rows).Error; err != nil {
				t.Fatal(err)
			}
			if len(rows) != tc.wantCount {
				t.Fatalf("announcements: %+v", rows)
			}
			got := map[string]string{}
			for _, row := range rows {
				got[row.Content] = row.Extra
			}
			if tc.keepOld {
				if _, ok := got["old"]; !ok {
					t.Fatalf("old announcement lost: %+v", rows)
				}
				return
			}
			for _, content := range []string{"object", "number", "text", "last"} {
				if _, ok := got[content]; !ok {
					t.Fatalf("missing %s: %+v", content, rows)
				}
			}
			if got["object"] != "" || got["number"] != "" || got["last"] != "" || got["text"] != "details" {
				t.Fatalf("wrong extras: %+v", got)
			}
		})
	}
}

func TestSiteInfoAnnouncementOptionalTypesReplaceOld(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprint(w, `{"success":true,"data":{"quota_per_unit":100000,"announcements":[{"content":"numeric","type":1,"publishDate":"2025-01-02T03:04:05Z","extra":{}},{"content":"object","type":{"x":1},"publishDate":"2025-01-02T03:04:05Z"},{"content":"date-invalid","type":"warning","publishDate":42,"extra":false},{"content":"normal","type":"info","publishDate":"2025-01-02T03:04:05Z","extra":"detail"}]}}`)
		case "/api/user/self":
			fmt.Fprint(w, `{"success":true,"data":{"quota":1000000}}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-mixed-announcements")
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}
	if err := f.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	old := models.SiteAnnouncement{GroupID: id, SiteRootURL: server.URL, Source: "announcement", DedupKey: "old", Content: "old"}
	if err := f.db.Create(&old).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	var rows []models.SiteAnnouncement
	if err := f.db.Where("group_id = ? AND source = ?", id, "announcement").Find(&rows).Error; err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 {
		t.Fatalf("old announcement not replaced by all valid contents: %+v", rows)
	}
	got := map[string]models.SiteAnnouncement{}
	for _, row := range rows {
		got[row.Content] = row
	}
	date, _ := time.Parse(time.RFC3339, "2025-01-02T03:04:05Z")
	for _, content := range []string{"numeric", "object"} {
		row, ok := got[content]
		if !ok || row.Level != "default" || row.EventAtMS != date.UnixMilli() {
			t.Fatalf("%s not normalized with timestamp: %+v", content, row)
		}
	}
	if row := got["date-invalid"]; row.Level != "warning" || row.EventAtMS <= 0 || row.Extra != "" {
		t.Fatalf("invalid optional fields: %+v", row)
	}
	if row := got["normal"]; row.Level != "info" || row.EventAtMS != date.UnixMilli() || row.Extra != "detail" {
		t.Fatalf("normal fields changed: %+v", row)
	}
}
