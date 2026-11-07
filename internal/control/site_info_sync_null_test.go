package control

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/platform/httpclient"
	"gpt-load/internal/storage/models"
)

func TestSiteInfoRejectsNullAndMissingResponseData(t *testing.T) {
	tests := []struct {
		name, user, status, notice                             string
		wantFailure, wantUnknown, keepAnnouncement, keepNotice bool
	}{
		{"user data null", `null`, `{"quota_per_unit":100000,"announcements":[]}`, `"new notice"`, true, false, true, true},
		{"user quota missing", `{}`, `{"quota_per_unit":100000,"announcements":[]}`, `"new notice"`, true, false, true, true},
		{"user quota null", `{"quota":null}`, `{"quota_per_unit":100000,"announcements":[]}`, `"new notice"`, true, false, true, true},
		{"status data null", `{"quota":1000000}`, `null`, `"new notice"`, false, true, true, false},
		{"status quota null", `{"quota":1000000}`, `{"quota_per_unit":null,"announcements":[{"content":"new announcement"}]}`, `"new notice"`, false, true, false, false},
		{"status quota invalid", `{"quota":1000000}`, `{"quota_per_unit":"bad","announcements":[{"content":"new announcement"}]}`, `"new notice"`, false, true, false, false},
		{"notice data null", `{"quota":1000000}`, `{"quota_per_unit":100000,"announcements":[]}`, `null`, false, false, false, true},
		{"notice empty clears", `{"quota":1000000}`, `{"quota_per_unit":100000,"announcements":[]}`, `""`, false, false, false, false},
		{"zero quota and missing conversion", `{"quota":0}`, `{}`, `""`, false, false, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var data string
				switch r.URL.Path {
				case "/api/user/self":
					data = tt.user
				case "/api/status":
					data = tt.status
				case "/api/notice":
					data = tt.notice
				case "/api/token/":
					data = `[]`
				default:
					w.WriteHeader(http.StatusNotFound)
					return
				}
				fmt.Fprintf(w, `{"success":true,"data":%s}`, data)
			}))
			defer server.Close()
			f := newServiceFixture(t)
			id := createGroupWithCredentials(t, f, "sk-null-response")
			enc, err := f.encryption.Encrypt("token")
			if err != nil {
				t.Fatal(err)
			}
			oldQuota, oldAmount, oldUnit, oldTime := int64(700000), 7.0, 100000.0, int64(123)
			cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "ok", BalanceQuota: &oldQuota, RemainingAmount: &oldAmount, QuotaPerUnit: &oldUnit, FetchedAtMS: &oldTime}
			if err := f.db.Create(&cfg).Error; err != nil {
				t.Fatal(err)
			}
			for _, source := range []string{"announcement", "notice"} {
				row := models.SiteAnnouncement{GroupID: id, SiteRootURL: server.URL, Source: source, DedupKey: "old-" + source, Content: "old-" + source}
				if err := f.db.Create(&row).Error; err != nil {
					t.Fatal(err)
				}
			}
			c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
			if err := c.syncGroup(t.Context(), id); err != nil {
				t.Fatal(err)
			}
			cfg = models.GroupSiteConfig{}
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
				t.Fatal(err)
			}
			if tt.wantFailure {
				if cfg.Status != "error" || cfg.BalanceQuota == nil || *cfg.BalanceQuota != oldQuota || cfg.RemainingAmount == nil || *cfg.RemainingAmount != oldAmount || cfg.FetchedAtMS == nil || *cfg.FetchedAtMS != oldTime {
					t.Fatalf("invalid user overwrote snapshot: %+v", cfg)
				}
			} else if cfg.BalanceQuota == nil || (tt.wantUnknown && (cfg.Status != "error" || cfg.RemainingAmount == nil || *cfg.RemainingAmount != oldAmount || cfg.QuotaPerUnit == nil || *cfg.QuotaPerUnit != oldUnit)) || (!tt.wantUnknown && (cfg.Status != "ok" || cfg.RemainingAmount == nil)) {
				t.Fatalf("unexpected balance: %+v", cfg)
			}
			if tt.name == "zero quota and missing conversion" && (*cfg.BalanceQuota != 0 || *cfg.RemainingAmount != 0 || cfg.QuotaPerUnit == nil || *cfg.QuotaPerUnit != 500000) {
				t.Fatalf("valid zero quota or absent conversion rejected: %+v", cfg)
			}
			var rows []models.SiteAnnouncement
			if err := f.db.Where("group_id = ?", id).Find(&rows).Error; err != nil {
				t.Fatal(err)
			}
			old := map[string]bool{}
			for _, row := range rows {
				if row.Content == "old-"+row.Source {
					old[row.Source] = true
				}
			}
			if old["announcement"] != tt.keepAnnouncement || old["notice"] != tt.keepNotice {
				t.Fatalf("source retention mismatch: %+v", rows)
			}
			if tt.name == "status quota null" || tt.name == "status quota invalid" {
				found := false
				for _, row := range rows {
					if row.Source == "announcement" && row.Content == "new announcement" {
						found = true
					}
				}
				if !found || cfg.AnnouncementStatus != "ok" || cfg.AnnouncementFetchedAtMS == nil {
					t.Fatalf("valid announcement not synced: rows=%+v cfg=%+v", rows, cfg)
				}
			}
			if tt.name == "notice empty clears" && len(rows) != 0 {
				t.Fatalf("empty notice must clear: %+v", rows)
			}
		})
	}
}
