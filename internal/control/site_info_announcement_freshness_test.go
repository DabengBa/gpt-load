package control

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/platform/httpclient"
	"gpt-load/internal/storage/models"
)

func TestAnnouncementSourceFreshness(t *testing.T) {
	initControlI18n(t)
	fail := false
	coreFail := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/user/self":
			if coreFail {
				w.WriteHeader(503)
			} else {
				fmt.Fprint(w, `{"success":true,"data":{"quota":1}}`)
			}
		case "/api/status":
			if fail {
				w.WriteHeader(503)
			} else {
				fmt.Fprint(w, `{"success":true,"data":{"announcements":[{"content":"old"}]}}`)
			}
		case "/api/notice":
			fmt.Fprint(w, `{"success":true,"data":"notice"}`)
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-abcdefghijklmno")
	enc, _ := f.encryption.Encrypt("token")
	if err := f.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}).Error; err != nil {
		t.Fatal(err)
	}
	check := func(announcement, notice string, items int) {
		t.Helper()
		result, err := f.service.GetAnnouncements(context.Background(), "", "")
		if err != nil {
			t.Fatal(err)
		}
		raw, _ := json.Marshal(result)
		var body struct {
			Items   []json.RawMessage `json:"items"`
			Sources []struct {
				Announcement struct {
					Status string `json:"status"`
				} `json:"announcement"`
				Notice struct {
					Status string `json:"status"`
				} `json:"notice"`
			} `json:"sources"`
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}
		if len(body.Sources) != 1 || body.Sources[0].Announcement.Status != announcement || body.Sources[0].Notice.Status != notice || len(body.Items) != items {
			t.Fatalf("unexpected feed: %s", raw)
		}
	}
	check("pending", "pending", 0)
	c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
	if err := c.syncGroup(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	check("ok", "ok", 2)
	fail = true
	if err := c.syncGroup(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	check("error", "ok", 2)
	// A successful but old source is stale even when no subsequent attempt ran.
	if err := f.db.Model(&models.GroupSiteConfig{}).Where("group_id = ?", id).Update("notice_fetched_at_ms", time.Now().Add(-3*time.Hour).UnixMilli()).Error; err != nil {
		t.Fatal(err)
	}
	check("error", "stale", 2)
	coreFail = true
	if err := c.syncGroup(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	check("error", "error", 2)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, f.service).RegisterRoutes(engine)
	legacy := performGroupCollectionRequest(engine, "/api/announcements", "Bearer test-auth-key")
	if legacy.Code != http.StatusOK || strings.Contains(legacy.Body.String(), `"sources"`) {
		t.Fatalf("legacy response: %s", legacy.Body.String())
	}
	modern := performGroupCollectionRequest(engine, "/api/announcements?include=source_status", "Bearer test-auth-key")
	if modern.Code != http.StatusOK || !strings.Contains(modern.Body.String(), `"sources"`) || strings.Contains(modern.Body.String(), "token") {
		t.Fatalf("modern response: %s", modern.Body.String())
	}
}
