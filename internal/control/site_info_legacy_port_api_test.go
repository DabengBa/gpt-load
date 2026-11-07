package control

import (
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/storage/models"
)

func TestLegacyOutOfRangeProviderPortSiteAccessAPI(t *testing.T) {
	initControlI18n(t)
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-legacy-port-api")
	legacy := "https://legacy.example:99999/path"
	if err := f.db.Model(&models.Group{}).Where("id = ?", id).Update("provider_url", legacy).Error; err != nil {
		t.Fatal(err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, f.service).RegisterRoutes(engine)
	path := "/api/groups/" + stringGroupID(id) + "/settings"
	put := func(body string) int {
		t.Helper()
		return serveGroupSettingsRequest(t, engine, http.MethodPut, path, "test-auth-key", body).Code
	}
	if code := put(`{"site_access":{"user_id":"42","access_token":"new-site-token"}}`); code != http.StatusBadRequest {
		t.Fatalf("adding site credentials with legacy invalid port returned %d, want 400", code)
	}
	if code := put(`{"provider_url":"https://legacy.example:99998/path","name":"must not save"}`); code != http.StatusBadRequest {
		t.Fatalf("changing to another out-of-range port returned %d, want 400", code)
	}
	if code := put(`{"provider_url":"` + legacy + `","name":"legacy echoed"}`); code != http.StatusOK {
		t.Fatalf("echoing unchanged historical URL returned %d, want 200", code)
	}
	var group models.Group
	if err := f.db.First(&group, id).Error; err != nil {
		t.Fatal(err)
	}
	if group.ProviderURL == nil || *group.ProviderURL != legacy || group.Name != "legacy echoed" {
		t.Fatalf("unchanged legacy URL/name not preserved: %+v", group)
	}
	if code := put(`{"provider_url":"https://legacy.example:99999/other","name":"must not save"}`); code != http.StatusBadRequest {
		t.Fatalf("changing path on legacy invalid URL returned %d, want 400", code)
	}
	if code := put(`{"provider_url":" ` + legacy + ` ","name":"must not save"}`); code != http.StatusBadRequest {
		t.Fatalf("normalized match of changed raw URL returned %d, want 400", code)
	}
	if err := f.db.First(&group, id).Error; err != nil || group.Name != "legacy echoed" || group.ProviderURL == nil || *group.ProviderURL != legacy {
		t.Fatalf("rejected update changed group: %+v, err=%v", group, err)
	}
	if code := put(`{"enabled":false}`); code != http.StatusOK {
		t.Fatalf("unrelated update returned %d, want 200", code)
	}
}

func TestLegacyOutOfRangeProviderPortCanClearSiteAccessAPI(t *testing.T) {
	initControlI18n(t)
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-legacy-port-clear")
	legacy := "https://legacy.example:99999/path"
	if err := f.db.Model(&models.Group{}).Where("id = ?", id).Update("provider_url", legacy).Error; err != nil {
		t.Fatal(err)
	}
	cipher, err := f.encryption.Encrypt("old-site-token")
	if err != nil {
		t.Fatal(err)
	}
	if err := f.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: cipher, SiteRootURL: "https://legacy.example:99999", Status: "error", UpdatedAtMS: 1}).Error; err != nil {
		t.Fatal(err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, f.service).RegisterRoutes(engine)
	path := "/api/groups/" + stringGroupID(id) + "/settings"
	replacement := serveGroupSettingsRequest(t, engine, http.MethodPut, path, "test-auth-key", `{"site_access":{"access_token":"replacement"}}`)
	if replacement.Code != http.StatusBadRequest {
		t.Errorf("replacing credentials returned %d, want 400: %s", replacement.Code, replacement.Body)
	}
	var before models.GroupSiteConfig
	if err := f.db.First(&before, "group_id = ?", id).Error; err != nil || before.AccessToken != cipher {
		t.Fatalf("rejected replacement changed credentials: %+v, err=%v", before, err)
	}
	unrelated := serveGroupSettingsRequest(t, engine, http.MethodPut, path, "test-auth-key", `{"provider_url":"`+legacy+`","name":"still works"}`)
	if unrelated.Code != http.StatusOK {
		t.Fatalf("unrelated update with existing credentials returned %d: %s", unrelated.Code, unrelated.Body)
	}
	w := serveGroupSettingsRequest(t, engine, http.MethodPut, path, "test-auth-key", `{"provider_url":"`+legacy+`","site_access":{"clear_access_token":true}}`)
	if w.Code != http.StatusOK {
		t.Fatalf("clearing credentials returned %d: %s", w.Code, w.Body)
	}
	var cfg models.GroupSiteConfig
	if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.AccessToken != "" || cfg.SiteRootURL != "" {
		t.Fatalf("site access not cleared: %+v", cfg)
	}
}

func TestLegacyPortUserIDChangeAndLongPortAPI(t *testing.T) {
	initControlI18n(t)
	for _, port := range []string{"99999", strings.Repeat("9", 80)} {
		t.Run(port, func(t *testing.T) {
			f := newServiceFixture(t)
			id := createGroupWithCredentials(t, f, "sk-legacy-user-id")
			root := "https://legacy.example:" + port
			legacy := root + "/path"
			if err := f.db.Model(&models.Group{}).Where("id = ?", id).Update("provider_url", legacy).Error; err != nil {
				t.Fatal(err)
			}
			cipher, err := f.encryption.Encrypt("old-site-token")
			if err != nil {
				t.Fatal(err)
			}
			if err := f.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: cipher, SiteRootURL: root, Status: "error", UpdatedAtMS: 1}).Error; err != nil {
				t.Fatal(err)
			}
			engine := gin.New()
			NewServer(&config.Config{AuthKey: "test-auth-key"}, f.service).RegisterRoutes(engine)
			path := "/api/groups/" + stringGroupID(id) + "/settings"
			put := func(body string, expected int) {
				t.Helper()
				w := serveGroupSettingsRequest(t, engine, http.MethodPut, path, "test-auth-key", body)
				if w.Code != expected {
					t.Errorf("body %s: got %d, want %d: %s", body, w.Code, expected, w.Body)
				}
			}
			put(`{"name":"unrelated"}`, http.StatusOK)
			put(`{"provider_url":"`+legacy+`","site_access":{"user_id":" 42 "}}`, http.StatusOK)
			put(`{"name":"must not save","site_access":{"user_id":"43"}}`, http.StatusBadRequest)
			var cfg models.GroupSiteConfig
			var group models.Group
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
				t.Fatal(err)
			}
			if err := f.db.First(&group, id).Error; err != nil {
				t.Fatal(err)
			}
			if cfg.UserID != "42" || cfg.AccessToken != cipher || cfg.ConfigVersion != 1 || group.Name != "unrelated" {
				t.Errorf("unexpected persisted changes: cfg=%+v name=%s", cfg, group.Name)
			}
			put(`{"provider_url":"`+root+`/changed"}`, http.StatusBadRequest)
			put(`{"site_access":{"access_token":"new-token"}}`, http.StatusBadRequest)
			put(`{"provider_url":"`+legacy+`","site_access":{"user_id":"43","clear_access_token":true}}`, http.StatusOK)
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil || cfg.AccessToken != "" || cfg.SiteRootURL != "" {
				t.Errorf("clear failed: %+v err=%v", cfg, err)
			}
			put(`{"provider_url":"https://valid.example","site_access":{"user_id":"42","access_token":"new-token"}}`, http.StatusOK)
			put(`{"site_access":{"user_id":"43"}}`, http.StatusOK)
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil || cfg.UserID != "43" || cfg.AccessToken == "" {
				t.Errorf("valid URL user change failed: %+v err=%v", cfg, err)
			}
		})
	}
}
