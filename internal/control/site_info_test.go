package control

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/httpclient"
	"gpt-load/internal/platform/i18n"
	"gpt-load/internal/storage/models"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSiteInfoTokenGroup(t *testing.T) {
	tests := []struct {
		name, rawKey, userGroup, wantToken, wantEffective string
		tokens                                            []siteToken
	}{
		{"short", "123", "default", "", "", []siteToken{{Key: "123", Group: "x", Status: 1}}},
		{"none", "abcdefgh", "default", "", "", nil},
		{"explicit", "abcdefgh", "user", "vip", "vip", []siteToken{{Key: "abcd****efgh", Group: "vip", Status: 1}}},
		{"follow", "abcdefgh", "user", "", "user", []siteToken{{Key: "abcd****efgh", Status: 1}}},
		{"mixed", "abcdefgh", "user", "", "user", []siteToken{{Key: "abcd****efgh", Group: "user", Status: 1}, {Key: "abcd****efgh", Status: 1}}},
		{"ambiguous", "abcdefgh", "user", "", "", []siteToken{{Key: "abcd****efgh", Group: "a", Status: 1}, {Key: "abcd****efgh", Group: "b", Status: 1}}},
		{"auto", "abcdefgh", "user", "auto", "auto", []siteToken{{Key: "abcd****efgh", Group: "auto", Status: 1}}},
		{"disabled", "abcdefgh", "user", "", "", []siteToken{{Key: "abcd****efgh", Group: "x", Status: 2}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			token, effective := siteInfoTokenGroup(tt.rawKey, tt.userGroup, tt.tokens)
			if token != tt.wantToken || effective != tt.wantEffective {
				t.Errorf("got %q, %q; want %q, %q", token, effective, tt.wantToken, tt.wantEffective)
			}
		})
	}
}

func TestSiteInfoAuthenticatedFetchRetry(t *testing.T) {
	var headers []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = append(headers, r.Header.Get("Authorization"))
		if r.Header.Get("New-Api-User") != "42" {
			t.Error("missing user header")
		}
		if r.Header.Get("Authorization") == "secret" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true,"data":{"quota":123}}`))
	}))
	defer server.Close()
	var data struct {
		Quota int64 `json:"quota"`
	}
	if err := fetchSiteJSON(context.Background(), server.Client(), server.URL, "/api/user/self", "42", "secret", &data); err != nil {
		t.Fatal(err)
	}
	if data.Quota != 123 || strings.Join(headers, ",") != "secret,Bearer secret" {
		t.Fatalf("data=%+v headers=%v", data, headers)
	}
}

func TestSiteInfoTokenListForms(t *testing.T) {
	for _, raw := range []string{`[{"key":"abc"}]`, `{"items":[{"key":"abc"}]}`, `{"data":[{"key":"abc"}]}`} {
		tokens, err := parseSiteTokens(json.RawMessage(raw))
		if err != nil || len(tokens) != 1 || tokens[0].Key != "abc" {
			t.Fatalf("%s: %+v %v", raw, tokens, err)
		}
	}
}

func TestSiteInfoSanitizeError(t *testing.T) {
	secret := "sensitive-token"
	got := sanitizeSiteError(fmt.Errorf("failed %s %s", secret, strings.Repeat("x", 300)), secret)
	if strings.Contains(got, secret) || !strings.Contains(got, "[REDACTED]") || len([]rune(got)) > 240 {
		t.Fatalf("unsafe error: %q", got)
	}
}

func TestSiteInfoCurrency(t *testing.T) {
	for _, tt := range []struct {
		currency, symbol string
		rate, usd        float64
		wantSymbol       string
		want             float64
	}{
		{"USD", "¤", 2, 7, "$", 2}, {"CUSTOM", "€", 3, 7, "€", 6}, {"CUSTOM", "€", 0, 7, "$", 2}, {"CNY", "", 0, 7, "¥", 14},
	} {
		symbol, amount := siteInfoAmount(1000000, 0, tt.currency, tt.symbol, tt.rate, tt.usd)
		if symbol != tt.wantSymbol || amount != tt.want {
			t.Errorf("%+v: got %s %f", tt, symbol, amount)
		}
	}
}

func TestAnnouncementsAggregationAndFilters(t *testing.T) {
	rows := []announcementRow{
		{ID: 1, GroupID: 2, GroupName: "Beta", SiteRootURL: "https://a", DedupKey: "x", Content: "same", EventAtMS: 1, FetchedAtMS: 3},
		{ID: 2, GroupID: 1, GroupName: "Alpha", SiteRootURL: "https://a", DedupKey: "x", Content: "same", SiteName: "Site A", EventAtMS: 5, FetchedAtMS: 4},
		{ID: 3, GroupID: 3, GroupName: "Gamma", SiteRootURL: "https://b", DedupKey: "y", Content: "other", EventAtMS: 7},
	}
	got := aggregateAnnouncements(rows, "https://a", " BETA ")
	if len(got.Items) != 1 || len(got.Sites) != 2 || got.Sites[0].Count != 1 || got.Sites[1].Count != 1 {
		t.Fatalf("unexpected result: %+v", got)
	}
	item := got.Items[0]
	if item.ID != "https://a#x" || item.EventAtMS != 5 || item.FetchedAtMS != 4 || item.SiteName != "Site A" || len(item.Groups) != 2 || item.Groups[0].ID != 1 || item.Groups[1].ID != 2 {
		t.Fatalf("bad aggregate: %+v", item)
	}
	empty := aggregateAnnouncements(nil, "", "")
	if empty.Items == nil || empty.Sites == nil {
		t.Fatalf("nil arrays: %+v", empty)
	}
}

func TestAnnouncementsDatabaseJoin(t *testing.T) {
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-announcement-test")
	a := models.SiteAnnouncement{GroupID: id, SiteRootURL: "https://example.com", DedupKey: "abc", Content: "test", Source: "notice", EventAtMS: 1, FetchedAtMS: 2}
	if err := fixture.db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	result, err := fixture.service.GetAnnouncements(t.Context(), "", "")
	if err != nil || len(result.Items) != 1 || len(result.Items[0].Groups) != 1 || result.Items[0].Groups[0].ID != id {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestSiteInfoSettingsPersistence(t *testing.T) {
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-site-settings-test")
	var request GroupSettingsUpdateRequest
	if err := json.Unmarshal([]byte(`{"provider_url":"https://example.org/v1", "site_access":{"user_id":"42","access_token":"secret-site-token"}}`), &request); err != nil {
		t.Fatal(err)
	}
	got, err := fixture.service.UpdateGroupSettings(t.Context(), id, request)
	if err != nil {
		t.Fatal(err)
	}
	if !got.SiteAccess.TokenConfigured || got.SiteAccess.SiteRootURL != "https://example.org" {
		t.Fatalf("response=%+v", got.SiteAccess)
	}
	var cfg models.GroupSiteConfig
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.AccessToken == "secret-site-token" || cfg.AccessToken == "" || cfg.Status != "pending" {
		t.Fatalf("invalid config %+v", cfg)
	}
}

func TestSiteInfoCollectionProjection(t *testing.T) {
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-collection-site")
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: "encrypted", SiteRootURL: "https://example.org", Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	_, records, err := fixture.service.captureGroupCollectionRecords(t.Context(), false)
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range records {
		if record.ID == id {
			if record.SiteInfo == nil || record.SiteInfo.Status != "pending" {
				t.Fatalf("site info %+v", record.SiteInfo)
			}
			return
		}
	}
	t.Fatal("group not found")
}

func TestSiteInfoCreateIdempotent(t *testing.T) {
	fixture := newServiceFixture(t)
	req := GroupCreateRequest{Name: stringPointer("site-create"), ChannelID: channel.OpenAICompatible, Params: json.RawMessage(`{"base_url":"https://create.example/v1"}`), Models: optionalGroupModels{Set: true}, Credentials: "key"}
	if err := json.Unmarshal([]byte(`{"provider_url":"https://site.example/path","site_access":{"user_id":"42","access_token":"secret"}}`), &req); err != nil {
		t.Fatal(err)
	}
	key := "318f47a2-9c35-4d6e-8b1a-1234567890ab"
	result, err := fixture.service.CreateGroupIdempotent(t.Context(), key, req)
	if err != nil {
		t.Fatal(err)
	}
	var cfg models.GroupSiteConfig
	if err := fixture.db.First(&cfg, "group_id = ?", result.GroupID).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.UserID != "42" || cfg.SiteRootURL != "https://site.example" || cfg.AccessToken == "secret" || cfg.Status != "pending" {
		t.Fatalf("bad config: %+v", cfg)
	}
	if _, err := fixture.service.CreateGroupIdempotent(t.Context(), key, req); err != nil {
		t.Fatal(err)
	}
}

func TestSiteInfoSyncFetchAndFailurePreservesSnapshot(t *testing.T) {
	fail := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			_, _ = w.Write([]byte(`{"success":true,"data":{"system_name":"Example","quota_per_unit":500000,"announcements":[]}}`))
		case "/api/notice":
			_, _ = w.Write([]byte(`{"success":true,"data":"hello"}`))
		case "/api/user/self":
			if fail {
				w.WriteHeader(500)
				return
			}
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":1000000,"group":"default"}}`))
		case "/api/token/":
			_, _ = w.Write([]byte(`{"success":true,"data":[]}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-sync-credential")
	encrypted, err := fixture.encryption.Encrypt("site-secret")
	if err != nil {
		t.Fatal(err)
	}
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: encrypted, SiteRootURL: server.URL, Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	coordinator := NewSiteInfoSyncCoordinator(fixture.service, httpclient.NewHTTPClientManager())
	if err := coordinator.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.Status != "ok" || cfg.RemainingAmount == nil || *cfg.RemainingAmount != 2 || cfg.FetchedAtMS == nil {
		t.Fatalf("bad snapshot %+v", cfg)
	}
	fetched := *cfg.FetchedAtMS
	fail = true
	if err := coordinator.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.Status != "error" || cfg.FetchedAtMS == nil || *cfg.FetchedAtMS != fetched || cfg.RemainingAmount == nil || *cfg.RemainingAmount != 2 {
		t.Fatalf("failed snapshot %+v", cfg)
	}
}

func TestSiteInfoSettingsRootConfirmationAndClear(t *testing.T) {
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-root-check")
	update := func(raw string) error {
		var request GroupSettingsUpdateRequest
		if err := json.Unmarshal([]byte(raw), &request); err != nil {
			t.Fatal(err)
		}
		_, err := fixture.service.UpdateGroupSettings(t.Context(), id, request)
		return err
	}
	if err := update(`{"provider_url":"https://a.example", "site_access":{"user_id":"42","access_token":"secret"}}`); err != nil {
		t.Fatal(err)
	}
	if err := update(`{"provider_url":"https://b.example"}`); err == nil {
		t.Fatal("root change accepted without confirmation")
	}
	if err := update(`{"provider_url":"https://b.example","site_access":{"confirm_site_root_change":true}}`); err != nil {
		t.Fatal(err)
	}
	var cfg models.GroupSiteConfig
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.SiteRootURL != "https://b.example" || cfg.ConfigVersion != 2 {
		t.Fatalf("bad version %+v", cfg)
	}
	a := models.SiteAnnouncement{GroupID: id, SiteRootURL: cfg.SiteRootURL, DedupKey: "n", Content: "n", EventAtMS: 1, FetchedAtMS: 1}
	if err := fixture.db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	if err := update(`{"provider_url":null,"site_access":{"clear_access_token":true}}`); err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	var n int64
	fixture.db.Model(&models.SiteAnnouncement{}).Where("group_id = ?", id).Count(&n)
	if cfg.AccessToken != "" || cfg.Status != "unconfigured" || cfg.ConfigVersion != 3 || n != 0 {
		t.Fatalf("clear %+v announcements=%d", cfg, n)
	}
}

func TestSiteInfoSyncStaleFailureDoesNotOverwrite(t *testing.T) {
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-stale-site")
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 2, UserID: "42", AccessToken: "cipher", SiteRootURL: "https://a", Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	coordinator := NewSiteInfoSyncCoordinator(fixture.service, httpclient.NewHTTPClientManager())
	stale := cfg
	stale.ConfigVersion = 1
	if err := coordinator.storeSiteFailure(t.Context(), stale, "bad"); err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.Status != "pending" || cfg.LastError != "" {
		t.Fatalf("stale write %+v", cfg)
	}
}

func TestSiteInfoValidationErrorCodes(t *testing.T) {
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-error-code")
	var req GroupSettingsUpdateRequest
	if err := json.Unmarshal([]byte(`{"site_access":{"access_token":"secret"}}`), &req); err != nil {
		t.Fatal(err)
	}
	_, err := fixture.service.UpdateGroupSettings(t.Context(), id, req)
	assertAPIErrorCode(t, err, "INVALID_GROUP_SETTINGS")
	create := GroupCreateRequest{Name: stringPointer("bad-site"), ChannelID: channel.OpenAICompatible, Params: json.RawMessage(`{"base_url":"https://bad.example/v1"}`), Models: optionalGroupModels{Set: true}, Credentials: "key"}
	if err := json.Unmarshal([]byte(`{"site_access":{"access_token":"secret"}}`), &create); err != nil {
		t.Fatal(err)
	}
	_, err = fixture.service.CreateGroup(t.Context(), create)
	assertAPIErrorCode(t, err, "INVALID_GROUP_CREATE")
}

func TestCopyGroupSiteInfoNotCopied(t *testing.T) {
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-copy-site")
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: "cipher", SiteRootURL: "https://a", Status: "ok", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	a := models.SiteAnnouncement{GroupID: id, SiteRootURL: "https://a", DedupKey: "1", Content: "notice", EventAtMS: 1, FetchedAtMS: 1}
	if err := fixture.db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	result, err := fixture.service.CopyGroupIdempotent(t.Context(), "918f47a2-9c35-4d6e-8b1a-1234567890ab", id)
	if err != nil {
		t.Fatal(err)
	}
	var n int64
	fixture.db.Model(&models.GroupSiteConfig{}).Where("group_id = ?", result.GroupID).Count(&n)
	if n != 0 {
		t.Fatal("copied config")
	}
	fixture.db.Model(&models.SiteAnnouncement{}).Where("group_id = ?", result.GroupID).Count(&n)
	if n != 0 {
		t.Fatal("copied announcements")
	}
}

func TestSiteInfoSyncStaleSuccessAndSourceRetention(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	credential := "sk-sync-credential-secret"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.String(), credential) {
			t.Error("credential in URL")
		}
		switch r.URL.Path {
		case "/api/status":
			w.WriteHeader(500)
		case "/api/notice":
			_, _ = w.Write([]byte(`{"success":true,"data":"new notice"}`))
		case "/api/user/self":
			close(entered)
			<-release
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":10}}`))
		case "/api/token/":
			_, _ = w.Write([]byte(`{"success":true,"data":[]}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, credential)
	enc, err := fixture.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	old := models.SiteAnnouncement{GroupID: id, SiteRootURL: server.URL, DedupKey: "old", Content: "old", Source: "announcement", EventAtMS: 1, FetchedAtMS: 1}
	if err := fixture.db.Create(&old).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(fixture.service, httpclient.NewHTTPClientManager())
	done := make(chan error, 1)
	go func() { done <- c.syncGroup(t.Context(), id) }()
	<-entered
	if err := fixture.db.Model(&models.GroupSiteConfig{}).Where("group_id = ?", id).Updates(map[string]any{"config_version": 2, "status": "pending"}).Error; err != nil {
		t.Fatal(err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	var count int64
	fixture.db.Model(&models.SiteAnnouncement{}).Where("group_id = ?", id).Count(&count)
	if cfg.Status != "pending" || cfg.ConfigVersion != 2 || count != 1 {
		t.Fatalf("stale success wrote cfg=%+v count=%d", cfg, count)
	}
}

func TestSiteInfoSyncOptionalSourceRetentionAndRatio(t *testing.T) {
	apiKey := "sk-abcdefghijklmno"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.String(), apiKey) {
			t.Error("API key leaked to URL")
		}
		switch r.URL.Path {
		case "/api/status":
			w.WriteHeader(500)
		case "/api/notice":
			_, _ = w.Write([]byte(`{"success":true,"data":""}`))
		case "/api/user/self":
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":1000000,"group":"default"}}`))
		case "/api/token/":
			_, _ = w.Write([]byte(`{"success":true,"data":[{"key":"abcd****lmno","group":"vip","status":1}]}`))
		case "/api/user/self/groups":
			_, _ = w.Write([]byte(`{"success":true,"data":{"vip":{"ratio":0.75}}}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, apiKey)
	enc, err := fixture.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	for _, a := range []models.SiteAnnouncement{{GroupID: id, SiteRootURL: server.URL, DedupKey: "old-a", Content: "old", Source: "announcement", EventAtMS: 1, FetchedAtMS: 1}, {GroupID: id, SiteRootURL: server.URL, DedupKey: "old-n", Content: "old", Source: "notice", EventAtMS: 1, FetchedAtMS: 1}} {
		if err := fixture.db.Create(&a).Error; err != nil {
			t.Fatal(err)
		}
	}
	c := NewSiteInfoSyncCoordinator(fixture.service, httpclient.NewHTTPClientManager())
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.TokenGroup != "vip" || cfg.EffectiveGroup != "vip" || cfg.GroupRatio == nil || *cfg.GroupRatio != 0.75 {
		t.Fatalf("bad group snapshot %+v", cfg)
	}
	var announcements []models.SiteAnnouncement
	if err := fixture.db.Where("group_id = ?", id).Find(&announcements).Error; err != nil {
		t.Fatal(err)
	}
	if len(announcements) != 1 || announcements[0].Source != "announcement" {
		t.Fatalf("bad source retention %+v", announcements)
	}
}

func TestSiteInfoSettingsErrorsHaveClientMessage(t *testing.T) {
	initControlI18n(t)
	for _, tc := range []struct {
		operation string
		code      string
		err       *app_errors.APIError
	}{
		{"update_group_settings", "SITE_ROOT_CHANGE_CONFIRMATION_REQUIRED", siteRootChangeConfirmationRequired},
		{"update_group_settings", "INVALID_GROUP_SETTINGS", siteInvalidGroupSettings},
		{"create_group", "INVALID_GROUP_CREATE", siteInvalidGroupCreate},
		{"update_group_settings", "site_root_url_change_confirmation_required", &app_errors.APIError{HTTPStatus: http.StatusBadRequest, Code: "site_root_url_change_confirmation_required"}},
		{"update_group_settings", "site_access_provider_url_required", &app_errors.APIError{HTTPStatus: http.StatusBadRequest, Code: "site_access_provider_url_required"}},
		{"create_group", "site_access_incomplete", &app_errors.APIError{HTTPStatus: http.StatusBadRequest, Code: "site_access_incomplete"}},
	} {
		t.Run(tc.code, func(t *testing.T) {
			if got := serviceErrorMessageID(tc.operation, tc.err, tc.err); got != "bad_request" {
				t.Errorf("message ID=%s", got)
			}
			w := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(w)
			ctx.Request = httptest.NewRequest(http.MethodPost, "/", nil)
			wantMessage := i18n.Message(ctx, "bad_request")
			writeServiceError(ctx, tc.operation, tc.err)
			var body struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if w.Code != http.StatusBadRequest || body.Code != tc.code || body.Message != wantMessage {
				t.Fatalf("status=%d body=%+v want code=%s message=%q", w.Code, body, tc.code, wantMessage)
			}
		})
	}
}

func TestSiteInfoHTTPSettingsAndAnnouncements(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-http-site")
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	path := "/api/groups/" + stringGroupID(id) + "/settings"
	got := serveGroupSettingsRequest(t, engine, http.MethodPut, path, "test-auth-key", `{"provider_url":"https://old.example/v1","site_access":{"user_id":"42","access_token":"secret-token-value"}}`)
	if got.Code != http.StatusOK || strings.Contains(got.Body.String(), "secret-token-value") {
		t.Fatalf("update status=%d body=%s", got.Code, got.Body)
	}
	got = serveGroupSettingsRequest(t, engine, http.MethodPut, path, "test-auth-key", `{"provider_url":"https://new.example"}`)
	var body struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(got.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if got.Code != http.StatusBadRequest || body.Code != "SITE_ROOT_CHANGE_CONFIRMATION_REQUIRED" || body.Message == "" {
		t.Fatalf("error status=%d body=%+v", got.Code, body)
	}
	ann := models.SiteAnnouncement{GroupID: id, SiteRootURL: "https://old.example", SiteName: "Old", DedupKey: "n", Content: "hello", Source: "notice", EventAtMS: 1, FetchedAtMS: 1}
	if err := fixture.db.Create(&ann).Error; err != nil {
		t.Fatal(err)
	}
	got = serveGroupSettingsRequest(t, engine, http.MethodGet, "/api/announcements?site=https%3A%2F%2Fold.example&q=hello", "test-auth-key", "")
	if got.Code != http.StatusOK || !strings.Contains(got.Body.String(), "hello") || !strings.Contains(got.Body.String(), "site_root_url") {
		t.Fatalf("announcements status=%d body=%s", got.Code, got.Body)
	}
}

func TestSiteInfoDeleteRemovesRows(t *testing.T) {
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-delete-site")
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: "cipher", SiteRootURL: "https://a", Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	a := models.SiteAnnouncement{GroupID: id, SiteRootURL: "https://a", DedupKey: "n", Content: "n", EventAtMS: 1, FetchedAtMS: 1}
	if err := fixture.db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	if err := fixture.service.DeleteGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	var count int64
	fixture.db.Model(&models.GroupSiteConfig{}).Where("group_id = ?", id).Count(&count)
	if count != 0 {
		t.Fatal("config not deleted")
	}
	fixture.db.Model(&models.SiteAnnouncement{}).Where("group_id = ?", id).Count(&count)
	if count != 0 {
		t.Fatal("announcement not deleted")
	}
}

func TestSiteInfoScanRefreshIntervalAndPending(t *testing.T) {
	var selfCalls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/user/self" {
			selfCalls++
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":42}}`))
			return
		}
		w.WriteHeader(404)
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-scan-site")
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	fixture.service.now = func() time.Time { return now }
	enc, err := fixture.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	last := now.Add(-59 * time.Minute).UnixMilli()
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "ok", LastAttemptAtMS: &last, UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(fixture.service, httpclient.NewHTTPClientManager())
	c.scan(t.Context())
	c.workers.Wait()
	if selfCalls != 0 {
		t.Fatal("refreshed too early")
	}
	now = now.Add(time.Minute)
	c.scan(t.Context())
	c.workers.Wait()
	if selfCalls != 1 {
		t.Fatalf("due calls=%d", selfCalls)
	}
	if err := fixture.db.Model(&models.GroupSiteConfig{}).Where("group_id = ?", id).Update("status", "pending").Error; err != nil {
		t.Fatal(err)
	}
	c.scan(t.Context())
	c.workers.Wait()
	if selfCalls != 2 {
		t.Fatalf("pending calls=%d", selfCalls)
	}
}

func TestAnnouncementOrderingAndEmptyFilter(t *testing.T) {
	rows := []announcementRow{{ID: 1, GroupID: 1, SiteRootURL: "https://z", DedupKey: "z", EventAtMS: 2}, {ID: 2, GroupID: 2, SiteRootURL: "https://a", DedupKey: "b", EventAtMS: 2}, {ID: 3, GroupID: 3, SiteRootURL: "https://a", DedupKey: "a", EventAtMS: 3}}
	result := aggregateAnnouncements(rows, "", "")
	if len(result.Items) != 3 || result.Items[0].ID != "https://a#a" || result.Items[1].ID != "https://a#b" || result.Items[2].ID != "https://z#z" {
		t.Fatalf("bad order %+v", result.Items)
	}
	result = aggregateAnnouncements(rows, "missing", "")
	if len(result.Items) != 0 || len(result.Sites) != 2 {
		t.Fatalf("filter lost sites %+v", result)
	}
}

func TestSiteInfoRatioNullDoesNotFallBack(t *testing.T) {
	key := "sk-abcdefghijklmno"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			_, _ = w.Write([]byte(`{"success":true,"data":{}}`))
		case "/api/notice":
			_, _ = w.Write([]byte(`{"success":true,"data":""}`))
		case "/api/user/self":
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":1}}`))
		case "/api/token/":
			_, _ = w.Write([]byte(`{"success":true,"data":[{"key":"abcd***lmno","group":"vip"}]}`))
		case "/api/user/self/groups":
			_, _ = w.Write([]byte(`{"success":true,"data":{"vip":null}}`))
		case "/api/user/groups", "/api/pricing":
			t.Errorf("anonymous ratio endpoint called: %s", r.URL.Path)
			_, _ = w.Write([]byte(`{"success":true,"data":{"vip":0.5}}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, key)
	enc, err := fixture.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(fixture.service, httpclient.NewHTTPClientManager())
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.GroupRatio != nil {
		t.Fatalf("ratio=%v", cfg.GroupRatio)
	}
}

func TestSiteInfoCreateDigestOmissionCompatibility(t *testing.T) {
	body := groupCreateDigestBody{}
	encoded, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "site_access") {
		t.Fatalf("legacy digest changed: %s", encoded)
	}
	body.SiteAccess = &GroupCreateSiteAccessInput{UserID: "42", AccessToken: "secret"}
	encoded, err = json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"site_access"`) {
		t.Fatalf("configured digest missing: %s", encoded)
	}
}

func TestSiteInfoCoordinatorRequestGroupSync(t *testing.T) {
	calls := make(chan struct{}, 3)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/user/self" {
			calls <- struct{}{}
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":1}}`))
			return
		}
		w.WriteHeader(404)
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-wake-site")
	enc, err := fixture.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(fixture.service, httpclient.NewHTTPClientManager())
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()
	defer func() { cancel(); <-done }()
	select {
	case <-calls:
	case <-time.After(5 * time.Second):
		t.Fatal("initial scan did not fetch")
	}
	// Wait until the first writeback is complete before marking a new pending version.
	deadline := time.Now().Add(5 * time.Second)
	for {
		var current models.GroupSiteConfig
		if err := fixture.db.First(&current, "group_id = ?", id).Error; err != nil {
			t.Fatal(err)
		}
		if current.Status == "ok" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("initial scan did not commit")
		}
		time.Sleep(time.Millisecond)
	}
	if err := fixture.db.Model(&models.GroupSiteConfig{}).Where("group_id = ?", id).Updates(map[string]any{"status": "pending", "config_version": 2}).Error; err != nil {
		t.Fatal(err)
	}
	c.RequestGroupSync(id)
	select {
	case <-calls:
	case <-time.After(5 * time.Second):
		t.Fatal("request did not wake coordinator")
	}
}

func TestSiteInfoAnnouncementEventTimesStable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			_, _ = w.Write([]byte(`{"success":true,"data":{"system_name":"site","announcements":[{"content":"undated","publishDate":"bad"},{"content":"dated","publishDate":"2026-01-02T03:04:05Z"}]}}`))
		case "/api/notice":
			_, _ = w.Write([]byte(`{"success":true,"data":" notice "}`))
		case "/api/user/self":
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":1}}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-event-site")
	enc, err := fixture.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 2, 1, 0, 0, 0, 0, time.UTC)
	fixture.service.now = func() time.Time { return now }
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending", UpdatedAtMS: 1}
	if err := fixture.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(fixture.service, httpclient.NewHTTPClientManager())
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	var rows []models.SiteAnnouncement
	if err := fixture.db.Where("group_id = ?", id).Find(&rows).Error; err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 {
		t.Fatalf("rows=%+v", rows)
	}
	times := map[string]int64{}
	for _, row := range rows {
		times[row.Content] = row.EventAtMS
	}
	dated, _ := time.Parse(time.RFC3339, "2026-01-02T03:04:05Z")
	if times["dated"] != dated.UnixMilli() || times["undated"] != now.UnixMilli() || times["notice"] != now.UnixMilli() {
		t.Fatalf("times=%v", times)
	}
	now = now.Add(time.Hour)
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	rows = nil
	if err := fixture.db.Where("group_id = ?", id).Find(&rows).Error; err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if row.EventAtMS != times[row.Content] {
			t.Fatalf("unstable event %+v", row)
		}
	}
}
