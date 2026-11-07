package control

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
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

func TestSiteInfoNullTokenSecondPagePreservesRatio(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/user/self":
			fmt.Fprint(w, `{"success":true,"data":{"quota":2,"group":"default"}}`)
		case "/api/status":
			fmt.Fprint(w, `{"success":true,"data":{}}`)
		case "/api/notice":
			fmt.Fprint(w, `{"success":true,"data":""}`)
		case "/api/token/":
			if r.URL.Query().Get("p") == "2" {
				fmt.Fprint(w, `{"success":true,"data":{"items":null}}`)
				return
			}
			items := make([]map[string]any, 100)
			for i := range items {
				items[i] = map[string]any{"key": fmt.Sprintf("other%d", i), "group": "default"}
			}
			items[0] = map[string]any{"key": "abcd****lmno", "group": "new"}
			_ = json.NewEncoder(w).Encode(map[string]any{"success": true, "data": map[string]any{"items": items}})
		case "/api/user/self/groups":
			fmt.Fprint(w, `{"success":true,"data":{"new":2}}`)
		}
	}))
	defer server.Close()
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-abcdefghijklmno")
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	ratio := 0.75
	if err := f.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, TokenGroup: "old", EffectiveGroup: "old", GroupRatio: &ratio}).Error; err != nil {
		t.Fatal(err)
	}
	if err := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager()).syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	var cfg models.GroupSiteConfig
	if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.Status != "error" || cfg.LastError == "" || cfg.TokenGroup != "old" || cfg.EffectiveGroup != "old" || cfg.GroupRatio == nil || *cfg.GroupRatio != ratio {
		t.Fatalf("invalid page lost prior ratio: %+v", cfg)
	}
}

func TestSiteInfoOptionalRatioFailurePreservesSnapshot(t *testing.T) {
	for _, failing := range []string{"/api/token/", "/api/user/self/groups", "malformed-ratio"} {
		t.Run(failing, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == failing {
					w.WriteHeader(503)
					return
				}
				switch r.URL.Path {
				case "/api/user/self":
					fmt.Fprint(w, `{"success":true,"data":{"quota":2,"group":"default"}}`)
				case "/api/status":
					fmt.Fprint(w, `{"success":true,"data":{}}`)
				case "/api/notice":
					fmt.Fprint(w, `{"success":true,"data":""}`)
				case "/api/token/":
					fmt.Fprint(w, `{"success":true,"data":[{"key":"abcd****lmno","group":"vip"}]}`)
				case "/api/user/self/groups":
					fmt.Fprint(w, `{"success":true,"data":{"vip":null}}`)
				default:
					w.WriteHeader(404)
				}
			}))
			defer server.Close()
			f := newServiceFixture(t)
			id := createGroupWithCredentials(t, f, "sk-abcdefghijklmno")
			enc, err := f.encryption.Encrypt("secret-site-token")
			if err != nil {
				t.Fatal(err)
			}
			ratio := 0.75
			cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "ok", TokenGroup: "vip", EffectiveGroup: "vip", GroupRatio: &ratio}
			if err := f.db.Create(&cfg).Error; err != nil {
				t.Fatal(err)
			}
			if err := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager()).syncGroup(t.Context(), id); err != nil {
				t.Fatal(err)
			}
			cfg = models.GroupSiteConfig{}
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
				t.Fatal(err)
			}
			if cfg.Status != "error" || cfg.LastError == "" || strings.Contains(cfg.LastError, "secret-site-token") || cfg.TokenGroup != "vip" || cfg.EffectiveGroup != "vip" || cfg.GroupRatio == nil || *cfg.GroupRatio != ratio {
				t.Fatalf("optional failure lost snapshot or error: %+v", cfg)
			}
		})
	}
}

func TestSiteInfoSourceErrorsAreIndependentAndRecover(t *testing.T) {
	statusCode, noticeCode, coreCode := 503, 429, 200
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		code, data := 200, `{}`
		switch r.URL.Path {
		case "/api/user/self":
			code, data = coreCode, `{"quota":1}`
		case "/api/status":
			code = statusCode
		case "/api/notice":
			code, data = noticeCode, `"notice"`
		case "/api/token/":
			data = `[]`
		}
		if code != 200 {
			w.WriteHeader(code)
			fmt.Fprint(w, "secret-site-token")
			return
		}
		fmt.Fprintf(w, `{"success":true,"data":%s}`, data)
	}))
	defer server.Close()
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-abcdefghijklmno")
	enc, err := f.encryption.Encrypt("secret-site-token")
	if err != nil {
		t.Fatal(err)
	}
	if err := f.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
	check := func(aStatus, aError, nStatus, nError string) {
		t.Helper()
		feed, err := f.service.GetAnnouncements(t.Context(), "", "")
		if err != nil {
			t.Fatal(err)
		}
		if len(feed.Sources) != 1 {
			t.Fatalf("sources: %+v", feed.Sources)
		}
		a, n := feed.Sources[0].Announcement, feed.Sources[0].Notice
		if a.Status != aStatus || a.LastError != aError || n.Status != nStatus || n.LastError != nError {
			t.Fatalf("source errors: announcement=%+v notice=%+v", a, n)
		}
	}
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	check("error", "site returned HTTP 503", "error", "site returned HTTP 429")
	statusCode = 200
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	check("ok", "", "error", "site returned HTTP 429")
	coreCode = 502
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	check("error", "site returned HTTP 502", "error", "site returned HTTP 502")
	coreCode, noticeCode = 200, 200
	if err := c.syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	check("ok", "", "ok", "")
}

func TestSiteInfoTokenPaginationFindsLaterPage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/user/self":
			fmt.Fprint(w, `{"success":true,"data":{"quota":1000000,"group":"default"}}`)
		case "/api/status":
			fmt.Fprint(w, `{"success":true,"data":{}}`)
		case "/api/notice":
			fmt.Fprint(w, `{"success":true,"data":""}`)
		case "/api/token/":
			items := make([]map[string]any, 0, 100)
			if r.URL.Query().Get("p") == "1" {
				for i := 0; i < 100; i++ {
					items = append(items, map[string]any{"key": fmt.Sprintf("unrelated%d", i), "group": "default", "status": 1})
				}
			} else if r.URL.Query().Get("p") == "2" {
				items = append(items, map[string]any{"key": "abcd****lmno", "group": "vip", "status": 1})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"success": true, "data": map[string]any{"items": items}})
		case "/api/user/self/groups":
			fmt.Fprint(w, `{"success":true,"data":{"vip":0.75}}`)
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-abcdefghijklmno")
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	if err := f.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}).Error; err != nil {
		t.Fatal(err)
	}
	if err := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager()).syncGroup(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	var cfg models.GroupSiteConfig
	if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.TokenGroup != "vip" || cfg.GroupRatio == nil || *cfg.GroupRatio != 0.75 {
		t.Fatalf("later-page token not resolved: %+v", cfg)
	}
}

func TestSiteInfoOptionalTimeoutPreservesCoreAndOtherSources(t *testing.T) {
	for _, blocked := range []string{"/api/status", "/api/notice", "/api/token/"} {
		t.Run(blocked, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == blocked {
					<-r.Context().Done()
					return
				}
				switch r.URL.Path {
				case "/api/user/self":
					fmt.Fprint(w, `{"success":true,"data":{"quota":1000000}}`)
				case "/api/status":
					fmt.Fprint(w, `{"success":true,"data":{"announcements":[{"content":"new announcement"}]}}`)
				case "/api/notice":
					fmt.Fprint(w, `{"success":true,"data":"new notice"}`)
				case "/api/token/":
					fmt.Fprint(w, `{"success":true,"data":[]}`)
				default:
					w.WriteHeader(404)
				}
			}))
			defer server.Close()
			f := newServiceFixture(t)
			id := createGroupWithCredentials(t, f, "sk-abcdefghijklmno")
			enc, err := f.encryption.Encrypt("token")
			if err != nil {
				t.Fatal(err)
			}
			cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}
			if err := f.db.Create(&cfg).Error; err != nil {
				t.Fatal(err)
			}
			for _, source := range []string{"announcement", "notice"} {
				a := models.SiteAnnouncement{GroupID: id, SiteRootURL: server.URL, DedupKey: "old-" + source, Content: "old", Source: source}
				if err := f.db.Create(&a).Error; err != nil {
					t.Fatal(err)
				}
			}
			c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
			c.timeout = 300 * time.Millisecond
			if err := c.syncGroup(context.Background(), id); err != nil {
				t.Fatal(err)
			}
			cfg = models.GroupSiteConfig{}
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
				t.Fatal(err)
			}
			wantStatus := "ok"
			if blocked == "/api/status" || blocked == "/api/token/" {
				wantStatus = "error"
			}
			if cfg.Status != wantStatus || cfg.BalanceQuota == nil || *cfg.BalanceQuota != 1000000 {
				t.Fatalf("balance lost: %+v", cfg)
			}
			var rows []models.SiteAnnouncement
			if err := f.db.Where("group_id = ?", id).Find(&rows).Error; err != nil {
				t.Fatal(err)
			}
			got := map[string]string{}
			for _, row := range rows {
				got[row.Source] = row.Content
			}
			wantA, wantN := "new announcement", "new notice"
			if blocked == "/api/status" {
				wantA = "old"
			}
			if blocked == "/api/notice" {
				wantN = "old"
			}
			if got["announcement"] != wantA || got["notice"] != wantN {
				t.Fatalf("partial results: %v", got)
			}
		})
	}
}

func TestSiteInfoStatusFailureDoesNotPublishDefaultConvertedBalance(t *testing.T) {
	initControlI18n(t)
	for _, failure := range []string{"503", "timeout"} {
		for _, previous := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/previous=%v", failure, previous), func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					switch r.URL.Path {
					case "/api/status":
						if failure == "timeout" {
							<-r.Context().Done()
							return
						}
						w.WriteHeader(http.StatusServiceUnavailable)
					case "/api/user/self":
						fmt.Fprint(w, `{"success":true,"data":{"quota":1000000}}`)
					case "/api/notice":
						fmt.Fprint(w, `{"success":true,"data":"updated notice"}`)
					default:
						w.WriteHeader(http.StatusNotFound)
					}
				}))
				defer server.Close()
				f := newServiceFixture(t)
				id := createGroupWithCredentials(t, f, "sk-status-failure")
				enc, err := f.encryption.Encrypt("token")
				if err != nil {
					t.Fatal(err)
				}
				cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}
				if previous {
					quotaPerUnit, amount, fetched := 100000.0, 7.0, int64(10)
					cfg.QuotaPerUnit, cfg.RemainingAmount, cfg.FetchedAtMS = &quotaPerUnit, &amount, &fetched
					cfg.CurrencySymbol, cfg.SiteName = "€", "Existing site"
				}
				if err := f.db.Create(&cfg).Error; err != nil {
					t.Fatal(err)
				}
				c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
				c.timeout = 400 * time.Millisecond
				if err := c.syncGroup(context.Background(), id); err != nil {
					t.Fatal(err)
				}
				// Reload into a zero value so NULL columns are not masked by the fixture.
				cfg = models.GroupSiteConfig{}
				if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
					t.Fatal(err)
				}
				if cfg.Status != "error" || cfg.BalanceQuota == nil || *cfg.BalanceQuota != 1000000 {
					t.Fatalf("core result lost: %+v", cfg)
				}
				if previous {
					if cfg.RemainingAmount == nil || *cfg.RemainingAmount != 7 || cfg.QuotaPerUnit == nil || *cfg.QuotaPerUnit != 100000 || cfg.CurrencySymbol != "€" || cfg.Status != "error" {
						t.Fatalf("failed status lost prior converted amount or failure status: %+v", cfg)
					}
				} else if cfg.RemainingAmount != nil || cfg.QuotaPerUnit != nil {
					t.Fatalf("failed status produced a fresh converted amount: %+v", cfg)
				}
				engine := gin.New()
				NewServer(&config.Config{AuthKey: "test-auth-key"}, f.service).RegisterRoutes(engine)
				w := performGroupCollectionRequest(engine, "/api/groups?include=site_info", "Bearer test-auth-key")
				if w.Code != http.StatusOK || !json.Valid(w.Body.Bytes()) || (!previous && !strings.Contains(w.Body.String(), `"remaining_amount":null`)) {
					t.Fatalf("groups did not show an unknown balance: status=%d body=%q", w.Code, w.Body.String())
				}
				if previous && cfg.SiteName != "Existing site" {
					t.Fatalf("lost known site name: %+v", cfg)
				}
				var notice models.SiteAnnouncement
				if err := f.db.Where("group_id = ? AND source = ?", id, "notice").Take(&notice).Error; err != nil || notice.Content != "updated notice" {
					t.Fatalf("notice lost: %+v err=%v", notice, err)
				}
			})
		}
	}
}

func TestSiteInfoSaturatedScanPrioritizesNeverAndOldestAttempt(t *testing.T) {
	started := make(chan string, 20)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/api/user/self") {
			started <- strings.Split(r.URL.Path, "/")[1]
			fmt.Fprint(w, `{"success":true,"data":{"quota":1}}`)
			return
		}
		w.WriteHeader(404)
	}))
	defer server.Close()
	f := newServiceFixture(t)
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 12; i++ {
		group := createGroupCollectionGroup(t, f, fmt.Sprintf("fair-%d", i), true, nil)
		last := int64(1000 + (10-i)*100)
		cfg := models.GroupSiteConfig{GroupID: group.ID, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: fmt.Sprintf("%s/%d", server.URL, i), Status: "pending", LastAttemptAtMS: &last}
		if i == 11 {
			cfg.LastAttemptAtMS = nil
		}
		if err := f.db.Create(&cfg).Error; err != nil {
			t.Fatal(err)
		}
	}
	c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
	for i := 0; i < siteInfoMaxConcurrentSyncs-1; i++ {
		c.active[uint(10000+i)] = struct{}{}
	}
	for _, want := range []string{"11", "10", "9"} {
		c.scan(t.Context())
		c.workers.Wait()
		select {
		case got := <-started:
			if got != want {
				t.Fatalf("got group %s, want %s", got, want)
			}
		default:
			t.Fatal("no group started")
		}
	}
}

func TestSiteInfoTimedOutBacklogDoesNotStarveNewGroup(t *testing.T) {
	started := make(chan string, 32)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/api/user/self") {
			started <- strings.Split(r.URL.Path, "/")[1]
			<-r.Context().Done()
			return
		}
		w.WriteHeader(404)
	}))
	defer server.Close()
	f := newServiceFixture(t)
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	add := func(i int, last *int64) {
		group := createGroupCollectionGroup(t, f, fmt.Sprintf("backlog-%d", i), true, nil)
		cfg := models.GroupSiteConfig{GroupID: group.ID, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: fmt.Sprintf("%s/%d", server.URL, i), Status: "pending", LastAttemptAtMS: last}
		if err := f.db.Create(&cfg).Error; err != nil {
			t.Fatal(err)
		}
	}
	old := int64(1)
	for i := 0; i < 16; i++ {
		add(i, &old)
	}
	c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
	c.timeout = time.Second
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()
	defer func() { cancel(); <-done }()
	for i := 0; i < siteInfoMaxConcurrentSyncs; i++ {
		select {
		case <-started:
		case <-time.After(3 * time.Second):
			t.Fatal("backlog did not saturate workers")
		}
	}
	add(16, nil)
	c.RequestGroupSync(0)
	found := false
	for i := 0; i < siteInfoMaxConcurrentSyncs; i++ {
		select {
		case got := <-started:
			found = found || got == "16"
		case <-time.After(3 * time.Second):
			t.Fatal("next wave did not start after timeouts")
		}
	}
	if !found {
		t.Fatal("new group starved behind previously attempted backlog")
	}
}

func TestSiteInfoNonFiniteAmountsDoNotBreakGroupsAPI(t *testing.T) {
	initControlI18n(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprint(w, `{"success":true,"data":{"quota_per_unit":1e-320}}`)
		case "/api/user/self":
			fmt.Fprint(w, `{"success":true,"data":{"quota":1000000}}`)
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-nonfinite-test")
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	oldQuota, oldAmount := 100000.0, 7.0
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending", QuotaPerUnit: &oldQuota, RemainingAmount: &oldAmount, CurrencySymbol: "€"}
	if err := f.db.Create(&cfg).Error; err != nil {
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
	if cfg.Status != "error" || cfg.LastError == "" || cfg.RemainingAmount == nil || *cfg.RemainingAmount != oldAmount || cfg.QuotaPerUnit == nil || *cfg.QuotaPerUnit != oldQuota || cfg.CurrencySymbol != "€" {
		t.Fatalf("nonfinite conversion lost previous snapshot: %+v", cfg)
	}
	if cfg.RemainingAmount != nil && (math.IsInf(*cfg.RemainingAmount, 0) || math.IsNaN(*cfg.RemainingAmount)) {
		t.Fatalf("stored nonfinite amount: %v", *cfg.RemainingAmount)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, f.service).RegisterRoutes(engine)
	check := func(wantNull bool) {
		w := performGroupCollectionRequest(engine, "/api/groups?include=site_info", "Bearer test-auth-key")
		if w.Code != http.StatusOK || !json.Valid(w.Body.Bytes()) {
			t.Fatalf("groups status=%d body=%q", w.Code, w.Body.String())
		}
		if wantNull && (!strings.Contains(w.Body.String(), `"remaining_amount":null`) || !strings.Contains(w.Body.String(), `"group_ratio":null`)) {
			t.Fatalf("nonfinite values not suppressed: %s", w.Body.String())
		}
	}
	check(false)
	if err := f.db.Model(&models.GroupSiteConfig{}).Where("group_id = ?", id).Updates(map[string]any{"remaining_amount": math.Inf(1), "group_ratio": math.Inf(1)}).Error; err != nil {
		t.Fatal(err)
	}
	check(true)
	w := serveGroupSettingsRequest(t, engine, http.MethodGet, fmt.Sprintf("/api/groups/%d/settings?include=site_info", id), "test-auth-key", "")
	if w.Code != http.StatusOK || !json.Valid(w.Body.Bytes()) {
		t.Fatalf("settings status=%d body=%q", w.Code, w.Body.String())
	}
}
