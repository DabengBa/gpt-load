package control

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/platform/httpclient"
	"gpt-load/internal/storage/models"
)

func TestSiteInfoCurrencyAndAnnouncementSync(t *testing.T) {
	cases := []struct {
		name, status, symbol string
		amount               *float64
	}{
		{"official CNY", `{"quota_display_type":"CNY","quota_per_unit":500000,"usd_exchange_rate":7}`, "¥", floatPtr(14)},
		{"official CUSTOM", `{"quota_display_type":"CUSTOM","quota_per_unit":500000,"custom_currency_symbol":"€","custom_currency_exchange_rate":3}`, "€", floatPtr(6)},
		{"legacy currency_type", `{"currency_type":"CNY","quota_per_unit":500000,"usd_exchange_rate":7}`, "¥", floatPtr(14)},
		{"official takes precedence", `{"quota_display_type":"CNY","currency_type":"CUSTOM","quota_per_unit":500000,"usd_exchange_rate":7,"custom_currency_symbol":"€","custom_currency_exchange_rate":3}`, "¥", floatPtr(14)},
		{"invalid exchange rate", `{"quota_display_type":"CNY","quota_per_unit":500000,"usd_exchange_rate":"bad"}`, "", nil},
		{"invalid system name", `{"system_name":{"bad":true},"quota_per_unit":500000}`, "$", floatPtr(2)},
		{"invalid custom rate", `{"quota_display_type":"CUSTOM","custom_currency_symbol":"€","custom_currency_exchange_rate":"bad"}`, "", nil},
		{"invalid custom symbol", `{"quota_display_type":"CUSTOM","custom_currency_symbol":{},"custom_currency_exchange_rate":3}`, "", nil},
		{"invalid display type", `{"quota_display_type":{},"currency_type":"CNY","usd_exchange_rate":7}`, "", nil},
		{"unused invalid currency fields", `{"quota_display_type":"USD","usd_exchange_rate":"bad","custom_currency_symbol":{},"custom_currency_exchange_rate":[]}`, "$", floatPtr(2)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/user/self":
					fmt.Fprint(w, `{"success":true,"data":{"quota":1000000}}`)
				case "/api/status":
					fmt.Fprintf(w, `{"success":true,"data":%s}`, tc.status[:len(tc.status)-1]+`,"announcements":[{"content":"new"}]}`)
				case "/api/notice":
					fmt.Fprint(w, `{"success":true,"data":""}`)
				default:
					w.WriteHeader(404)
				}
			}))
			defer server.Close()
			f := newServiceFixture(t)
			id := createGroupWithCredentials(t, f, "sk-currency-sync")
			token, err := f.encryption.Encrypt("token")
			if err != nil {
				t.Fatal(err)
			}
			if err := f.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: token, SiteRootURL: server.URL, Status: "pending"}).Error; err != nil {
				t.Fatal(err)
			}
			if err := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager()).syncGroup(t.Context(), id); err != nil {
				t.Fatal(err)
			}
			var cfg models.GroupSiteConfig
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
				t.Fatal(err)
			}
			if cfg.CurrencySymbol != tc.symbol || (tc.amount == nil && cfg.RemainingAmount != nil) || (tc.amount != nil && (cfg.RemainingAmount == nil || *cfg.RemainingAmount != *tc.amount)) {
				t.Fatalf("currency: symbol=%q amount=%v expected=%q %v", cfg.CurrencySymbol, cfg.RemainingAmount, tc.symbol, tc.amount)
			}
			access, err := f.service.readSiteAccess(f.db, id)
			if err != nil || access.Snapshot == nil || access.Snapshot.CurrencySymbol != tc.symbol || (tc.amount == nil && access.Snapshot.RemainingAmount != nil) || (tc.amount != nil && (access.Snapshot.RemainingAmount == nil || *access.Snapshot.RemainingAmount != *tc.amount)) {
				t.Fatalf("projected snapshot: %+v err=%v", access.Snapshot, err)
			}
			feed, err := f.service.GetAnnouncements(t.Context(), "", "")
			if err != nil {
				t.Fatal(err)
			}
			if len(feed.Items) != 1 || feed.Items[0].Content != "new" || len(feed.Sources) != 1 || feed.Sources[0].Announcement.Status != "ok" {
				t.Fatalf("announcement lost: %+v", feed)
			}
		})
	}
}

func floatPtr(v float64) *float64 { return &v }
