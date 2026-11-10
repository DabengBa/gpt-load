package control

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/httpclient"
	"gpt-load/internal/storage/models"
)

func TestSiteInfoSlowGroupDoesNotBlockHealthyGroup(t *testing.T) {
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	var slowRequests atomic.Int32
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		slowRequests.Add(1)
		select {
		case entered <- struct{}{}:
		default:
		}
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer slow.Close()
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			_, _ = w.Write([]byte(`{"success":true,"data":{}}`))
		case "/api/notice":
			_, _ = w.Write([]byte(`{"success":true,"data":""}`))
		case "/api/user/self":
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":42}}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer healthy.Close()
	f := newServiceFixture(t)
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	for i, url := range []string{slow.URL, healthy.URL} {
		name := fmt.Sprintf("sync-group-%d", i)
		group, err := f.service.CreateGroup(t.Context(), GroupCreateRequest{Name: &name, ChannelID: channel.OpenAICompatible, Params: json.RawMessage(fmt.Sprintf(`{"base_url":"https://site%d.example/v1"}`, i)), Models: optionalGroupModels{Set: true}, Credentials: "sk-test"})
		if err != nil {
			t.Fatal(err)
		}
		id := group.GroupID
		cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: url, Status: "pending"}
		if err := f.db.Create(&cfg).Error; err != nil {
			t.Fatal(err)
		}
	}
	c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()
	defer func() { cancel(); close(release); <-done }()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("slow group never started")
	}
	deadline := time.After(3 * time.Second)
	for {
		var cfg models.GroupSiteConfig
		if err := f.db.Where("site_root_url = ?", healthy.URL).First(&cfg).Error; err != nil {
			t.Fatal(err)
		}
		if cfg.Status == "ok" && cfg.BalanceQuota != nil && *cfg.BalanceQuota == 42 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("healthy group blocked by slow group")
		case <-time.After(10 * time.Millisecond):
		}
	}
	for i := 0; i < 5; i++ {
		c.RequestGroupSync(0)
		time.Sleep(20 * time.Millisecond)
	}
	if got := slowRequests.Load(); got != 1 {
		t.Fatalf("slow group has %d simultaneous requests; want 1", got)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("coordinator did not stop after cancellation")
	}
}

func TestSiteInfoSyncTotalTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-timeout-group")
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}
	if err := f.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
	c.timeout = 100 * time.Millisecond
	start := time.Now()
	_ = c.syncGroup(context.Background(), id)
	if time.Since(start) > 2*time.Second {
		t.Fatal("sync exceeded total timeout")
	}
	if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.Status != "error" || cfg.LastError != "site sync timed out" || cfg.LastAttemptAtMS == nil {
		t.Fatalf("timeout not recorded: %+v", cfg)
	}
}
