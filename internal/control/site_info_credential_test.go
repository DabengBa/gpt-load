package control

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/platform/httpclient"
	"gpt-load/internal/storage/models"
)

func TestSiteInfoRatioFailureNeverUsesAnonymousEndpoint(t *testing.T) {
	for _, status := range []int{401, 500} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/user/self":
					_, _ = w.Write([]byte(`{"success":true,"data":{"quota":1}}`))
				case "/api/token/":
					_, _ = w.Write([]byte(`{"success":true,"data":[{"key":"abcd***lmno","group":"vip"}]}`))
				case "/api/user/self/groups":
					if r.Header.Get("Authorization") == "" || r.Header.Get("New-Api-User") != "42" {
						t.Error("missing authentication")
					}
					w.WriteHeader(status)
				case "/api/user/groups", "/api/pricing":
					t.Errorf("anonymous ratio endpoint called: %s", r.URL.Path)
					_, _ = w.Write([]byte(`{"success":true,"data":{"vip":0.5}}`))
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
			c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
			if err := c.syncGroup(t.Context(), id); err != nil {
				t.Fatal(err)
			}
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
				t.Fatal(err)
			}
			if cfg.GroupRatio != nil {
				t.Fatalf("unexpected ratio %v", *cfg.GroupRatio)
			}
		})
	}
}

func TestSiteInfoCredentialMutationInvalidates(t *testing.T) {
	for _, action := range []string{"update", "delete", "import", "idempotent-import"} {
		t.Run(action, func(t *testing.T) {
			f := newServiceFixture(t)
			id := createGroupWithCredentials(t, f, "sk-abcdefghijklmno")
			var cred models.Credential
			if err := f.db.Where("group_id = ?", id).First(&cred).Error; err != nil {
				t.Fatal(err)
			}
			if action == "import" || action == "idempotent-import" {
				if err := f.service.DeleteGroupCredential(t.Context(), id, cred.ID); err != nil {
					t.Fatal(err)
				}
			}
			ratio := 0.5
			cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: "cipher", SiteRootURL: "https://example.com", Status: "ok", TokenGroup: "vip", EffectiveGroup: "vip", GroupRatio: &ratio}
			if err := f.db.Create(&cfg).Error; err != nil {
				t.Fatal(err)
			}
			c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
			var err error
			switch action {
			case "update":
				_, err = f.service.UpdateGroupCredential(t.Context(), id, cred.ID, CredentialUpdateRequest{Credentials: optionalField[string]{Set: true, Value: "sk-newcredential"}})
			case "delete":
				err = f.service.DeleteGroupCredential(t.Context(), id, cred.ID)
			case "import":
				_, err = f.service.ImportGroupCredentials(t.Context(), id, CredentialImportRequest{Credentials: "sk-newcredential"})
			case "idempotent-import":
				_, err = f.service.ImportGroupCredentialsIdempotent(t.Context(), "12345678-1234-4123-8123-123456789abc", id, CredentialImportRequest{Credentials: "sk-newcredential"})
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
				t.Fatal(err)
			}
			if cfg.ConfigVersion != 2 || cfg.Status != "pending" || cfg.GroupRatio != nil || cfg.EffectiveGroup != "" || cfg.TokenGroup != "" {
				t.Fatalf("stale snapshot: %+v", cfg)
			}
			select {
			case <-c.wake:
			default:
				t.Fatal("sync not requested")
			}
		})
	}
}

func TestSiteInfoCredentialRaceDiscardsOldSync(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/token/":
			close(entered)
			<-release
			_, _ = w.Write([]byte(`{"success":true,"data":[{"key":"abcd***lmno","group":"vip"}]}`))
		case "/api/user/self":
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":1}}`))
		case "/api/user/self/groups":
			_, _ = w.Write([]byte(`{"success":true,"data":{"vip":0.5}}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-abcdefghijklmno")
	var cred models.Credential
	if err := f.db.Where("group_id = ?", id).First(&cred).Error; err != nil {
		t.Fatal(err)
	}
	enc, err := f.encryption.Encrypt("token")
	if err != nil {
		t.Fatal(err)
	}
	cfg := models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: enc, SiteRootURL: server.URL, Status: "pending"}
	if err := f.db.Create(&cfg).Error; err != nil {
		t.Fatal(err)
	}
	c := NewSiteInfoSyncCoordinator(f.service, httpclient.NewHTTPClientManager())
	done := make(chan error, 1)
	go func() { done <- c.syncGroup(t.Context(), id) }()
	<-entered
	_, err = f.service.UpdateGroupCredential(t.Context(), id, cred.ID, CredentialUpdateRequest{Credentials: optionalField[string]{Set: true, Value: "sk-newcredential"}})
	close(release)
	if err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := f.db.First(&cfg, "group_id = ?", id).Error; err != nil {
		t.Fatal(err)
	}
	if cfg.ConfigVersion != 2 || cfg.Status != "pending" || cfg.GroupRatio != nil {
		t.Fatalf("old sync committed: %+v", cfg)
	}
}
