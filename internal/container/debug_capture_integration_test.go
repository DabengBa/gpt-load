//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package container

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/debugcapture"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/platform/i18n"
	"gpt-load/internal/state"
	"gpt-load/internal/storage"
)

func TestBuildContainerCapturesAllowedPreflightAtOutermostBoundary(t *testing.T) {
	t.Setenv("AUTH_KEY", "test-auth-key")
	t.Setenv("DATA_DIR", t.TempDir())
	t.Setenv("DATABASE_DSN", ":memory:")
	t.Setenv("ENCRYPTION_KEY", "test-master-key-long")
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}

	dependencyContainer, err := BuildContainer()
	if err != nil {
		t.Fatalf("BuildContainer() error = %v", err)
	}
	var engine *gin.Engine
	var manager *state.Manager
	var store *debugcapture.Store
	var captureRuntime *debugcapture.Runtime
	if err := dependencyContainer.Invoke(func(
		resolvedEngine *gin.Engine,
		resolvedManager *state.Manager,
		resolvedStore *debugcapture.Store,
		resolvedCaptureRuntime *debugcapture.Runtime,
		db *gorm.DB,
	) {
		engine = resolvedEngine
		manager = resolvedManager
		store = resolvedStore
		captureRuntime = resolvedCaptureRuntime
		if err := storage.AutoMigrate(db); err != nil {
			t.Fatalf("AutoMigrate() error = %v", err)
		}
		sqlDB, dbErr := db.DB()
		if dbErr == nil {
			t.Cleanup(func() { _ = sqlDB.Close() })
		}
		if err := captureRuntime.Start(); err != nil {
			t.Fatalf("start capture runtime: %v", err)
		}
		t.Cleanup(func() { _ = captureRuntime.Stop(context.Background()) })
	}); err != nil {
		t.Fatalf("resolve container dependencies: %v", err)
	}
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		SystemSettings: config.Settings{
			state.SettingCORS: map[string]any{
				"enabled":         true,
				"allowed_origins": []any{"app://obsidian.md"},
				"allowed_methods": []any{"POST"},
				"allowed_headers": []any{"Authorization"},
			},
		},
	}); err != nil {
		t.Fatalf("publish CORS settings: %v", err)
	}

	request := httptest.NewRequest(http.MethodOptions, "/v1/responses", nil)
	request.Header.Set("Origin", "app://obsidian.md")
	request.Header.Set("Authorization", "Bearer preflight-secret")
	request.Header.Set("Access-Control-Request-Method", "POST")
	request.Header.Set("Access-Control-Request-Headers", "Authorization")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	if response.Code != http.StatusNoContent {
		t.Fatalf("preflight status = %d %s", response.Code, response.Body.String())
	}

	var records []debugcapture.SessionRecord
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		records, err = store.QuerySessions(debugcapture.SessionQuery{Limit: 10})
		if err == nil && len(records) == 1 && records[0].State != debugcapture.StateActive {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(records) != 1 {
		t.Fatalf("captured preflight sessions = %#v, err = %v", records, err)
	}
	record := records[0]
	if record.State != debugcapture.StateCompleted {
		t.Fatalf("preflight capture state = %q, error = %q", record.State, record.Error)
	}
	if len(record.Attempts) == 0 {
		t.Fatal("preflight capture has no client attempt")
	}
	headers, err := store.ReadPart(record.ID, record.Attempts[0].ID, debugcapture.PartHeaders, debugcapture.DirectionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(headers, []byte("Authorization: Bearer preflight-secret")) {
		t.Fatalf("preflight headers = %q", headers)
	}
}
