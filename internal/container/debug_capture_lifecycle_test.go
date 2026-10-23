//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package container

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"gpt-load/internal/debugcapture"
	"gpt-load/internal/gateway"
	"gpt-load/internal/httplifecycle"
	"gpt-load/internal/platform/i18n"
	"gpt-load/internal/storage"
)

func TestBuildContainerCapturesSuccessfulDataPlaneRequest(t *testing.T) {
	t.Setenv("AUTH_KEY", "test-auth-key")
	t.Setenv("DATA_DIR", t.TempDir())
	t.Setenv("DATABASE_DSN", ":memory:")
	t.Setenv("ENCRYPTION_KEY", "test-master-key-long")
	t.Setenv("DEBUG_CAPTURE_ENABLED", "true")
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}

	dependencyContainer, err := BuildContainer()
	if err != nil {
		t.Fatalf("BuildContainer() error = %v", err)
	}
	var handler *gateway.Handler
	var lifecycle *httplifecycle.Coordinator
	var store *debugcapture.Store
	var captureRuntime *debugcapture.Runtime
	if err := dependencyContainer.Invoke(func(
		resolvedHandler *gateway.Handler,
		resolvedLifecycle *httplifecycle.Coordinator,
		resolvedStore *debugcapture.Store,
		resolvedCaptureRuntime *debugcapture.Runtime,
		db *gorm.DB,
	) {
		handler = resolvedHandler
		lifecycle = resolvedLifecycle
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

	engine := gin.New()
	engine.Use(lifecycle.TrackAll(), handler.CaptureMiddleware())
	engine.POST("/v1/responses", handler.HTTPModule().BeforeAuth[0], func(c *gin.Context) {
		requestBody, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.Status(http.StatusBadRequest)
			return
		}
		if !bytes.Equal(requestBody, []byte(`{"input":"capture-me"}`)) {
			c.Status(http.StatusBadRequest)
			return
		}
		c.Data(http.StatusOK, "application/json", []byte(`{"ok":true}`))
	})

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewBufferString(`{"input":"capture-me"}`))
	engine.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("response status = %d, body=%q", response.Code, response.Body.String())
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
		t.Fatalf("captured sessions = %#v, err = %v", records, err)
	}
	record := records[0]
	if record.State != debugcapture.StateCompleted {
		t.Fatalf("capture state = %q, error = %q, want completed", record.State, record.Error)
	}
	if len(record.Attempts) != 1 {
		t.Fatalf("capture attempts = %#v, want client attempt only", record.Attempts)
	}
	requestBody, err := store.ReadPart(record.ID, record.Attempts[0].ID, debugcapture.PartBody, debugcapture.DirectionRequest)
	if err != nil {
		t.Fatalf("read captured request body: %v", err)
	}
	if !bytes.Equal(requestBody, []byte(`{"input":"capture-me"}`)) {
		t.Fatalf("captured request body = %q", requestBody)
	}
	responseBody, err := store.ReadPart(record.ID, record.Attempts[0].ID, debugcapture.PartBody, debugcapture.DirectionResponse)
	if err != nil {
		t.Fatalf("read captured response body: %v", err)
	}
	if !bytes.Equal(responseBody, response.Body.Bytes()) {
		t.Fatalf("captured response body = %q, want %q", responseBody, response.Body.Bytes())
	}
}
