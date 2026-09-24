package control

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func TestSQLiteMaintenanceStatusReportsEffectiveRetentionWithoutChangingHealthWire(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	check := func(wantDays int, wantSource string) {
		t.Helper()
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, "/api/sqlite-maintenance", nil)
		request.Header.Set("Authorization", "Bearer test-auth-key")
		engine.ServeHTTP(response, request)
		if response.Code != http.StatusOK {
			t.Fatalf("GET /api/sqlite-maintenance status = %d, body = %s", response.Code, response.Body.String())
		}
		var body struct {
			Data sqliteMaintenanceResponse `json:"data"`
		}
		if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		got := body.Data
		if got.RetentionDays != wantDays || got.RetentionSource != wantSource || got.SweepIntervalSeconds != 3600 || got.SQLite.MaintenanceMode != "offline_only" || got.SQLite.JournalMode != "memory" || got.SQLite.Error != "" {
			t.Fatalf("maintenance status = %+v, want %d days from %s, hourly sweep and memory SQLite", got, wantDays, wantSource)
		}
	}
	if got := state.DefaultRuntimeSettings().RequestLogRetentionDays; got != 7 {
		t.Fatalf("default retention = %d, want 7", got)
	}
	check(7, "default")
	_, err := fixture.service.UpdateSettings(t.Context(), SettingsUpdateRequest{Settings: map[string]json.RawMessage{state.SettingRequestLogRetentionDays: json.RawMessage(`30`)}})
	if err != nil {
		t.Fatalf("UpdateSettings() error = %v", err)
	}
	check(30, "system_setting")
	unauthenticated := httptest.NewRecorder()
	engine.ServeHTTP(unauthenticated, httptest.NewRequest(http.MethodGet, "/api/sqlite-maintenance", nil))
	if unauthenticated.Code == http.StatusOK {
		t.Fatal("maintenance status accessible without authentication")
	}
}
