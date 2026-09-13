package control

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
)

func TestGroupSettingsHTTPPersistsResponsesReasoningStatusFilter(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	groupID := createGroupForCredentialImport(t, fixture, "sk-settings-reasoning-status-http")
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	path := "/api/groups/" + stringGroupID(groupID) + "/settings"

	initial, err := fixture.service.GetGroupSettings(t.Context(), groupID)
	if err != nil {
		t.Fatal(err)
	}
	if initial.Effective.ResponsesReasoningStatusFilterEnabled {
		t.Fatal("new group reasoning status filter = true, want false")
	}

	updated := serveGroupSettingsRequest(t, engine, http.MethodPut, path, "test-auth-key", `{"overrides":{"responses_reasoning_status_filter_enabled":true}}`)
	if updated.Code != http.StatusOK {
		t.Fatalf("PUT settings = %d %s, want 200", updated.Code, updated.Body.String())
	}
	var updateEnvelope struct {
		Data GroupSettingsResponse `json:"data"`
	}
	if err := json.Unmarshal(updated.Body.Bytes(), &updateEnvelope); err != nil {
		t.Fatal(err)
	}
	if !updateEnvelope.Data.Effective.ResponsesReasoningStatusFilterEnabled ||
		updateEnvelope.Data.Overrides[state.SettingResponsesReasoningStatusFilterEnabled] != true {
		t.Fatalf("PUT settings data = %#v", updateEnvelope.Data)
	}

	read := serveGroupSettingsRequest(t, engine, http.MethodGet, path, "test-auth-key", "")
	if read.Code != http.StatusOK {
		t.Fatalf("GET settings = %d %s, want 200", read.Code, read.Body.String())
	}
	var readEnvelope struct {
		Data GroupSettingsResponse `json:"data"`
	}
	if err := json.Unmarshal(read.Body.Bytes(), &readEnvelope); err != nil {
		t.Fatal(err)
	}
	if !readEnvelope.Data.Effective.ResponsesReasoningStatusFilterEnabled {
		t.Fatalf("GET settings data = %#v", readEnvelope.Data)
	}

	reloadedManager := state.NewManager()
	if err := stateloader.New(fixture.db, reloadedManager, state.NewCredentialRegistry()).Load(t.Context()); err != nil {
		t.Fatalf("reload runtime settings: %v", err)
	}
	if !reloadedManager.Current().Groups[groupID].ResponsesReasoningStatusFilterEnabled {
		t.Fatal("reloaded group reasoning status filter = false, want true")
	}
}
