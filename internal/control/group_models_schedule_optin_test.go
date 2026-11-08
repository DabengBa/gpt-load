package control

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"gpt-load/internal/channel"
)

// Integration: the group model and scheduling endpoints share database state.
func TestGroupModelsScheduleOptIn(t *testing.T) {
	s := newScheduleTestScenario(t)
	path := "/api/groups/1/models?include_schedule=true"
	get := s.perform(http.MethodGet, path, "", s.authKey)
	if get.Code != http.StatusOK || !strings.Contains(get.Body.String(), `"reasoning_effort":`) || !strings.Contains(get.Body.String(), `"enabled":true`) {
		t.Fatalf("opt-in GET: %d %s", get.Code, get.Body.String())
	}

	body := fmt.Sprintf(`{"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","reasoning_effort":"high","enabled":false}]}`, scheduleEntryOneA)
	put := s.perform(http.MethodPut, path, body, s.authKey)
	if put.Code != http.StatusOK || !strings.Contains(put.Body.String(), `"enabled":false`) {
		t.Fatalf("opt-in PUT: %d %s", put.Code, put.Body.String())
	}
	detail := s.perform(http.MethodGet, "/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions", "", s.authKey)
	var result modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, detail, &result)
	if result.Groups[0].Entries[0].EntryID != scheduleEntryOneA || result.Groups[0].Entries[0].Enabled || result.Groups[0].Entries[0].Reasoning.Configured == nil || *result.Groups[0].Entries[0].Reasoning.Configured != "high" {
		t.Fatalf("schedule: %#v", result.Groups[0].Entries[0])
	}
	legacy := s.perform(http.MethodGet, "/api/groups/1/models", "", s.authKey)
	if strings.Contains(legacy.Body.String(), `"reasoning_effort"`) || strings.Contains(legacy.Body.String(), `"enabled"`) {
		t.Fatalf("legacy GET leaked fields: %s", legacy.Body.String())
	}
	// Old clients' enabled false/null are ignored and omitted schedule fields
	// retain the latest transaction state.
	for _, enabled := range []string{"false", "null"} {
		body = fmt.Sprintf(`{"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","enabled":%s}]}`, scheduleEntryOneA, enabled)
		put = s.perform(http.MethodPut, "/api/groups/1/models", body, s.authKey)
		if put.Code != http.StatusOK || strings.Contains(put.Body.String(), `"enabled"`) {
			t.Fatalf("legacy PUT: %d %s", put.Code, put.Body.String())
		}
		stored := loadCreatedGroupModels(t, s.fixture, 1)[0]
		if stored.Enabled == nil || *stored.Enabled || stored.ReasoningEffort != "high" {
			t.Fatalf("legacy changed schedule: %#v", stored)
		}
	}
	body = fmt.Sprintf(`{"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","reasoning_effort":null}]}`, scheduleEntryOneA)
	put = s.perform(http.MethodPut, path, body, s.authKey)
	if put.Code != http.StatusOK || !strings.Contains(put.Body.String(), `"reasoning_effort":""`) || !strings.Contains(put.Body.String(), `"enabled":false`) {
		t.Fatalf("clear: %d %s", put.Code, put.Body.String())
	}
	body = fmt.Sprintf(`{"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","enabled":true},{"id":"new-model","alias_enabled":false}]}`, scheduleEntryOneA)
	put = s.perform(http.MethodPut, path, body, s.authKey)
	if put.Code != http.StatusOK || strings.Count(put.Body.String(), `"enabled":true`) != 2 || strings.Count(put.Body.String(), `"reasoning_effort":""`) != 2 {
		t.Fatalf("enable and default new row: %d %s", put.Code, put.Body.String())
	}
	before := loadStoredGroupModelsJSON(t, s.fixture, 1)
	batch := fmt.Sprintf(`{"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","enabled":false},{"id":"new-model","alias_enabled":false,"reasoning_effort":"invalid"}]}`, scheduleEntryOneA)
	if rejected := s.perform(http.MethodPut, path, batch, s.authKey); rejected.Code != http.StatusBadRequest || loadStoredGroupModelsJSON(t, s.fixture, 1) != before {
		t.Fatalf("batch rejection: %d %s", rejected.Code, rejected.Body.String())
	}
	for _, field := range []string{`"enabled":null`, `"reasoning_effort":"invalid"`, `"reasoning_effort":""`} {
		body = fmt.Sprintf(`{"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s",%s}]}`, scheduleEntryOneA, field)
		put = s.perform(http.MethodPut, path, body, s.authKey)
		if put.Code != http.StatusBadRequest {
			t.Fatalf("invalid %s: %d %s", field, put.Code, put.Body.String())
		}
		if after := loadStoredGroupModelsJSON(t, s.fixture, 1); after != before {
			t.Fatalf("invalid request wrote %s", after)
		}
	}
}

func TestGroupModelsScheduleOptInGeminiRejectsNone(t *testing.T) {
	for _, tc := range []struct {
		id                  channel.ID
		params, credentials string
	}{
		{channel.Gemini, `{}`, `gemini-key`},
		{channel.GoogleVertex, `{"location":"us-central1"}`, `{"service_account_json":"{\"type\":\"service_account\",\"project_id\":\"project-one\",\"client_email\":\"svc@example.iam.gserviceaccount.com\",\"private_key\":\"private-secret\"}"}`},
	} {
		t.Run(string(tc.id), func(t *testing.T) {
			s := newScheduleTestScenario(t)
			created, err := s.fixture.service.CreateGroup(t.Context(), GroupCreateRequest{ChannelID: tc.id, Params: json.RawMessage(tc.params), Credentials: tc.credentials, Models: optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gemini-model"}}}})
			if err != nil {
				t.Fatal(err)
			}
			before := loadStoredGroupModelsJSON(t, s.fixture, created.GroupID)
			path := fmt.Sprintf("/api/groups/%d/models?include_schedule=true", created.GroupID)
			put := s.perform(http.MethodPut, path, `{"models":[{"id":"gemini-model","alias_enabled":false,"reasoning_effort":"none"}]}`, s.authKey)
			if put.Code != http.StatusBadRequest || loadStoredGroupModelsJSON(t, s.fixture, created.GroupID) != before {
				t.Fatalf("none: %d %s", put.Code, put.Body.String())
			}
		})
	}
}

func TestGroupModelsScheduleIdentityFallback(t *testing.T) {
	no := false
	previous := []groupModelEntry{{ID: "single", EntryID: "e000000000001", ReasoningEffort: "high", Enabled: &no}, {ID: "multi", Alias: "a", EntryID: "e000000000002", ReasoningEffort: "low"}, {ID: "multi", Alias: "b", EntryID: "e000000000003", ReasoningEffort: "high"}}
	for _, tc := range []struct {
		name               string
		row                GroupModel
		wantID, wantEffort string
	}{
		{"same without id", GroupModel{ID: "single"}, "e000000000001", "high"},
		{"alias change fallback", GroupModel{ID: "single", Alias: "new"}, "e000000000001", "high"},
		{"multiple aliases no guess", GroupModel{ID: "multi", Alias: "new"}, "", ""},
		{"explicit id alias change", GroupModel{ID: "multi", Alias: "new", EntryID: "e000000000002"}, "e000000000002", "low"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := preserveGroupModelFields(previous, []GroupModel{tc.row})
			if err != nil {
				t.Fatal(err)
			}
			if (tc.wantID != "" && got[0].EntryID != tc.wantID) || (tc.wantID == "" && (got[0].EntryID == previous[1].EntryID || got[0].EntryID == previous[2].EntryID)) || got[0].ReasoningEffort != tc.wantEffort {
				t.Fatalf("row = %#v", got[0])
			}
		})
	}
}

func TestGroupModelsScheduleConcurrentFieldPreservation(t *testing.T) {
	s := newScheduleTestScenario(t)
	path := "/api/groups/1/models?include_schedule=true"
	// The details editor has loaded before a scheduling edit happens.
	decodeScheduleSuccess(t, s.perform(http.MethodGet, path, "", s.authKey), new(map[string]any))
	patch := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","enabled":false,"reasoning_effort":"high"}]}`, s.fixture.manager.Current().Revision, scheduleEntryOneA)
	decodeScheduleSuccess(t, s.perform(http.MethodPatch, "/api/model-route/schedule", patch, s.authKey), new(modelRouteSchedulePatchResponse))
	body := fmt.Sprintf(`{"models":[{"id":"up-a","alias":"renamed","alias_enabled":true,"entry_id":"%s"}]}`, scheduleEntryOneA)
	put := s.perform(http.MethodPut, path, body, s.authKey)
	if put.Code != http.StatusOK || !strings.Contains(put.Body.String(), `"enabled":false`) || !strings.Contains(put.Body.String(), `"reasoning_effort":"high"`) {
		t.Fatalf("concurrent save: %d %s", put.Code, put.Body.String())
	}
}
