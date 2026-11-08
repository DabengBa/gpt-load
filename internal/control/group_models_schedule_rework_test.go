package control

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

func TestGroupSettingsChannelSwitchRejectsGeminiNoneReasoning(t *testing.T) {
	for _, target := range []channel.ID{channel.Gemini, channel.GoogleVertex} {
		t.Run(string(target), func(t *testing.T) {
			s := newScheduleTestScenario(t)
			path := "/api/groups/1/models?include_schedule=true"
			body := fmt.Sprintf(`{"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","reasoning_effort":"none"}]}`, scheduleEntryOneA)
			decodeScheduleSuccess(t, s.perform(http.MethodPut, path, body, s.authKey), new(map[string]any))
			if target == channel.GoogleVertex {
				// No credentials avoids an unrelated OpenAI/Vertex credential-schema rejection.
				if err := s.fixture.db.Where("group_id = ?", 1).Delete(&models.Credential{}).Error; err != nil {
					t.Fatal(err)
				}
			}
			before := captureGroupRow(t, s.fixture, 1)
			revision := s.fixture.manager.Current().Revision
			settings := fmt.Sprintf(`{"channel_id":%q,"params":{}}`, target)
			response := s.perform(http.MethodPut, "/api/groups/1/settings", settings, s.authKey)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("switch to %s: %d %s, want 400", target, response.Code, response.Body.String())
			}
			after := captureGroupRow(t, s.fixture, 1)
			if after.ChannelID != before.ChannelID || string(after.Models) != string(before.Models) || s.fixture.manager.Current().Revision != revision {
				t.Fatalf("rejected switch changed persisted or runtime state")
			}
		})
	}
}

func TestGroupCombinedUpdateIsAtomic(t *testing.T) {
	s := newScheduleTestScenario(t)
	path := "/api/groups/1/models?include_schedule=true"
	body := fmt.Sprintf(`{"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","reasoning_effort":"none"}]}`, scheduleEntryOneA)
	decodeScheduleSuccess(t, s.perform(http.MethodPut, path, body, s.authKey), new(map[string]any))
	combined := fmt.Sprintf(`{"settings":{"channel_id":"gemini","params":{}},"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","reasoning_effort":null}]}`, scheduleEntryOneA)
	response := s.perform(http.MethodPut, "/api/groups/1/configuration", combined, s.authKey)
	if response.Code != http.StatusOK {
		t.Fatalf("combined transition: %d %s", response.Code, response.Body.String())
	}
	group := captureGroupRow(t, s.fixture, 1)
	if group.ChannelID != "gemini" || loadCreatedGroupModels(t, s.fixture, 1)[0].ReasoningEffort != "" {
		t.Fatal("combined transition not committed")
	}
	before := captureGroupRow(t, s.fixture, 1)
	invalid := fmt.Sprintf(`{"settings":{"name":"should-not-save"},"models":[{"id":"up-a","alias":"pub","alias_enabled":true,"entry_id":"%s","reasoning_effort":"none"}]}`, scheduleEntryOneA)
	response = s.perform(http.MethodPut, "/api/groups/1/configuration", invalid, s.authKey)
	if response.Code == http.StatusOK {
		t.Fatal("invalid combined update succeeded")
	}
	after := captureGroupRow(t, s.fixture, 1)
	if after.Name != before.Name || string(after.Models) != string(before.Models) {
		t.Fatal("invalid combined update partially committed")
	}
}

type failingScheduleReconciler struct {
	delegate state.SnapshotReconciler
	fail     bool
	calls    int
}

func (r *failingScheduleReconciler) ReconcileConfigSnapshot(snapshot *state.ConfigSnapshot) error {
	r.calls++
	if r.fail {
		return errors.New("forced snapshot reconciliation failure")
	}
	return r.delegate.ReconcileConfigSnapshot(snapshot)
}

func TestGroupModelsRuntimeRecoversScheduleFields(t *testing.T) {
	for _, tc := range []struct {
		name, fields, effort string
		enabled              bool
	}{
		{"enabled", `"enabled":false`, "", false},
		{"reasoning", `"reasoning_effort":"high"`, "high", true},
		{"both", `"enabled":false,"reasoning_effort":"high"`, "high", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newScheduleTestScenario(t)
			r := &failingScheduleReconciler{delegate: controlAccessQuotaReconciler{runtime: s.fixture.accessQuota}, fail: true}
			s.fixture.manager.SetSnapshotReconciler(r)
			revision := s.fixture.manager.Current().Revision
			patch := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s",%s}]}`, revision, scheduleEntryOneA, tc.fields)
			response := s.perform(http.MethodPatch, "/api/model-route/schedule", patch, s.authKey)
			if response.Code == http.StatusOK || r.calls < 2 {
				t.Fatalf("expected publication and recovery failures: %d calls=%d %s", response.Code, r.calls, response.Body.String())
			}
			stored := loadCreatedGroupModels(t, s.fixture, 1)[0]
			if (stored.Enabled == nil || *stored.Enabled) != tc.enabled || stored.ReasoningEffort != tc.effort {
				t.Fatalf("DB did not commit: %#v", stored)
			}
			if s.fixture.manager.Current().Revision != revision {
				t.Fatal("failed publication changed snapshot")
			}
			r.fail = false
			response = s.perform(http.MethodGet, "/api/groups/1/models?include_schedule=true", "", s.authKey)
			if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"reasoning_effort":`) {
				t.Fatalf("GET: %d %s", response.Code, response.Body.String())
			}
			model := s.fixture.manager.Current().GroupCatalog[1].Models[0]
			if (model.Enabled == nil || *model.Enabled) != tc.enabled || model.ReasoningEffort != tc.effort {
				t.Fatalf("snapshot not recovered: %#v", model)
			}
		})
	}
}
