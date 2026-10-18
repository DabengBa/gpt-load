package control

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
)

func TestModelRouteScheduleConcurrencyLimitPatch(t *testing.T) {
	scenario := newScheduleTestScenario(t)
	detailPath := fmt.Sprintf(
		"/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=%d",
		scenario.accessKeyID,
	)
	detailLimit := func(stage string) int {
		t.Helper()
		var detail modelRouteScheduleDetailResponse
		decodeScheduleSuccess(t,
			scenario.perform(http.MethodGet, detailPath, "", scenario.authKey), &detail)
		for _, entry := range detail.Groups[0].Entries {
			if entry.EntryID == scheduleEntryOneA {
				return entry.ConcurrencyLimit
			}
		}
		t.Fatalf("%s detail has no entry %s", stage, scheduleEntryOneA)
		return 0
	}

	revision := scenario.revision
	if got := detailLimit("initial"); got != 0 {
		t.Fatalf("initial concurrency_limit = %d, want 0", got)
	}
	set := fmt.Sprintf(
		`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","concurrency_limit":5}]}`,
		revision, scheduleEntryOneA,
	)
	var patched modelRouteSchedulePatchResponse
	decodeScheduleSuccess(t,
		scenario.perform(http.MethodPatch, "/api/model-route/schedule", set, scenario.authKey), &patched)
	if got := loadCreatedGroupModels(t, scenario.fixture, 1)[0].ConcurrencyLimit; got != 5 {
		t.Fatalf("stored concurrency_limit = %d, want 5", got)
	}
	if got := detailLimit("after set"); got != 5 {
		t.Fatalf("detail concurrency_limit after set = %d, want 5", got)
	}

	unrelated := fmt.Sprintf(
		`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","weight":40}]}`,
		patched.SnapshotRevisionNew, scheduleEntryOneA,
	)
	decodeScheduleSuccess(t,
		scenario.perform(http.MethodPatch, "/api/model-route/schedule", unrelated, scenario.authKey), &patched)
	if got := loadCreatedGroupModels(t, scenario.fixture, 1)[0].ConcurrencyLimit; got != 5 {
		t.Fatalf("concurrency_limit after unrelated patch = %d, want preserved 5", got)
	}

	before := loadStoredGroupModelsJSON(t, scenario.fixture, 1)
	for _, body := range []string{
		`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"` + scheduleEntryOneA + `","concurrency_limit":null}]}`,
		fmt.Sprintf(
			`{"snapshot_revision":%%d,"updates":[{"group_id":1,"entry_id":"%s","concurrency_limit":%d}]}`,
			scheduleEntryOneA, MaxConcurrencyLimit+1,
		),
	} {
		recorder := scenario.perform(
			http.MethodPatch, "/api/model-route/schedule",
			fmt.Sprintf(body, patched.SnapshotRevisionNew), scenario.authKey,
		)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("invalid concurrency_limit response = %d %s, want 400", recorder.Code, recorder.Body.String())
		}
		if got := loadStoredGroupModelsJSON(t, scenario.fixture, 1); got != before {
			t.Fatalf("invalid concurrency_limit wrote models: %s", got)
		}
	}
}

func TestGroupModelConcurrencyLimitJSONContract(t *testing.T) {
	var model GroupModel
	if err := json.Unmarshal([]byte(`{"id":"model","alias_enabled":true,"concurrency_limit":7}`), &model); err != nil {
		t.Fatalf("decode concurrency_limit: %v", err)
	}
	if model.ConcurrencyLimit != 7 {
		t.Fatalf("concurrency_limit = %d, want 7", model.ConcurrencyLimit)
	}

	var request GroupModelsUpdateRequest
	if err := json.Unmarshal([]byte(`{"models":[{"id":"model","alias_enabled":true,"concurrency_limit":null}]}`), &request); err == nil {
		t.Fatal("null concurrency_limit was accepted")
	}
}

func TestValidateConcurrencyLimit(t *testing.T) {
	for _, value := range []int{0, 1, MaxConcurrencyLimit} {
		if err := validateConcurrencyLimit(value); err != nil {
			t.Errorf("validateConcurrencyLimit(%d) = %v", value, err)
		}
	}
	if err := validateConcurrencyLimit(MaxConcurrencyLimit + 1); err == nil {
		t.Fatal("value above maximum was accepted")
	}
}

func TestGroupModelsConcurrencyLimitRoundTrip(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		ChannelID: channel.OpenAI,
		Params:    json.RawMessage(`{}`),
		Models: optionalGroupModels{Set: true, Values: []GroupModel{
			{ID: "limited", AliasEnabled: false, ConcurrencyLimit: 3},
			{ID: "unlimited", AliasEnabled: false},
		}},
		Credentials: "sk-concurrency-phase-one", ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatalf("CreateGroup() error = %v", err)
	}

	limitsByModel := func(items []GroupModelResponse) map[string]int {
		result := make(map[string]int, len(items))
		for _, item := range items {
			result[item.ID] = item.ConcurrencyLimit
		}
		return result
	}
	read := func(stage string) map[string]int {
		t.Helper()
		got, err := fixture.service.GetGroupModels(t.Context(), created.GroupID)
		if err != nil {
			t.Fatalf("%s GetGroupModels() error = %v", stage, err)
		}
		return limitsByModel(got.Items)
	}

	if got, want := read("after create"), map[string]int{"limited": 3, "unlimited": 0}; !equalLimits(got, want) {
		t.Fatalf("after create limits = %v, want %v", got, want)
	}

	patchLimit := func(value *int, modelID string) {
		t.Helper()
		entry, err := fixture.service.GetGroupModels(t.Context(), created.GroupID)
		if err != nil {
			t.Fatal(err)
		}
		updates := make([]GroupModel, 0, len(entry.Items))
		for _, item := range entry.Items {
			update := GroupModel{ID: item.ID, AliasEnabled: item.AliasEnabled, Alias: item.Alias}
			if item.ID == modelID && value != nil {
				update.ConcurrencyLimit = *value
				update.concurrencyLimitSet = true
			}
			updates = append(updates, update)
		}
		if _, err := fixture.service.UpdateGroupModels(t.Context(), created.GroupID, GroupModelsUpdateRequest{
			Models: optionalGroupModels{Set: true, Values: updates},
		}); err != nil {
			t.Fatalf("UpdateGroupModels(limit=%v model=%s) error = %v", value, modelID, err)
		}
	}

	patchLimit(nil, "")
	if got, want := read("after preserving update"), map[string]int{"limited": 3, "unlimited": 0}; !equalLimits(got, want) {
		t.Fatalf("after preserving update limits = %v, want %v", got, want)
	}

	zero := 0
	patchLimit(&zero, "limited")
	if got, want := read("after clearing limit"), map[string]int{"limited": 0, "unlimited": 0}; !equalLimits(got, want) {
		t.Fatalf("after clearing limit = %v, want %v", got, want)
	}

	seven := 7
	patchLimit(&seven, "limited")
	if got, want := read("after setting limit"), map[string]int{"limited": 7, "unlimited": 0}; !equalLimits(got, want) {
		t.Fatalf("after setting limit = %v, want %v", got, want)
	}

	snapshot := fixture.manager.Current()
	if snapshot == nil {
		t.Fatal("manager.Current() = nil")
	}
	published := make(map[string]int)
	for _, entry := range snapshot.GroupCatalog[created.GroupID].Models {
		published[entry.ID] = entry.ConcurrencyLimit
	}
	if got, want := published, map[string]int{"limited": 7, "unlimited": 0}; !equalLimits(got, want) {
		t.Fatalf("published snapshot limits = %v, want %v", got, want)
	}

	stored, err := stateloader.BuildCompileInput(t.Context(), fixture.service.db, fixture.channelRegistry)
	if err != nil {
		t.Fatalf("LoadCompileInput() error = %v", err)
	}
	reloaded := make(map[string]int)
	for _, group := range stored.Groups {
		if group.ID != created.GroupID {
			continue
		}
		for _, entry := range group.Models {
			reloaded[entry.ID] = entry.ConcurrencyLimit
		}
	}
	if got, want := reloaded, map[string]int{"limited": 7, "unlimited": 0}; !equalLimits(got, want) {
		t.Fatalf("reloaded models limits = %v, want %v", got, want)
	}

	tooLarge := MaxConcurrencyLimit + 1
	if err := json.Unmarshal(
		[]byte(`{"models":[{"id":"model","alias_enabled":true,"concurrency_limit":`+itoa(tooLarge)+`}]}`),
		&GroupModelsUpdateRequest{},
	); !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("decode out-of-range concurrency_limit error = %v, want ErrValidation", err)
	}
}

func TestSnapshotRouteTargetsCarryConcurrencyLimit(t *testing.T) {
	t.Parallel()
	input := state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{{
			ID: 11, Name: "limited-group", ChannelID: channel.OpenAI,
			ConnectionType: "api_key", Params: json.RawMessage(`{}`),
			Models:  []state.ModelConfig{{ID: "upstream", Alias: "public", ConcurrencyLimit: 4}},
			Enabled: true,
		}},
	}
	snapshot, err := state.Compile(input)
	if err != nil {
		t.Fatalf("state.Compile() error = %v", err)
	}
	for _, protocolKeys := range snapshot.ExecutionCandidates {
		for _, byModel := range protocolKeys {
			targets := byModel["public"]
			if len(targets) == 0 {
				continue
			}
			if got := targets[0].ConcurrencyLimit; got != 4 {
				t.Fatalf("RouteTarget.ConcurrencyLimit = %d, want 4", got)
			}
		}
	}
	if got := snapshot.GroupCatalog[11].Models[0].ConcurrencyLimit; got != 4 {
		t.Fatalf("catalog model ConcurrencyLimit = %d, want 4", got)
	}
}

func equalLimits(left, right map[string]int) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range right {
		if left[key] != value {
			return false
		}
	}
	return true
}

func itoa(value int) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}
