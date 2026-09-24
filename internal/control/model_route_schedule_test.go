// Contract tests for the central model-route schedule management API
// (design docs/design/model-route-central-scheduling.md §5, U004). All
// candidate assertions run against the real inspection kernel via published
// snapshots and HTTP endpoints, not hand-built response fixtures.
package control

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

// scheduleTestScenario seeds a database-backed, published snapshot with two
// groups sharing the external model "pub" (three entries in group 1, one in
// group 2) plus one alias-free model, and one active access key. The publish
// counter starts at zero after setup so each test can assert exactly one
// snapshot publish per schedule mutation.
type scheduleTestScenario struct {
	fixture      serviceFixture
	engine       *gin.Engine
	authKey      string
	accessKeyID  uint
	credentialID uint
	now          time.Time
	publishCalls int
	revision     uint64
}

const (
	scheduleEntryOneA    = "e000000000001"
	scheduleEntryOneB    = "e000000000002"
	scheduleEntryOneC    = "e000000000003"
	scheduleEntryTwoB    = "e000000000004"
	scheduleEntrySolo    = "e000000000005"
	scheduleMissingID    = "e999999999999"
	scheduleWeightOneA   = 30
	scheduleWeightOneB   = 50
	scheduleWeightOneC   = 20
	scheduleWeightTwoB   = 100
	schedulePriorityOneC = 2
)

func newScheduleTestScenario(t *testing.T) *scheduleTestScenario {
	t.Helper()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	now := healthNow()
	fixture.service.now = func() time.Time { return now }
	basePublish := fixture.service.publishSnapshot
	scenario := &scheduleTestScenario{
		fixture: fixture,
		authKey: "test-auth-key",
		now:     now,
	}
	fixture.service.publishSnapshot = func(input state.CompileInput) (*state.ConfigSnapshot, error) {
		scenario.publishCalls++
		return basePublish(input)
	}
	threshold := 2
	cooldownSixty := 60
	zeroCooldown := 0
	scenario.mustCreateGroup(t, "one", []GroupModel{
		{ID: "up-a", Alias: "pub", AliasEnabled: true, EntryID: scheduleEntryOneA,
			Weight:         intPtr(scheduleWeightOneA),
			CircuitBreaker: &state.EntryCircuitBreaker{BlacklistThreshold: &threshold}},
		{ID: "up-b", Alias: "pub", AliasEnabled: true, EntryID: scheduleEntryOneB,
			Weight: intPtr(scheduleWeightOneB),
			CircuitBreaker: &state.EntryCircuitBreaker{
				BlacklistThreshold: intPtr(5), CooldownSeconds: &cooldownSixty,
			}},
		{ID: "up-c", Alias: "pub", AliasEnabled: true, EntryID: scheduleEntryOneC,
			Weight: intPtr(scheduleWeightOneC), Priority: intPtr(schedulePriorityOneC),
			CircuitBreaker: &state.EntryCircuitBreaker{CooldownSeconds: &zeroCooldown}},
		{ID: "solo-model", EntryID: scheduleEntrySolo, Weight: intPtr(scheduleWeightTwoB)},
	})
	scenario.mustCreateGroup(t, "two", []GroupModel{
		{ID: "up-b", Alias: "pub", AliasEnabled: true, EntryID: scheduleEntryTwoB,
			Weight: intPtr(scheduleWeightTwoB)},
	})
	key, err := fixture.service.CreateAccessKeyIdempotent(
		t.Context(), "00000000-0000-4000-8000-000000000001",
		AccessKeyCreateRequest{Name: "client"},
	)
	if err != nil {
		t.Fatalf("CreateAccessKeyIdempotent() error = %v", err)
	}
	snapshot := fixture.manager.Current()
	if snapshot == nil {
		t.Fatal("manager.Current() = nil after setup")
	}
	keys := fixture.registry.Snapshot()
	if len(keys) == 0 {
		t.Fatal("registry.Snapshot() is empty after setup")
	}
	scenario.accessKeyID = key.ID
	scenario.credentialID = keys[0].ID
	scenario.revision = snapshot.Revision
	scenario.publishCalls = 0
	scenario.engine = gin.New()
	NewServer(&config.Config{AuthKey: scenario.authKey}, fixture.service).
		RegisterRoutes(scenario.engine)
	return scenario
}

func (scenario *scheduleTestScenario) mustCreateGroup(
	t *testing.T,
	name string,
	models []GroupModel,
) uint {
	t.Helper()
	result, err := scenario.fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: &name, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
		Models:      optionalGroupModels{Set: true, Values: models},
		Credentials: "sk-schedule-" + name, ConnectionType: "api_key",
		// The scenario intentionally shares one channel target across groups.
		ConfirmSameTarget: true,
	})
	if err != nil {
		t.Fatalf("CreateGroup(%q) error = %v", name, err)
	}
	return result.GroupID
}

func intPtr(value int) *int {
	cloned := value
	return &cloned
}

func (scenario *scheduleTestScenario) perform(
	method string,
	path string,
	body string,
	authKey string,
) *httptest.ResponseRecorder {
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("")
	} else {
		reader = strings.NewReader(body)
	}
	request := httptest.NewRequest(method, path, reader)
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	if authKey != "" {
		request.Header.Set("Authorization", "Bearer "+authKey)
	}
	recorder := httptest.NewRecorder()
	scenario.engine.ServeHTTP(recorder, request)
	return recorder
}

func decodeScheduleSuccess(
	t *testing.T,
	recorder *httptest.ResponseRecorder,
	target any,
) {
	t.Helper()
	if recorder.Code != http.StatusOK {
		t.Fatalf("response = %d %s, want 200", recorder.Code, recorder.Body.String())
	}
	var envelope struct {
		Code int             `json:"code"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if envelope.Code != 0 {
		t.Fatalf("code = %d: %s", envelope.Code, recorder.Body.String())
	}
	if err := json.Unmarshal(envelope.Data, target); err != nil {
		t.Fatalf("decode data: %v", err)
	}
}

func decodeScheduleError(
	t *testing.T,
	recorder *httptest.ResponseRecorder,
	wantStatus int,
) string {
	t.Helper()
	if recorder.Code != wantStatus {
		t.Fatalf("response = %d %s, want %d", recorder.Code, recorder.Body.String(), wantStatus)
	}
	var envelope struct {
		Code string `json:"code"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	return envelope.Code
}

func TestModelRouteScheduleIndexAggregatesRealCandidates(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)

	recorder := scenario.perform(http.MethodGet, "/api/model-route/schedule", "", scenario.authKey)
	var result modelRouteScheduleIndexResponse
	decodeScheduleSuccess(t, recorder, &result)

	snapshot := scenario.fixture.manager.Current()
	if snapshot == nil {
		t.Fatal("manager.Current() = nil after setup")
	}
	testAliases := make(map[string]struct{})
	for _, group := range snapshot.GroupCatalog {
		for _, model := range group.Models {
			if model.TestAlias != "" {
				testAliases[model.TestAlias] = struct{}{}
			}
		}
	}

	byModel := make(map[string]modelRouteScheduleIndexItem, len(result.Items))
	for _, item := range result.Items {
		if _, hidden := testAliases[item.ExternalModel]; hidden {
			t.Fatalf("test alias %q is visible in schedule index", item.ExternalModel)
		}
		byModel[item.ExternalModel] = item
	}
	if len(result.Items) != 2 {
		t.Fatalf("items = %#v, want ordinary external models only", result.Items)
	}
	pub := byModel["pub"]
	if pub.ExternalModel != "pub" || pub.Protocol != protocol.OpenAICompletions ||
		pub.Operation != execution.OperationChatCompletion || pub.CandidateCount != 4 || pub.GroupCount != 2 ||
		!pub.HasFallback || pub.CooledCandidates != 0 || pub.BlacklistedCandidates != 0 {
		t.Fatalf("pub item = %#v", pub)
	}
	solo := byModel["solo-model"]
	if solo.CandidateCount != 1 || solo.GroupCount != 1 || solo.HasFallback ||
		solo.Protocol != protocol.OpenAICompletions || solo.Operation != execution.OperationChatCompletion {
		t.Fatalf("solo item = %#v", solo)
	}

	_, _ = scenario.fixture.registry.SetEntryCooldownForEntry(
		1, scheduleEntryOneB, scenario.now.Add(30*time.Minute),
	)
	_, _ = scenario.fixture.registry.SetEntryBlacklistedForEntry(2, scheduleEntryTwoB)
	cooled := scenario.perform(http.MethodGet, "/api/model-route/schedule", "", scenario.authKey)
	var cooledResult modelRouteScheduleIndexResponse
	decodeScheduleSuccess(t, cooled, &cooledResult)
	for _, item := range cooledResult.Items {
		if item.ExternalModel == "pub" &&
			(item.CooledCandidates != 1 || item.BlacklistedCandidates != 1) {
			t.Fatalf("pub runtime badges = %#v", item)
		}
	}
}

func TestModelRouteScheduleIndexRequiresManagementAuthentication(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)
	recorder := scenario.perform(http.MethodGet, "/api/model-route/schedule", "", "")
	if code := decodeScheduleError(t, recorder, http.StatusUnauthorized); code != app_errors.ErrUnauthorized.Code {
		t.Fatalf("index without auth code = %q", code)
	}
	detail := scenario.perform(
		http.MethodGet,
		fmt.Sprintf("/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=%d", scenario.accessKeyID),
		"", "",
	)
	if code := decodeScheduleError(t, detail, http.StatusUnauthorized); code != app_errors.ErrUnauthorized.Code {
		t.Fatalf("detail without auth code = %q", code)
	}
	patch := scenario.perform(http.MethodPatch, "/api/model-route/schedule", `{}`, "")
	if code := decodeScheduleError(t, patch, http.StatusUnauthorized); code != app_errors.ErrUnauthorized.Code {
		t.Fatalf("patch without auth code = %q", code)
	}
	recover := scenario.perform(
		http.MethodPost, "/api/model-route/schedule/recover", `{}`, "",
	)
	if code := decodeScheduleError(t, recover, http.StatusUnauthorized); code != app_errors.ErrUnauthorized.Code {
		t.Fatalf("recover without auth code = %q", code)
	}
}

func TestModelRouteScheduleDetailShowsContextBreakerAndRuntime(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)

	path := fmt.Sprintf(
		"/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=%d",
		scenario.accessKeyID,
	)
	recorder := scenario.perform(http.MethodGet, path, "", scenario.authKey)
	var result modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, recorder, &result)

	if result.ExternalModel == nil || *result.ExternalModel != "pub" ||
		result.Protocol != protocol.OpenAICompletions ||
		result.Operation != execution.OperationChatCompletion ||
		result.RouteRequirement != execution.RouteRequirementAny ||
		result.SnapshotRevision != scenario.revision ||
		result.ObservedAtMS != scenario.now.UnixMilli() ||
		result.AccessKey == nil ||
		result.AccessKey.ID != scenario.accessKeyID ||
		result.AccessKey.Name != "client" ||
		result.AccessKey.Status != state.AccessKeyStatusActive ||
		!result.Routable || result.ReasonCode != nil {
		t.Fatalf("detail header = %#v", result)
	}

	if len(result.Groups) != 2 {
		t.Fatalf("groups = %#v, want 2", result.Groups)
	}
	first := result.Groups[0]
	if first.GroupID != 1 || first.GroupName != "one" || first.ChannelID != channel.OpenAI ||
		len(first.Entries) != 3 {
		t.Fatalf("group one = %#v", first)
	}
	second := result.Groups[1]
	if second.GroupID != 2 || len(second.Entries) != 1 {
		t.Fatalf("group two = %#v", second)
	}

	entryA := first.Entries[0]
	if entryA.EntryID != scheduleEntryOneA || entryA.ModelID != "up-a" ||
		entryA.Alias != "pub" || entryA.Weight != scheduleWeightOneA || entryA.Priority != 1 || entryA.Fallback {
		t.Fatalf("entry up-a = %#v", entryA)
	}
	// Entry configured threshold 2 only: cooldown stays on the judge default.
	if entryA.CircuitBreaker.Configured.BlacklistThreshold == nil ||
		*entryA.CircuitBreaker.Configured.BlacklistThreshold != 2 ||
		entryA.CircuitBreaker.Configured.CooldownSeconds != nil ||
		entryA.CircuitBreaker.Effective.BlacklistThreshold == nil ||
		*entryA.CircuitBreaker.Effective.BlacklistThreshold != 2 ||
		entryA.CircuitBreaker.Effective.CooldownSeconds == nil ||
		*entryA.CircuitBreaker.Effective.CooldownSeconds != scheduleDefaultModelCooldownSeconds ||
		entryA.CircuitBreaker.Sources.BlacklistThreshold != scheduleBreakerSourceEntry ||
		entryA.CircuitBreaker.Sources.CooldownSeconds != scheduleBreakerSourceDefault {
		t.Fatalf("entry up-a breaker = %#v", entryA.CircuitBreaker)
	}
	if entryA.Runtime.State != state.EntryRuntimeAvailable ||
		entryA.Runtime.CooldownUntilMS != nil || entryA.Runtime.FailureCount != 0 {
		t.Fatalf("entry up-a runtime = %#v", entryA.Runtime)
	}
	if first.Enabled != true || second.Enabled != true {
		t.Fatalf("group enabled state = %v/%v, want true/true", first.Enabled, second.Enabled)
	}

	// Effective shares are normalized only among routable entries in the
	// currently active priority tier.
	shareTotal := first.Entries[0].EffectiveShare + first.Entries[1].EffectiveShare +
		first.Entries[2].EffectiveShare + second.Entries[0].EffectiveShare
	if first.Entries[0].EffectiveShare <= 0 || first.Entries[1].EffectiveShare <= 0 ||
		second.Entries[0].EffectiveShare <= 0 || first.Entries[2].EffectiveShare != 0 ||
		shareTotal < 1-1e-9 || shareTotal > 1+1e-9 {
		t.Fatalf("same-priority effective shares = %v/%v/%v/%v, total=%v",
			first.Entries[0].EffectiveShare, first.Entries[1].EffectiveShare,
			first.Entries[2].EffectiveShare, second.Entries[0].EffectiveShare, shareTotal)
	}

	configured := []struct {
		name string
		got  float64
		want float64
	}{
		{"up-a", first.Entries[0].ConfiguredShare, 30.0 / 180.0},
		{"up-b", first.Entries[1].ConfiguredShare, 50.0 / 180.0},
		{"two/up-b", second.Entries[0].ConfiguredShare, 100.0 / 180.0},
		{"up-c", first.Entries[2].ConfiguredShare, 1},
	}
	for _, test := range configured {
		if diff := test.got - test.want; diff < -1e-9 || diff > 1e-9 {
			t.Fatalf("entry %s configured_share = %v, want %v", test.name, test.got, test.want)
		}
	}

	entryC := first.Entries[2]
	if entryC.EntryID != scheduleEntryOneC || !entryC.Fallback ||
		entryC.Priority != schedulePriorityOneC ||
		entryC.EffectiveShare != 0 {
		t.Fatalf("entry up-c = %#v", entryC)
	}
	// Partial override: zero cooldown configured, threshold never counts.
	if entryC.CircuitBreaker.Configured.BlacklistThreshold != nil ||
		entryC.CircuitBreaker.Configured.CooldownSeconds == nil ||
		*entryC.CircuitBreaker.Configured.CooldownSeconds != 0 ||
		entryC.CircuitBreaker.Effective.CooldownSeconds == nil ||
		*entryC.CircuitBreaker.Effective.CooldownSeconds != 0 ||
		entryC.CircuitBreaker.Sources.BlacklistThreshold != scheduleBreakerSourceDefault ||
		entryC.CircuitBreaker.Sources.CooldownSeconds != scheduleBreakerSourceEntry {
		t.Fatalf("entry up-c breaker = %#v", entryC.CircuitBreaker)
	}

	entryTwoB := second.Entries[0]
	if entryTwoB.EntryID != scheduleEntryTwoB || entryTwoB.Weight != scheduleWeightTwoB ||
		entryTwoB.CircuitBreaker.Configured.BlacklistThreshold != nil ||
		entryTwoB.CircuitBreaker.Configured.CooldownSeconds != nil ||
		entryTwoB.CircuitBreaker.Sources.CooldownSeconds != scheduleBreakerSourceDefault ||
		!entryTwoB.Routable {
		t.Fatalf("entry two/up-b = %#v", entryTwoB)
	}
	if len(entryA.Credentials) == 0 || !entryA.Credentials[0].Available {
		t.Fatalf("entry up-a credentials = %#v", entryA.Credentials)
	}

	// Runtime badges: failure counting, future cooldown and blacklist surface
	_, _ = scenario.fixture.registry.IncrEntryFailureForEntry(1, scheduleEntryOneA)
	_, _ = scenario.fixture.registry.SetEntryCooldownForEntry(
		1, scheduleEntryOneB, scenario.now.Add(30*time.Minute),
	)
	_, _ = scenario.fixture.registry.SetEntryBlacklistedForEntry(2, scheduleEntryTwoB)
	if !scenario.fixture.registry.SetEntryBlacklistReleaseAt(
		2, scheduleEntryTwoB, scenario.now.Add(2*time.Hour),
	) {
		t.Fatal("entry blacklist deadline = false")
	}
	runtime := scenario.perform(http.MethodGet, path, "", scenario.authKey)
	var runtimeResult modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, runtime, &runtimeResult)
	first = runtimeResult.Groups[0]
	second = runtimeResult.Groups[1]
	if first.Entries[0].Runtime.FailureCount != 1 ||
		first.Entries[0].Runtime.State != state.EntryRuntimeAvailable {
		t.Fatalf("up-a runtime after failure = %#v", first.Entries[0].Runtime)
	}
	if first.Entries[0].Runtime.FailureVersion == 0 ||
		first.Entries[0].Runtime.BlacklistReleaseAtMS != nil {
		t.Fatalf("up-a runtime proof fields = %#v", first.Entries[0].Runtime)
	}
	cooled := first.Entries[1]
	if cooled.Runtime.State != state.EntryRuntimeCooldown ||
		cooled.Runtime.CooldownUntilMS == nil ||
		*cooled.Runtime.CooldownUntilMS != scenario.now.Add(30*time.Minute).UnixMilli() {
		t.Fatalf("up-b runtime after cooldown = %#v", cooled.Runtime)
	}
	assertScheduleReason(t, cooled.ReasonCode, scheduler.ReasonEntryCooldown)
	blacklisted := second.Entries[0]
	if blacklisted.Runtime.State != state.EntryRuntimeBlacklisted ||
		blacklisted.Runtime.CooldownUntilMS != nil ||
		blacklisted.Runtime.BlacklistReleaseAtMS == nil ||
		*blacklisted.Runtime.BlacklistReleaseAtMS != scenario.now.Add(2*time.Hour).UnixMilli() ||
		blacklisted.Runtime.FailureVersion == 0 {
		t.Fatalf("two/up-b runtime after blacklist = %#v", blacklisted.Runtime)
	}
	assertScheduleReason(t, blacklisted.ReasonCode, scheduler.ReasonEntryBlacklisted)
	if cooled.ConfiguredShare != 50.0/180.0 || blacklisted.ConfiguredShare != 100.0/180.0 {
		t.Fatalf("unavailable configured shares = %v/%v, want %v/%v", cooled.ConfiguredShare, blacklisted.ConfiguredShare, 50.0/180.0, 100.0/180.0)
	}
}

func newReasoningScheduleTestScenario(t *testing.T) *scheduleTestScenario {
	t.Helper()
	scenario := newScheduleTestScenario(t)
	entries := loadCreatedGroupModels(t, scenario.fixture, 1)
	for index := range entries {
		entries[index].ID = fmt.Sprintf("gpt-5.%d", index+1)
	}
	encoded, err := json.Marshal(entries)
	if err != nil {
		t.Fatal(err)
	}
	if err := scenario.fixture.db.Model(&models.Group{}).Where("id = ?", 1).Update("models", models.JSON(encoded)).Error; err != nil {
		t.Fatal(err)
	}
	input, err := stateloader.BuildCompileInputWithProxy(t.Context(), scenario.fixture.db, scenario.fixture.service.encryption, scenario.fixture.service.environmentProxy, scenario.fixture.service.channelRegistry)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := scenario.fixture.manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	scenario.revision = scenario.fixture.manager.Current().Revision
	return scenario
}

func TestModelRouteScheduleDetailProjectsReasoningPolicy(t *testing.T) {
	scenario := newReasoningScheduleTestScenario(t)
	body := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","reasoning_effort":"high"}]}`, scenario.revision, scheduleEntryOneA)
	recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey)
	var patch modelRouteSchedulePatchResponse
	decodeScheduleSuccess(t, recorder, &patch)
	path := fmt.Sprintf("/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=%d", scenario.accessKeyID)
	recorder = scenario.perform(http.MethodGet, path, "", scenario.authKey)
	var detail modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, recorder, &detail)
	first := detail.Groups[0]
	entry := first.Entries[0]
	if entry.Reasoning.Configured == nil || *entry.Reasoning.Configured != "high" ||
		entry.Reasoning.Effective == nil || *entry.Reasoning.Effective != "high" || entry.Reasoning.Source != "entry" {
		t.Fatalf("entry reasoning projection = %#v", entry.Reasoning)
	}

	providerDefault := first.Entries[1]
	if providerDefault.Reasoning.Configured != nil || providerDefault.Reasoning.Effective != nil ||
		providerDefault.Reasoning.Source != reasoning.EffortSourceProviderDefault {
		t.Fatalf("unconfigured reasoning projection = %#v", providerDefault.Reasoning)
	}
}

func TestModelRouteScheduleDeletesGroupReasoningContract(t *testing.T) {
	scenario := newScheduleTestScenario(t)
	path := fmt.Sprintf(
		"/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=%d",
		scenario.accessKeyID,
	)
	detail := scenario.perform(http.MethodGet, path, "", scenario.authKey)
	if strings.Contains(detail.Body.String(), `"reasoning_effort_default"`) ||
		strings.Contains(detail.Body.String(), `"reasoning_entries"`) {
		t.Fatalf("schedule detail still exposes group reasoning fields: %s", detail.Body.String())
	}
	legacy := fmt.Sprintf(
		`{"snapshot_revision":%d,"group_updates":[{"group_id":1,"reasoning_effort_default":"low"}]}`,
		scenario.revision,
	)
	legacyResponse := scenario.perform(http.MethodPatch, "/api/model-route/schedule", legacy, scenario.authKey)
	if legacyResponse.Code != http.StatusBadRequest {
		t.Fatalf("legacy group reasoning patch = %d %s, want 400", legacyResponse.Code, legacyResponse.Body.String())
	}
}

func TestModelRouteScheduleRejectsDerivedEntryIdentityForPersistence(t *testing.T) {
	scenario := newScheduleTestScenario(t)
	var group models.Group
	if err := scenario.fixture.db.First(&group, 1).Error; err != nil {
		t.Fatal(err)
	}
	var entries []map[string]any
	if err := json.Unmarshal(group.Models, &entries); err != nil {
		t.Fatal(err)
	}
	delete(entries[0], "entry_id")
	encoded, _ := json.Marshal(entries)
	if err := scenario.fixture.db.Model(&models.Group{}).Where("id = ?", 1).Update("models", models.JSON(encoded)).Error; err != nil {
		t.Fatal(err)
	}
	input, err := stateloader.BuildCompileInputWithProxy(t.Context(), scenario.fixture.db, scenario.fixture.service.encryption, scenario.fixture.service.environmentProxy, scenario.fixture.service.channelRegistry)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := scenario.fixture.manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	revision := scenario.fixture.manager.Current().Revision
	body := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"derived:pub#up-a","reasoning_effort":"high"}]}`, revision)
	recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("derived entry patch = %d %s, want 400", recorder.Code, recorder.Body.String())
	}
	if got := loadStoredGroupModelsJSON(t, scenario.fixture, 1); got != string(encoded) {
		t.Fatalf("derived entry rejection changed storage: %s", got)
	}
}

func TestModelRouteScheduleValidatesReasoningPerEntry(t *testing.T) {
	for _, tc := range []struct {
		name    string
		initial string
		value   string
		want    int
		stored  string
	}{
		{name: "supported override", value: `"high"`, want: http.StatusOK, stored: "high"},
		{name: "canonical low override", value: `"low"`, want: http.StatusOK, stored: "low"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scenario := newScheduleTestScenario(t)
			name := "item-reasoning-gemini"
			created, err := scenario.fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
				Name: &name, ChannelID: channel.Gemini, Params: json.RawMessage(`{}`),
				Models: optionalGroupModels{Set: true, Values: []GroupModel{
					{ID: "gemini-3.7-flash", EntryID: scheduleEntryOneA},
					{ID: "gemini-3.1-flash-lite-image", EntryID: scheduleEntryOneB, ReasoningEffort: tc.initial},
				}}, Credentials: "gemini-key", ConnectionType: "api_key",
			})
			if err != nil {
				t.Fatal(err)
			}
			var before models.Group
			if err := scenario.fixture.db.First(&before, created.GroupID).Error; err != nil {
				t.Fatal(err)
			}
			revision := scenario.fixture.manager.Current().Revision
			publishes := scenario.publishCalls
			body := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":%d,"entry_id":"%s","reasoning_effort":%s}]}`,
				revision, created.GroupID, scheduleEntryOneB, tc.value)
			recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey)
			if recorder.Code != tc.want {
				t.Fatalf("PATCH = %d %s, want %d", recorder.Code, recorder.Body.String(), tc.want)
			}
			if tc.want == http.StatusBadRequest {
				if !strings.Contains(recorder.Body.String(), "gemini-3.1-flash-lite-image") {
					t.Fatalf("missing affected model: %s", recorder.Body.String())
				}
				var after models.Group
				if err := scenario.fixture.db.First(&after, created.GroupID).Error; err != nil {
					t.Fatal(err)
				}
				if string(before.Models) != string(after.Models) || scenario.fixture.manager.Current().Revision != revision || scenario.publishCalls != publishes {
					t.Fatal("rejected patch changed storage or snapshot")
				}
			} else {
				entries := loadCreatedGroupModels(t, scenario.fixture, created.GroupID)
				if entries[1].ReasoningEffort != tc.stored {
					t.Fatalf("image override = %q, want %q", entries[1].ReasoningEffort, tc.stored)
				}
			}
		})
	}
}

func TestModelRouteScheduleReasoningPatchTriStateAndAtomicValidation(t *testing.T) {
	scenario := newReasoningScheduleTestScenario(t)
	set := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","reasoning_effort":"high"}]}`, scenario.revision, scheduleEntryOneA)
	recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", set, scenario.authKey)
	var result modelRouteSchedulePatchResponse
	decodeScheduleSuccess(t, recorder, &result)
	if got := loadCreatedGroupModels(t, scenario.fixture, 1)[0].ReasoningEffort; got != "high" {
		t.Fatalf("stored entry reasoning = %q, want high", got)
	}
	beforeModels := loadStoredGroupModelsJSON(t, scenario.fixture, 1)
	invalid := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","reasoning_effort":"invalid"}]}`, result.SnapshotRevisionNew, scheduleEntryOneA)
	recorder = scenario.perform(http.MethodPatch, "/api/model-route/schedule", invalid, scenario.authKey)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("invalid reasoning response = %d %s, want 400", recorder.Code, recorder.Body.String())
	}
	if got := loadStoredGroupModelsJSON(t, scenario.fixture, 1); got != beforeModels {
		t.Fatalf("invalid batch wrote models: %s", got)
	}
	clear := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","reasoning_effort":null}]}`, result.SnapshotRevisionNew, scheduleEntryOneA)
	recorder = scenario.perform(http.MethodPatch, "/api/model-route/schedule", clear, scenario.authKey)
	decodeScheduleSuccess(t, recorder, &result)
	if got := loadCreatedGroupModels(t, scenario.fixture, 1)[0].ReasoningEffort; got != "" {
		t.Fatalf("cleared entry reasoning = %q, want empty", got)
	}
}

func TestModelRouteScheduleDetailKeepsDisabledGroupConfiguration(t *testing.T) {
	t.Parallel()
	scenario := newReasoningScheduleTestScenario(t)
	patch := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","reasoning_effort":"high"}]}`, scenario.revision, scheduleEntryOneA)
	var patchResult modelRouteSchedulePatchResponse
	decodeScheduleSuccess(t, scenario.perform(http.MethodPatch, "/api/model-route/schedule", patch, scenario.authKey), &patchResult)
	if err := scenario.fixture.db.Model(&models.Group{}).
		Where("id = ?", 1).
		Update("enabled", false).Error; err != nil {
		t.Fatalf("disable group: %v", err)
	}
	input, err := stateloader.BuildCompileInputWithProxy(
		t.Context(), scenario.fixture.db, scenario.fixture.service.encryption,
		scenario.fixture.service.environmentProxy, scenario.fixture.service.channelRegistry,
	)
	if err != nil {
		t.Fatalf("rebuild disabled snapshot input: %v", err)
	}
	if _, err := scenario.fixture.manager.Publish(input); err != nil {
		t.Fatalf("publish disabled snapshot: %v", err)
	}

	path := fmt.Sprintf(
		"/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=%d",
		scenario.accessKeyID,
	)
	recorder := scenario.perform(http.MethodGet, path, "", scenario.authKey)
	var result modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, recorder, &result)
	if len(result.Groups) != 2 || len(result.Groups[0].Entries) == 0 {
		t.Fatalf("disabled group detail = %#v", result.Groups)
	}
	entry := result.Groups[0].Entries[0]
	if entry.EntryID != scheduleEntryOneA || entry.Alias != "pub" ||
		entry.CircuitBreaker.Configured.BlacklistThreshold == nil ||
		*entry.CircuitBreaker.Configured.BlacklistThreshold != 2 ||
		entry.CircuitBreaker.Sources.BlacklistThreshold != scheduleBreakerSourceEntry {
		t.Fatalf("disabled group entry configuration = %#v", entry)
	}
	if entry.Reasoning.Configured == nil || *entry.Reasoning.Configured != "high" ||
		entry.Reasoning.Effective == nil || *entry.Reasoning.Effective != "high" || entry.Reasoning.Source != "entry" {
		t.Fatalf("disabled group reasoning configuration = %#v", entry.Reasoning)
	}
	assertScheduleReason(t, entry.ReasonCode, scheduler.ReasonGroupDisabled)
	for index, disabledEntry := range result.Groups[0].Entries {
		if disabledEntry.ConfiguredShare != 0 {
			t.Fatalf("disabled group entry %d configured share = %v, want 0", index, disabledEntry.ConfiguredShare)
		}
	}
	if len(result.Groups[1].Entries) != 1 || result.Groups[1].Entries[0].ConfiguredShare != 1 {
		t.Fatalf("enabled group configured shares = %#v, want 1", result.Groups[1].Entries)
	}
}

func TestModelRouteScheduleAcceptsCanonicalEffortForUnlistedModel(t *testing.T) {
	scenario := newScheduleTestScenario(t)
	body := fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","reasoning_effort":"max"}]}`, scenario.revision, scheduleEntryOneA)
	recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey)
	if recorder.Code != http.StatusOK {
		t.Fatalf("canonical effort = %d %s, want 200", recorder.Code, recorder.Body.String())
	}
	if got := loadCreatedGroupModels(t, scenario.fixture, 1)[0].ReasoningEffort; got != "max" {
		t.Fatalf("stored effort = %q, want max", got)
	}
}

func assertScheduleReason(
	t *testing.T,
	got *scheduler.ReasonCode,
	want scheduler.ReasonCode,
) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("reason = %v, want %q", got, want)
	}
}
func TestModelRouteScheduleDetailReturnsGroupStateAndRollingUsage(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)
	scenario.fixture.service.requestLogs = &scheduleUsageReader{usage: map[uint]requestlog.GroupUsage{
		1: {RequestCount: 8, SuccessCount: 6},
		2: {RequestCount: 4, SuccessCount: 4},
	}}

	path := fmt.Sprintf(
		"/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=%d",
		scenario.accessKeyID,
	)
	recorder := scenario.perform(http.MethodGet, path, "", scenario.authKey)
	var result modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, recorder, &result)
	if len(result.Groups) != 2 {
		t.Fatalf("groups = %#v, want 2", result.Groups)
	}
	if result.Groups[0].Enabled != true || result.Groups[0].RequestCount != 8 ||
		result.Groups[0].SuccessRate != 0.75 {
		t.Fatalf("group one state/usage = %#v", result.Groups[0])
	}
	if result.Groups[1].Enabled != true || result.Groups[1].RequestCount != 4 ||
		result.Groups[1].SuccessRate != 1 {
		t.Fatalf("group two state/usage = %#v", result.Groups[1])
	}

	if _, err := scenario.fixture.service.UpdateGroupSettings(t.Context(), 1, GroupSettingsUpdateRequest{
		Enabled: optionalField[bool]{Set: true, Value: false},
	}); err != nil {
		t.Fatalf("disable group: %v", err)
	}
	recorder = scenario.perform(http.MethodGet, path, "", scenario.authKey)
	decodeScheduleSuccess(t, recorder, &result)
	if result.Groups[0].Enabled {
		t.Fatalf("group one enabled = true after safe settings update")
	}
}

type scheduleUsageReader struct {
	usage map[uint]requestlog.GroupUsage
}

func (reader *scheduleUsageReader) List(context.Context, requestlog.ListQuery) (requestlog.Page, error) {
	return requestlog.Page{}, nil
}

func (reader *scheduleUsageReader) Get(context.Context, string) (requestlog.Record, error) {
	return requestlog.Record{}, nil
}

func (reader *scheduleUsageReader) QueryGroupUsage(
	context.Context,
	requestlog.GroupUsageQuery,
) (map[uint]requestlog.GroupUsage, error) {
	return reader.usage, nil
}

func TestModelRouteScheduleDetailResolvesProtocolAndOperationContext(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)

	responses := func(path string) modelRouteScheduleDetailResponse {
		recorder := scenario.perform(http.MethodGet, path, "", scenario.authKey)
		var result modelRouteScheduleDetailResponse
		decodeScheduleSuccess(t, recorder, &result)
		return result
	}
	base := fmt.Sprintf("/api/model-route/schedule/detail?external_model=pub&access_key_id=%d", scenario.accessKeyID)

	images := responses(base + "&protocol=openai-images")
	if images.Protocol != protocol.OpenAIImages ||
		images.Operation != execution.OperationImagesGenerate ||
		len(images.Groups) != 2 || len(images.Groups[0].Entries) != 3 {
		t.Fatalf("images detail = %#v", images)
	}

	responsesProtocol := responses(base + "&protocol=openai-responses&operation=responses_create")
	if responsesProtocol.Operation != execution.OperationResponsesCreate ||
		len(responsesProtocol.Groups) != 2 {
		t.Fatalf("responses detail = %#v", responsesProtocol)
	}

	responsesRetrieve := responses(base + "&protocol=openai-responses&operation=responses_retrieve")
	if responsesRetrieve.RouteRequirement != execution.RouteRequirementNative ||
		responsesRetrieve.Operation != execution.OperationResponsesRetrieve {
		t.Fatalf("responses retrieve detail = %#v", responsesRetrieve)
	}

	unsupported := responses(base + "&protocol=openai-completions&operation=images_generate")
	if unsupported.Routable || len(unsupported.Groups) != 0 {
		t.Fatalf("unsupported operation detail = %#v", unsupported)
	}
	assertScheduleReason(t, unsupported.ReasonCode, scheduler.ReasonOperationUnsupported)

	missingKey := scenario.perform(
		http.MethodGet,
		"/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=404",
		"", scenario.authKey,
	)
	if code := decodeScheduleError(t, missingKey, http.StatusNotFound); code != app_errors.ErrResourceNotFound.Code {
		t.Fatalf("missing access key code = %q", code)
	}

	// 调度中心不带 access_key_id：返回无过滤候选表，响应不回显密钥。
	unscoped := responses(
		"/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions",
	)
	if unscoped.AccessKey != nil || len(unscoped.Groups) != 2 || !unscoped.Routable {
		t.Fatalf("unscoped detail = %#v", unscoped)
	}

	invalid := scenario.perform(
		http.MethodGet,
		"/api/model-route/schedule/detail?external_model=pub&protocol=invalid&access_key_id=1",
		"", scenario.authKey,
	)
	if code := decodeScheduleError(t, invalid, http.StatusBadRequest); code != app_errors.ErrValidation.Code {
		t.Fatalf("invalid protocol code = %q", code)
	}
}

func TestModelRouteScheduleEntryEnabledIsPersistedAndExcludesRealScheduling(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)
	if legacy := loadCreatedGroupModels(t, scenario.fixture, 1)[0]; legacy.Enabled != nil {
		t.Fatalf("legacy entry has explicit enabled = %v, want omitted", *legacy.Enabled)
	}
	path := fmt.Sprintf("/api/model-route/schedule/detail?external_model=pub&protocol=openai-completions&access_key_id=%d", scenario.accessKeyID)
	var initial modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, scenario.perform(http.MethodGet, path, "", scenario.authKey), &initial)
	if !initial.Groups[0].Entries[0].Enabled {
		t.Fatal("legacy entry without enabled must default to enabled")
	}

	body := fmt.Sprintf(`{"snapshot_revision":%d,"protocol":"openai-completions","external_model":"pub","access_key_id":%d,"updates":[{"group_id":1,"entry_id":"%s","enabled":false}]}`, scenario.revision, scenario.accessKeyID, scheduleEntryOneA)
	var patch modelRouteSchedulePatchResponse
	decodeScheduleSuccess(t, scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey), &patch)
	if patch.Detail == nil {
		t.Fatal("patch detail = nil, want entry state echo")
	}
	var disabled modelRouteScheduleEntryResponse
	for _, entry := range patch.Detail.Groups[0].Entries {
		if entry.EntryID == scheduleEntryOneA {
			disabled = entry
			break
		}
	}
	if disabled.Enabled || disabled.Included || disabled.Routable || disabled.ConfiguredShare != 0 || disabled.EffectiveShare != 0 {
		t.Fatalf("disabled entry still participates: %#v", disabled)
	}
	assertScheduleReason(t, disabled.ReasonCode, scheduler.ReasonEntryDisabled)
	if !patch.Detail.Groups[0].Entries[1].Enabled || !patch.Detail.Groups[0].Enabled {
		t.Fatalf("sibling/group disabled by entry patch: %#v", patch.Detail.Groups[0])
	}
	if disabled.Weight != scheduleWeightOneA || disabled.Priority != 1 {
		t.Fatalf("disabled entry configuration changed: %#v", disabled)
	}

	for seed := int64(1); seed <= 100; seed++ {
		iterator := scheduler.New(
			scenario.fixture.manager.Current(), scenario.fixture.registry,
			scheduler.Query{
				ClientProtocol: protocol.OpenAICompletions,
				Operation:      execution.OperationChatCompletion,
				ExternalModel:  stringPointer("pub"),
			}, rand.New(rand.NewSource(seed)),
		)
		for {
			selected, err := iterator.Next()
			if err == scheduler.ErrExhausted {
				break
			}
			if err != nil {
				t.Fatalf("scheduler.Next() after disabling entry = %v", err)
			}
			if selected.EntryID == scheduleEntryOneA {
				t.Fatalf("scheduler selected disabled entry at seed %d: %#v", seed, selected)
			}
		}
	}

	// Reload from the database rather than reuse the published in-memory view.
	stored := loadCreatedGroupModels(t, scenario.fixture, 1)
	if stored[0].Enabled == nil || *stored[0].Enabled || stored[0].Weight == nil || *stored[0].Weight != scheduleWeightOneA {
		t.Fatalf("persisted disabled entry = %#v", stored[0])
	}
	input, err := stateloader.BuildCompileInputWithProxy(t.Context(), scenario.fixture.db, scenario.fixture.service.encryption, scenario.fixture.service.environmentProxy, scenario.fixture.service.channelRegistry)
	if err != nil {
		t.Fatalf("reload persisted input: %v", err)
	}
	if _, err := scenario.fixture.manager.Publish(input); err != nil {
		t.Fatalf("publish reloaded input: %v", err)
	}
	var reloaded modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, scenario.perform(http.MethodGet, path, "", scenario.authKey), &reloaded)
	if reloaded.Groups[0].Entries[0].Enabled || !reloaded.Groups[0].Enabled {
		t.Fatalf("reload lost entry-only disabled state: %#v", reloaded.Groups[0])
	}
	body = fmt.Sprintf(`{"snapshot_revision":%d,"protocol":"openai-completions","external_model":"pub","access_key_id":%d,"updates":[{"group_id":1,"entry_id":"%s","enabled":true}]}`, reloaded.SnapshotRevision, scenario.accessKeyID, scheduleEntryOneA)
	decodeScheduleSuccess(t, scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey), &patch)
	var restored modelRouteScheduleEntryResponse
	for _, entry := range patch.Detail.Groups[0].Entries {
		if entry.EntryID == scheduleEntryOneA {
			restored = entry
			break
		}
	}
	if !restored.Enabled || restored.Weight != scheduleWeightOneA || restored.Priority != 1 || !restored.Included {
		t.Fatalf("re-enabled entry configuration = %#v", restored)
	}
	before := loadStoredGroupModelsJSON(t, scenario.fixture, 1)
	body = fmt.Sprintf(`{"snapshot_revision":%d,"updates":[{"group_id":1,"entry_id":"%s","enabled":null}]}`, patch.SnapshotRevisionNew, scheduleEntryOneA)
	if code := decodeScheduleError(t, scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey), http.StatusBadRequest); code != app_errors.ErrValidation.Code {
		t.Fatalf("null enabled code = %q, want validation error", code)
	}
	if got := loadStoredGroupModelsJSON(t, scenario.fixture, 1); got != before {
		t.Fatalf("null enabled changed persisted models: %s", got)
	}
}

func TestModelRouteSchedulePatchAppliesTriStateAtomically(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)

	body := fmt.Sprintf(`{
		"snapshot_revision": %d,
		"protocol": "openai-completions",
		"external_model": "pub",
		"access_key_id": %d,
		"updates": [
			{"group_id": 1, "entry_id": "%s", "weight": 40,
			 "circuit_breaker": {"cooldown_seconds": 30}},
			{"group_id": 1, "entry_id": "%s", "weight": 10, "priority": 3,
			 "circuit_breaker": {"blacklist_threshold": null}},
			{"group_id": 2, "entry_id": "%s", "weight": 50, "circuit_breaker": {}}
		]
	}`, scenario.revision, scenario.accessKeyID, scheduleEntryOneA, scheduleEntryOneB, scheduleEntryTwoB)

	recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey)
	var result modelRouteSchedulePatchResponse
	decodeScheduleSuccess(t, recorder, &result)

	if scenario.publishCalls != 1 {
		t.Fatalf("publish calls = %d, want exactly 1", scenario.publishCalls)
	}
	if result.SnapshotRevisionNew != scenario.revision+1 {
		t.Fatalf("snapshot_revision_new = %d, want %d", result.SnapshotRevisionNew, scenario.revision+1)
	}
	if result.Detail == nil || len(result.Detail.Groups) == 0 {
		t.Fatalf("patch detail = %#v", result.Detail)
	}
	detail := result.Detail
	if detail.Protocol != protocol.OpenAICompletions ||
		detail.Operation != execution.OperationChatCompletion ||
		detail.ExternalModel == nil || *detail.ExternalModel != "pub" ||
		detail.AccessKey == nil || detail.AccessKey.ID != scenario.accessKeyID ||
		detail.SnapshotRevision != result.SnapshotRevisionNew {
		t.Fatalf("patch detail header = %#v", detail)
	}
	if detail.Groups[0].Entries[0].Weight != 40 {
		t.Fatalf("patch detail up-a weight = %#v", detail.Groups[0].Entries[0])
	}

	// Storage rows carry the merged tri-state result.
	firstModels := loadCreatedGroupModels(t, scenario.fixture, 1)
	if len(firstModels) != 4 {
		t.Fatalf("group one models = %#v", firstModels)
	}
	byEntry := make(map[string]GroupModel, len(firstModels))
	for _, model := range firstModels {
		byEntry[model.EntryID] = model
	}
	upA := byEntry[scheduleEntryOneA]
	if upA.Weight == nil || *upA.Weight != 40 || upA.Priority != nil {
		t.Fatalf("up-a after patch = %#v", upA)
	}
	if upA.CircuitBreaker == nil || upA.CircuitBreaker.BlacklistThreshold == nil ||
		*upA.CircuitBreaker.BlacklistThreshold != 2 ||
		upA.CircuitBreaker.CooldownSeconds == nil || *upA.CircuitBreaker.CooldownSeconds != 30 {
		t.Fatalf("up-a breaker after patch = %#v", upA.CircuitBreaker)
	}
	upB := byEntry[scheduleEntryOneB]
	if upB.Weight == nil || *upB.Weight != 10 ||
		upB.Priority == nil || *upB.Priority != 3 {
		t.Fatalf("up-b after patch = %#v", upB)
	}
	// Partial override: threshold cleared, cooldown kept.
	if upB.CircuitBreaker == nil || upB.CircuitBreaker.BlacklistThreshold != nil ||
		upB.CircuitBreaker.CooldownSeconds == nil || *upB.CircuitBreaker.CooldownSeconds != 60 {
		t.Fatalf("up-b breaker after patch = %#v", upB.CircuitBreaker)
	}
	secondModels := loadCreatedGroupModels(t, scenario.fixture, 2)
	if len(secondModels) != 1 || secondModels[0].Weight == nil || *secondModels[0].Weight != 50 {
		t.Fatalf("group two after patch = %#v", secondModels)
	}
	if secondModels[0].CircuitBreaker != nil {
		t.Fatalf("group two breaker must stay unchanged = %#v", secondModels[0].CircuitBreaker)
	}
	if secondModels[0].Priority != nil {
		t.Fatalf("absent priority must stay unchanged = %#v", secondModels[0].Priority)
	}

	// Re-publishing snapshot keeps the route catalog consistent with storage.
	index := scenario.perform(http.MethodGet, "/api/model-route/schedule", "", scenario.authKey)
	var indexResult modelRouteScheduleIndexResponse
	decodeScheduleSuccess(t, index, &indexResult)
	for _, item := range indexResult.Items {
		if item.ExternalModel == "pub" && item.CandidateCount != 4 {
			t.Fatalf("pub candidates after patch = %#v", item)
		}
	}
}

func TestModelRouteSchedulePatchRejectsInvalidBatchWithoutWrites(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)

	loadWeights := func(t *testing.T) []int {
		t.Helper()
		first := loadCreatedGroupModels(t, scenario.fixture, 1)
		second := loadCreatedGroupModels(t, scenario.fixture, 2)
		return []int{*first[0].Weight, *first[1].Weight, *first[2].Weight, *second[0].Weight}
	}
	before := loadWeights(t)
	beforeRevision := scenario.fixture.manager.Current().Revision

	// One valid update plus one update for a nonexistent entry: the whole
	// batch is rejected and nothing is written or published.
	body := fmt.Sprintf(`{
		"snapshot_revision": %d,
		"updates": [
			{"group_id": 1, "entry_id": "%s", "weight": 99},
			{"group_id": 2, "entry_id": "%s", "weight": 1}
		]
	}`, scenario.revision, scheduleEntryOneA, scheduleMissingID)
	recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", body, scenario.authKey)
	if code := decodeScheduleError(t, recorder, http.StatusBadRequest); code != app_errors.ErrValidation.Code {
		t.Fatalf("missing entry code = %q", code)
	}
	if scenario.publishCalls != 0 {
		t.Fatalf("publish calls after rejection = %d, want 0", scenario.publishCalls)
	}
	if after := loadWeights(t); !reflect.DeepEqual(after, before) {
		t.Fatalf("weights after rejected batch = %v, want %v", after, before)
	}
	if scenario.fixture.manager.Current().Revision != beforeRevision {
		t.Fatal("rejected batch changed the snapshot revision")
	}

	// Out-of-range weight (V3: 0..MaxWeight) also rejects the whole batch.
	invalidWeight := fmt.Sprintf(`{
		"snapshot_revision": %d,
		"updates": [
			{"group_id": 1, "entry_id": "%s", "weight": 101},
			{"group_id": 2, "entry_id": "%s", "weight": 7}
		]
	}`, scenario.revision, scheduleEntryOneA, scheduleEntryTwoB)
	recorder = scenario.perform(http.MethodPatch, "/api/model-route/schedule", invalidWeight, scenario.authKey)
	if code := decodeScheduleError(t, recorder, http.StatusBadRequest); code != app_errors.ErrValidation.Code {
		t.Fatalf("invalid weight code = %q", code)
	}
	if scenario.publishCalls != 0 {
		t.Fatalf("publish calls after invalid weight = %d, want 0", scenario.publishCalls)
	}
	if after := loadWeights(t); !reflect.DeepEqual(after, before) {
		t.Fatalf("weights after invalid weight = %v, want %v", after, before)
	}

	// Structural validation: unknown field, missing snapshot_revision,
	// empty updates, missing entry_id.
	for name, payload := range map[string]string{
		"unknown field": fmt.Sprintf(
			`{"snapshot_revision": %d, "updates": [{"group_id": 1, "entry_id": "%s", "weigth": 1}]}`,
			scenario.revision, scheduleEntryOneA),
		"missing revision":  `{"updates": [{"group_id": 1, "entry_id": "e000000000001", "weight": 1}]}`,
		"empty updates":     fmt.Sprintf(`{"snapshot_revision": %d, "updates": []}`, scenario.revision),
		"blank entry id":    fmt.Sprintf(`{"snapshot_revision": %d, "updates": [{"group_id": 1, "entry_id": " ", "weight": 1}]}`, scenario.revision),
		"zero group id":     fmt.Sprintf(`{"snapshot_revision": %d, "updates": [{"group_id": 0, "entry_id": "%s"}]}`, scenario.revision, scheduleEntryOneA),
		"wrong weight type": fmt.Sprintf(`{"snapshot_revision": %d, "updates": [{"group_id": 1, "entry_id": "%s", "weight": "5"}]}`, scenario.revision, scheduleEntryOneA),
	} {
		recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", payload, scenario.authKey)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("%s: response = %d %s, want 400", name, recorder.Code, recorder.Body.String())
		}
	}
	if scenario.publishCalls != 0 {
		t.Fatalf("publish calls after structural rejects = %d, want 0", scenario.publishCalls)
	}
}

func TestModelRouteSchedulePatchRejectsStaleSnapshotRevision(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)
	before := loadCreatedGroupModels(t, scenario.fixture, 1)

	stale := fmt.Sprintf(`{
		"snapshot_revision": %d,
		"updates": [{"group_id": 1, "entry_id": "%s", "weight": 77}]
	}`, scenario.revision-1, scheduleEntryOneA)
	recorder := scenario.perform(http.MethodPatch, "/api/model-route/schedule", stale, scenario.authKey)
	if code := decodeScheduleError(t, recorder, http.StatusConflict); code != modelRouteScheduleRevisionConflict.Code {
		t.Fatalf("stale revision code = %q", code)
	}
	if scenario.publishCalls != 0 {
		t.Fatalf("publish calls after 409 = %d, want 0", scenario.publishCalls)
	}
	after := loadCreatedGroupModels(t, scenario.fixture, 1)
	if len(after) != len(before) || after[0].Weight == nil || *after[0].Weight != *before[0].Weight {
		t.Fatalf("weights after 409 = %#v, want %#v", after, before)
	}

	// A matching revision succeeds and bumps the revision by exactly one.
	current := fmt.Sprintf(`{
		"snapshot_revision": %d,
		"updates": [{"group_id": 1, "entry_id": "%s", "weight": 77}]
	}`, scenario.fixture.manager.Current().Revision, scheduleEntryOneA)
	recorder = scenario.perform(http.MethodPatch, "/api/model-route/schedule", current, scenario.authKey)
	var result modelRouteSchedulePatchResponse
	decodeScheduleSuccess(t, recorder, &result)
	if scenario.publishCalls != 1 {
		t.Fatalf("publish calls after accepted patch = %d, want 1", scenario.publishCalls)
	}
	if result.SnapshotRevisionNew != scenario.revision+1 {
		t.Fatalf("snapshot_revision_new = %d, want %d", result.SnapshotRevisionNew, scenario.revision+1)
	}
	if result.Detail != nil {
		t.Fatalf("patch without echo context must omit detail = %#v", result.Detail)
	}
}

func TestModelRouteScheduleRecoverRejectsStaleFailureVersion(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)
	missing := scenario.perform(
		http.MethodPost, "/api/model-route/schedule/recover",
		fmt.Sprintf(`{"group_id": 1, "entry_id": "%s"}`, scheduleEntryOneA), scenario.authKey,
	)
	if code := decodeScheduleError(t, missing, http.StatusConflict); code != modelRouteScheduleRuntimeConflict.Code {
		t.Fatalf("missing failure version code = %q, want %q", code, modelRouteScheduleRuntimeConflict.Code)
	}
	key := state.RouteEntryKey{GroupID: 1, EntryID: scheduleEntryOneA}
	if _, changed := scenario.fixture.registry.SetEntryBlacklistedWithChange(key); !changed {
		t.Fatal("SetEntryBlacklistedWithChange() changed = false")
	}
	observed, ok := scenario.fixture.registry.EntryRuntime(key, scenario.now)
	if !ok {
		t.Fatal("EntryRuntime() observed = false")
	}
	if _, exists, _ := scenario.fixture.registry.RecordEntryFailureWithBlacklist(1, scheduleEntryOneA, 1, scenario.now.Add(time.Hour)); !exists {
		t.Fatal("RecordEntryFailureWithBlacklist() exists = false")
	}

	body := fmt.Sprintf(`{"group_id": 1, "entry_id": "%s", "failure_version": %d}`, scheduleEntryOneA, observed.FailureVersion)
	recorder := scenario.perform(http.MethodPost, "/api/model-route/schedule/recover", body, scenario.authKey)
	if code := decodeScheduleError(t, recorder, http.StatusConflict); code != modelRouteScheduleRuntimeConflict.Code {
		t.Fatalf("stale recovery code = %q, want %q", code, modelRouteScheduleRuntimeConflict.Code)
	}
	current, ok := scenario.fixture.registry.EntryRuntime(key, scenario.now)
	if !ok || !current.Blacklisted || current.FailureCount != 1 {
		t.Fatalf("runtime after stale recovery = %#v/%t, want unchanged", current, ok)
	}
}

func TestModelRouteScheduleRecoverClearsEntryRuntimeOnly(t *testing.T) {
	t.Parallel()
	scenario := newScheduleTestScenario(t)

	_, _ = scenario.fixture.registry.SetEntryCooldownForEntry(
		1, scheduleEntryOneA, scenario.now.Add(time.Hour),
	)
	if count, _ := scenario.fixture.registry.IncrEntryFailureForEntry(1, scheduleEntryOneA); count != 1 {
		t.Fatalf("IncrEntryFailureForEntry() = %d, want 1", count)
	}
	_, _ = scenario.fixture.registry.SetEntryCooldownForEntry(
		2, scheduleEntryTwoB, scenario.now.Add(time.Hour),
	)
	credentialCooldown := scenario.now.Add(45 * time.Minute)
	if !scenario.fixture.registry.SetCooldown(scenario.credentialID, credentialCooldown) {
		t.Fatal("SetCooldown() exists = false")
	}

	observed, ok := scenario.fixture.registry.EntryRuntime(
		state.RouteEntryKey{GroupID: 1, EntryID: scheduleEntryOneA}, scenario.now,
	)
	if !ok {
		t.Fatal("EntryRuntime() observed = false")
	}
	body := fmt.Sprintf(`{"group_id": 1, "entry_id": "%s", "failure_version": %d}`, scheduleEntryOneA, observed.FailureVersion)
	recorder := scenario.perform(http.MethodPost, "/api/model-route/schedule/recover", body, scenario.authKey)
	var result modelRouteScheduleRecoverResponse
	decodeScheduleSuccess(t, recorder, &result)
	if result.GroupID != 1 || result.EntryID != scheduleEntryOneA ||
		result.Runtime.State != state.EntryRuntimeAvailable ||
		result.Runtime.CooldownUntilMS != nil || result.Runtime.FailureCount != 0 ||
		result.Runtime.BlacklistReleaseAtMS != nil {
		t.Fatalf("recover response = %#v", result)
	}

	view, _ := scenario.fixture.registry.EntryRuntime(
		state.RouteEntryKey{GroupID: 1, EntryID: scheduleEntryOneA}, scenario.now,
	)
	if view.RuntimeState(scenario.now) != state.EntryRuntimeAvailable ||
		view.FailureCount != 0 || view.Blacklisted {
		t.Fatalf("recovered runtime = %#v", view)
	}
	sibling, _ := scenario.fixture.registry.EntryRuntime(
		state.RouteEntryKey{GroupID: 2, EntryID: scheduleEntryTwoB}, scenario.now,
	)
	if sibling.RuntimeState(scenario.now) != state.EntryRuntimeCooldown {
		t.Fatalf("unrelated entry runtime = %#v", sibling)
	}
	if until, ok := scenario.fixture.registry.CredentialCooldownUntil(scenario.credentialID); !ok ||
		!until.Equal(credentialCooldown) {
		t.Fatalf("credential runtime changed by entry recovery: %v %v", until, ok)
	}

	// A successful recovery increments the version; a later idempotent request
	// must carry the newly observed version rather than replaying stale proof.
	currentVersion, ok := scenario.fixture.registry.EntryRuntime(
		state.RouteEntryKey{GroupID: 1, EntryID: scheduleEntryOneA}, scenario.now,
	)
	if !ok {
		t.Fatal("EntryRuntime() after recovery = false")
	}
	body = fmt.Sprintf(`{"group_id": 1, "entry_id": "%s", "failure_version": %d}`, scheduleEntryOneA, currentVersion.FailureVersion)
	recorder = scenario.perform(http.MethodPost, "/api/model-route/schedule/recover", body, scenario.authKey)
	decodeScheduleSuccess(t, recorder, &result)
	if result.Runtime.State != state.EntryRuntimeAvailable || result.Runtime.FailureVersion == 0 {
		t.Fatalf("second recover = %#v", result.Runtime)
	}

	invalid := scenario.perform(
		http.MethodPost, "/api/model-route/schedule/recover",
		`{"group_id": 0, "entry_id": "e000000000001"}`, scenario.authKey,
	)
	if code := decodeScheduleError(t, invalid, http.StatusBadRequest); code != app_errors.ErrValidation.Code {
		t.Fatalf("invalid recover code = %q", code)
	}
}
