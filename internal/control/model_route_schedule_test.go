// Contract tests for the central model-route schedule management API
// (design docs/design/model-route-central-scheduling.md §5, U004). All
// candidate assertions run against the real inspection kernel via published
// snapshots and HTTP endpoints, not hand-built response fixtures.
package control

import (
	"encoding/json"
	"fmt"
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
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
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

	byModel := make(map[string]modelRouteScheduleIndexItem, len(result.Items))
	for _, item := range result.Items {
		byModel[item.ExternalModel] = item
	}
	if len(result.Items) != 2 {
		t.Fatalf("items = %#v, want 2 external models", result.Items)
	}
	pub := byModel["pub"]
	if pub.ExternalModel != "pub" || pub.CandidateCount != 4 || pub.GroupCount != 2 ||
		!pub.HasFallback || pub.CooledCandidates != 0 || pub.BlacklistedCandidates != 0 {
		t.Fatalf("pub item = %#v", pub)
	}
	solo := byModel["solo-model"]
	if solo.CandidateCount != 1 || solo.GroupCount != 1 || solo.HasFallback {
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
		result.RouteStrategy != state.RouteStrategyNativeFirst ||
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
		first.GroupWeight != nil || len(first.Entries) != 3 {
		t.Fatalf("group one = %#v", first)
	}
	second := result.Groups[1]
	if second.GroupID != 2 || len(second.Entries) != 1 {
		t.Fatalf("group two = %#v", second)
	}

	entryA := first.Entries[0]
	if entryA.EntryID != scheduleEntryOneA || entryA.ModelID != "up-a" ||
		entryA.Alias != "pub" || entryA.WeightManual == nil || *entryA.WeightManual != scheduleWeightOneA ||
		entryA.Weight != scheduleWeightOneA || entryA.Priority != 1 || entryA.Fallback {
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
	// through the same entry runtime the inspection kernel consumed.
	_, _ = scenario.fixture.registry.IncrEntryFailureForEntry(1, scheduleEntryOneA)
	_, _ = scenario.fixture.registry.SetEntryCooldownForEntry(
		1, scheduleEntryOneB, scenario.now.Add(30*time.Minute),
	)
	_, _ = scenario.fixture.registry.SetEntryBlacklistedForEntry(2, scheduleEntryTwoB)
	runtime := scenario.perform(http.MethodGet, path, "", scenario.authKey)
	var runtimeResult modelRouteScheduleDetailResponse
	decodeScheduleSuccess(t, runtime, &runtimeResult)
	first = runtimeResult.Groups[0]
	second = runtimeResult.Groups[1]
	if first.Entries[0].Runtime.FailureCount != 1 ||
		first.Entries[0].Runtime.State != state.EntryRuntimeAvailable {
		t.Fatalf("up-a runtime after failure = %#v", first.Entries[0].Runtime)
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
		blacklisted.Runtime.CooldownUntilMS != nil {
		t.Fatalf("two/up-b runtime after blacklist = %#v", blacklisted.Runtime)
	}
	assertScheduleReason(t, blacklisted.ReasonCode, scheduler.ReasonEntryBlacklisted)
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

	invalid := scenario.perform(
		http.MethodGet,
		"/api/model-route/schedule/detail?external_model=pub&protocol=invalid&access_key_id=1",
		"", scenario.authKey,
	)
	if code := decodeScheduleError(t, invalid, http.StatusBadRequest); code != app_errors.ErrValidation.Code {
		t.Fatalf("invalid protocol code = %q", code)
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
		detail.AccessKey.ID != scenario.accessKeyID ||
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

	body := fmt.Sprintf(`{"group_id": 1, "entry_id": "%s"}`, scheduleEntryOneA)
	recorder := scenario.perform(http.MethodPost, "/api/model-route/schedule/recover", body, scenario.authKey)
	var result modelRouteScheduleRecoverResponse
	decodeScheduleSuccess(t, recorder, &result)
	if result.GroupID != 1 || result.EntryID != scheduleEntryOneA ||
		result.Runtime.State != state.EntryRuntimeAvailable ||
		result.Runtime.CooldownUntilMS != nil || result.Runtime.FailureCount != 0 {
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

	// Idempotent: recovering again keeps the cleared state.
	recorder = scenario.perform(http.MethodPost, "/api/model-route/schedule/recover", body, scenario.authKey)
	decodeScheduleSuccess(t, recorder, &result)
	if result.Runtime.State != state.EntryRuntimeAvailable {
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
