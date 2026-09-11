package control

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

const probeTestModel = "gpt-4o"

func createProbeGroup(t *testing.T, fixture serviceFixture, models []string) uint {
	t.Helper()
	name := fmt.Sprintf("model-probe-group-%d", testIdempotencySequence.Add(1))
	values := make([]GroupModel, 0, len(models))
	for _, model := range models {
		values = append(values, GroupModel{ID: model})
	}
	result, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: &name, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
		Models:      optionalGroupModels{Set: true, Values: values},
		Credentials: "model-probe-secret", ConnectionType: "api_key", ConfirmSameTarget: true,
	})
	if err != nil {
		t.Fatalf("CreateGroup(%q) error = %v", name, err)
	}
	return result.GroupID
}

func takeGroupCredential(t *testing.T, fixture serviceFixture, groupID uint) models.Credential {
	t.Helper()
	var credential models.Credential
	if err := fixture.db.Where("group_id = ?", groupID).Take(&credential).Error; err != nil {
		t.Fatalf("load credential of group %d: %v", groupID, err)
	}
	return credential
}

func TestModelProbe(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "model-probe-single-secret")
	fixture.service.executor = &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}

	response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{
		Targets: []ModelProbeTargetRequest{{GroupID: groupID, Model: probeTestModel}},
	})
	if err != nil {
		t.Fatalf("ProbeGroupModels() error = %v", err)
	}
	if len(response.Results) != 1 {
		t.Fatalf("results = %#v, want exactly one result", response.Results)
	}
	result := response.Results[0]
	if result.GroupID != groupID || result.Model != probeTestModel || result.GroupName == "" {
		t.Fatalf("result identity = %#v", result)
	}
	if result.Outcome != ProbeOutcomePassed || result.Reason != nil {
		t.Fatalf("outcome/reason = %q/%#v, want passed/nil", result.Outcome, result.Reason)
	}
	if result.Protocol == nil || result.RouteMode == nil || !result.RouteMode.Valid() {
		t.Fatalf("protocol/route_mode = %#v/%#v, want the executed target route", result.Protocol, result.RouteMode)
	}
	if result.StatusCode == nil || *result.StatusCode != http.StatusOK {
		t.Fatalf("status_code = %#v, want 200", result.StatusCode)
	}
	if result.LatencyMS == nil || *result.LatencyMS < 0 {
		t.Fatalf("latency_ms = %#v, want a measured duration", result.LatencyMS)
	}
	if result.CredentialID == nil || *result.CredentialID == 0 {
		t.Fatalf("credential_id = %#v, want the selected credential", result.CredentialID)
	}
	if result.CredentialLabel == nil || *result.CredentialLabel == "" {
		t.Fatalf("credential_label = %#v, want the masked credential identity", result.CredentialLabel)
	}
	if result.TestedAtMS <= 0 {
		t.Fatalf("tested_at_ms = %d, want a positive timestamp", result.TestedAtMS)
	}
	if result.LogID == nil {
		t.Fatal("log_id must be the durable request-log primary key of an executed probe")
	}

	events := fixture.probeSink.Events()
	if len(events) != 1 {
		t.Fatalf("probe log events = %d, want exactly one", len(events))
	}
	event := events[0]
	if event.RequestID != *result.LogID {
		t.Fatalf("log event id = %q, want reported log_id %q", event.RequestID, *result.LogID)
	}
	if event.Operation != execution.OperationProbe || event.AccessKeyID != 0 {
		t.Fatalf("log event operation/access_key = %q/%d, want probe/0", event.Operation, event.AccessKeyID)
	}
	if event.Status != telemetry.RequestStatusSuccess ||
		event.ModelConsistency != telemetry.ModelConsistencyUnknown {
		t.Fatalf("log event status/model_consistency = %q/%q, want success/unknown",
			event.Status, event.ModelConsistency)
	}
	if event.UpstreamModel != probeTestModel || event.ClientModel != probeTestModel ||
		event.UpstreamReportedModel != "" {
		t.Fatalf("log event model identity = %q/%q/%q", event.ClientModel, event.UpstreamModel, event.UpstreamReportedModel)
	}
	if len(event.Attempts) != 1 {
		t.Fatalf("log event attempts = %d, want exactly one executed attempt", len(event.Attempts))
	}
	attempt := event.Attempts[0]
	if attempt.FailureCategory != telemetry.FailureCategoryOK ||
		attempt.Action != telemetry.ActionTerminate || attempt.Effect != telemetry.EffectNone {
		t.Fatalf("attempt category/action/effect = %q/%q/%q, want ok/terminate/none",
			attempt.FailureCategory, attempt.Action, attempt.Effect)
	}
	if attempt.GroupID != groupID || attempt.CredentialID != *result.CredentialID ||
		attempt.UpstreamModel != probeTestModel {
		t.Fatalf("attempt attribution = %#v", attempt)
	}
	if attempt.DispatchState != execution.DispatchMaybeSent || attempt.StatusCode != http.StatusOK {
		t.Fatalf("attempt dispatch/status = %q/%d", attempt.DispatchState, attempt.StatusCode)
	}
	if event.Usage.GroupID != groupID || event.Usage.CredentialID != *result.CredentialID ||
		event.Usage.AttemptSequence != attempt.Sequence {
		t.Fatalf("usage attribution = %#v, want the returned attempt", event.Usage)
	}
	if event.Usage.Result.State != usage.StateNotApplicable ||
		event.Usage.Pricing.CostState != string(pricing.CostStateNotApplicable) ||
		event.Usage.Pricing.PricingCompleteness != string(pricing.CompletenessNotApplicable) ||
		event.Usage.Pricing.EstimatedCostNanoUSD != 0 {
		t.Fatalf("probe must not carry usage or cost: %#v", event.Usage)
	}
}

// TestModelProbeDisabledGroup pins the honest behaviour for a disabled group: it
// keeps a compiled view, so an explicit probe still executes and reports the real
// group name instead of the "#<id>" placeholder that reads as a deleted group.
func TestModelProbeDisabledGroup(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createProbeGroup(t, fixture, []string{probeTestModel})
	if _, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		Enabled: optionalField[bool]{Set: true, Value: false},
	}); err != nil {
		t.Fatalf("UpdateGroupSettings(disable) error = %v", err)
	}
	var group models.Group
	if err := fixture.db.First(&group, groupID).Error; err != nil {
		t.Fatalf("load group %d: %v", groupID, err)
	}
	if group.Enabled {
		t.Fatalf("group %d is still enabled after the disable update", groupID)
	}
	fixture.service.executor = &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}

	response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{
		Targets: []ModelProbeTargetRequest{{GroupID: groupID, Model: probeTestModel}},
	})
	if err != nil {
		t.Fatalf("ProbeGroupModels() error = %v", err)
	}
	if len(response.Results) != 1 {
		t.Fatalf("results = %#v, want exactly one result", response.Results)
	}
	result := response.Results[0]
	if result.Outcome != ProbeOutcomePassed || result.Reason != nil {
		t.Fatalf("disabled group outcome/reason = %q/%#v, want passed/nil: a disabled group must stay probeable",
			result.Outcome, result.Reason)
	}
	if result.GroupName != group.Name {
		t.Fatalf("group_name = %q, want the persisted name %q", result.GroupName, group.Name)
	}
	if result.LogID == nil {
		t.Fatal("log_id must be the durable request-log primary key of an executed probe")
	}
}

func TestModelProbeBatch(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "model-probe-batch-secret")
	emptyGroupID := createProbeGroup(t, fixture, []string{probeTestModel})
	credential := takeGroupCredential(t, fixture, emptyGroupID)
	if !fixture.registry.SetCooldown(credential.ID, time.Now().Add(time.Hour)) {
		t.Fatal("SetCooldown() = false")
	}
	fixture.service.executor = &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}

	response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{
		Targets: []ModelProbeTargetRequest{
			{GroupID: groupID, Model: probeTestModel},
			{GroupID: 999_999, Model: probeTestModel},
			{GroupID: groupID, Model: "not-configured"},
			{GroupID: groupID, Model: probeTestModel},
			{GroupID: emptyGroupID, Model: probeTestModel},
		},
	})
	if err != nil {
		t.Fatalf("ProbeGroupModels() error = %v", err)
	}
	// Results follow deduplicated input order, and one target problem never fails
	// the whole batch.
	if len(response.Results) != 4 {
		t.Fatalf("results = %d, want 4 after deduplication", len(response.Results))
	}
	want := []struct {
		groupID uint
		reason  *ProbeReason
	}{
		{groupID: groupID},
		{groupID: 999_999, reason: probeReasonOf(ProbeReasonTargetUnavailable)},
		{groupID: groupID, reason: probeReasonOf(ProbeReasonTargetUnavailable)},
		{groupID: emptyGroupID, reason: probeReasonOf(ProbeReasonNoSchedulableCredential)},
	}
	for index, expected := range want {
		result := response.Results[index]
		if result.GroupID != expected.groupID {
			t.Fatalf("results[%d].group_id = %d, want %d", index, result.GroupID, expected.groupID)
		}
		if !probeReasonEqual(result.Reason, expected.reason) {
			t.Fatalf("results[%d].reason = %#v, want %#v", index, result.Reason, expected.reason)
		}
		if expected.reason == nil {
			if result.Outcome != ProbeOutcomePassed || result.LogID == nil {
				t.Fatalf("results[%d] = %#v, want an executed passing target", index, result)
			}
			continue
		}
		if result.Outcome != ProbeOutcomeInconclusive {
			t.Fatalf("results[%d].outcome = %q, want inconclusive for a target that never executed", index, result.Outcome)
		}
		if result.LogID != nil {
			t.Fatalf("results[%d].log_id = %q, want null when nothing was dispatched", index, *result.LogID)
		}
	}
	// The credential-less group still reports the route it would have probed.
	blocked := response.Results[3]
	if blocked.Protocol == nil || blocked.RouteMode == nil {
		t.Fatalf("no_schedulable_credential result = %#v, want the resolved probe route", blocked)
	}
	if blocked.CredentialID != nil || blocked.StatusCode != nil || blocked.LatencyMS != nil {
		t.Fatalf("no_schedulable_credential result = %#v, want no execution evidence", blocked)
	}

	tests := []struct {
		name    string
		request ModelProbeRequest
	}{
		{name: "no targets", request: ModelProbeRequest{}},
		{
			name: "too many targets",
			request: ModelProbeRequest{Targets: repeatProbeTargets(
				groupID,
				probeTestModel,
				modelProbeMaxTargets+1,
			)},
		},
		{
			name: "missing group id",
			request: ModelProbeRequest{Targets: []ModelProbeTargetRequest{
				{Model: probeTestModel},
			}},
		},
		{
			name: "untrimmed model",
			request: ModelProbeRequest{Targets: []ModelProbeTargetRequest{
				{GroupID: groupID, Model: " " + probeTestModel},
			}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := fixture.service.ProbeGroupModels(t.Context(), test.request); !errors.Is(err, app_errors.ErrBadRequest) {
				t.Fatalf("ProbeGroupModels() error = %v, want ErrBadRequest", err)
			}
		})
	}
}

func repeatProbeTargets(groupID uint, model string, count int) []ModelProbeTargetRequest {
	targets := make([]ModelProbeTargetRequest, 0, count)
	for range count {
		targets = append(targets, ModelProbeTargetRequest{GroupID: groupID, Model: model})
	}
	return targets
}

func probeReasonOf(reason ProbeReason) *ProbeReason {
	return &reason
}

func probeReasonEqual(left *ProbeReason, right *ProbeReason) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}

func TestModelProbeNoStateChange(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "model-probe-state-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	observedAt := time.Now()
	beforeEntries, err := fixture.registry.SnapshotGroupCredentialEntriesExact(groupID, []uint{credential.ID})
	if err != nil {
		t.Fatal(err)
	}
	beforeStats := fixture.stats.Snapshot(credential.ID, observedAt)
	beforeDirty := len(fixture.accessQuota.DirtySnapshots(1))
	fixture.service.executor = &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}

	if _, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{
		Targets: []ModelProbeTargetRequest{{GroupID: groupID, Model: probeTestModel}},
	}); err != nil {
		t.Fatalf("ProbeGroupModels() error = %v", err)
	}

	afterEntries, err := fixture.registry.SnapshotGroupCredentialEntriesExact(groupID, []uint{credential.ID})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(afterEntries, beforeEntries) {
		t.Fatalf("probing mutated credential runtime state:\nbefore=%#v\nafter=%#v", beforeEntries, afterEntries)
	}
	if afterStats := fixture.stats.Snapshot(credential.ID, observedAt); !reflect.DeepEqual(afterStats, beforeStats) {
		t.Fatalf("probing mutated credential health statistics: before=%#v after=%#v", beforeStats, afterStats)
	}
	if afterDirty := len(fixture.accessQuota.DirtySnapshots(1)); afterDirty != beforeDirty {
		t.Fatalf("probing marked access-key quota state dirty: before=%d after=%d", beforeDirty, afterDirty)
	}
}

type probeConcurrencyTracker struct {
	mu    sync.Mutex
	live  int
	peak  int
	total int
}

func (tracker *probeConcurrencyTracker) enter() {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	tracker.live++
	tracker.total++
	if tracker.live > tracker.peak {
		tracker.peak = tracker.live
	}
}

func (tracker *probeConcurrencyTracker) leave() {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	tracker.live--
}

func (tracker *probeConcurrencyTracker) peakAndTotal() (int, int) {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	return tracker.peak, tracker.total
}

func TestModelProbeConcurrency(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	models := make([]string, 0, 8)
	for index := range 8 {
		models = append(models, fmt.Sprintf("probe-model-%d", index))
	}
	groupID := createProbeGroup(t, fixture, models)
	tracker := &probeConcurrencyTracker{}
	fixture.service.executor = &credentialProbeTestExecutor{execute: func(execution.AttemptSpec) execution.AttemptResult {
		tracker.enter()
		time.Sleep(20 * time.Millisecond)
		tracker.leave()
		return successfulCredentialProbeResult()
	}}

	targets := make([]ModelProbeTargetRequest, 0, len(models))
	for _, model := range models {
		targets = append(targets, ModelProbeTargetRequest{GroupID: groupID, Model: model})
	}
	response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{Targets: targets})
	if err != nil {
		t.Fatalf("ProbeGroupModels() error = %v", err)
	}
	if len(response.Results) != len(models) {
		t.Fatalf("results = %d, want %d", len(response.Results), len(models))
	}
	for index, result := range response.Results {
		if result.Outcome != ProbeOutcomePassed {
			t.Fatalf("results[%d] = %#v, want passed", index, result)
		}
	}
	peak, total := tracker.peakAndTotal()
	if total != len(models) {
		t.Fatalf("executed probes = %d, want %d", total, len(models))
	}
	if peak > modelProbeConcurrency {
		t.Fatalf("probe concurrency peak = %d, want <= %d", peak, modelProbeConcurrency)
	}
	if peak < 2 {
		t.Fatalf("probe concurrency peak = %d, want the batch to run concurrently", peak)
	}
}

func TestCredentialProbeLog(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "credential-probe-log-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	fixture.service.executor = &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}

	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil {
		t.Fatalf("TestGroupCredential() error = %v", err)
	}
	if response.Outcome != ProbeOutcomePassed || response.LogID == nil {
		t.Fatalf("credential probe response = %#v, want a passed probe with a log id", response)
	}
	events := fixture.probeSink.Events()
	if len(events) != 1 {
		t.Fatalf("credential probe log events = %d, want exactly one", len(events))
	}
	event := events[0]
	if event.RequestID != *response.LogID {
		t.Fatalf("log event id = %q, want reported log_id %q", event.RequestID, *response.LogID)
	}
	if event.Operation != execution.OperationProbe || event.AccessKeyID != 0 ||
		event.Status != telemetry.RequestStatusSuccess {
		t.Fatalf("credential probe log event = %#v", event)
	}
	if len(event.Attempts) != 1 || event.Attempts[0].CredentialID != credential.ID {
		t.Fatalf("credential probe attempts = %#v", event.Attempts)
	}
}

// TestModelProbeResponseContract pins the response key set, because the web
// projector asserts that set exactly (web/src/app/resources/model-probe.ts): a
// field added on one side only fails at runtime instead of failing a build.
func TestModelProbeResponseContract(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "model-probe-contract-secret")
	fixture.service.executor = &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}

	response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{
		Targets: []ModelProbeTargetRequest{
			{GroupID: groupID, Model: probeTestModel},
			{GroupID: 999_999, Model: probeTestModel},
		},
	})
	if err != nil {
		t.Fatalf("ProbeGroupModels() error = %v", err)
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("marshal probe response: %v", err)
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &payload); err != nil {
		t.Fatalf("decode probe response: %v", err)
	}
	assertJSONKeys(t, payload, []string{"results"})
	var results []map[string]json.RawMessage
	if err := json.Unmarshal(payload["results"], &results); err != nil {
		t.Fatalf("decode probe results: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results = %d, want 2", len(results))
	}
	wantFields := []string{
		"group_id",
		"group_name",
		"model",
		"outcome",
		"reason",
		"protocol",
		"route_mode",
		"status_code",
		"latency_ms",
		"credential_id",
		"credential_label",
		"log_id",
		"tested_at_ms",
	}
	for index, result := range results {
		assertJSONKeys(t, result, wantFields)
		if index == 1 {
			continue
		}
		for _, field := range []string{"reason", "log_id"} {
			if string(result[field]) == "" {
				t.Fatalf("executed target lost field %q", field)
			}
		}
	}
	// A target that never dispatched must not claim execution evidence.
	for _, field := range []string{
		"protocol", "route_mode", "status_code", "latency_ms",
		"credential_id", "credential_label", "log_id",
	} {
		if string(results[1][field]) != "null" {
			t.Fatalf("unexecuted target %s = %s, want null", field, results[1][field])
		}
	}
}

func assertJSONKeys(t *testing.T, payload map[string]json.RawMessage, want []string) {
	t.Helper()
	got := make([]string, 0, len(payload))
	for key := range payload {
		got = append(got, key)
	}
	sort.Strings(got)
	expected := append([]string(nil), want...)
	sort.Strings(expected)
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("response field set = %v, want %v", got, expected)
	}
}

// probeLogObservationForSummary builds the probe log observation of a single
// 403 attempt so the row/attempt summary mapping is asserted in isolation.
func probeLogObservationForSummary(result execution.AttemptResult, completedAt time.Time) probeLogObservation {
	return probeLogObservation{
		group:      state.GroupView{ID: 7, Name: "probe-log-group", ChannelID: channel.OpenAI},
		model:      probeTestModel,
		credential: state.CredentialRef{ID: 11},
		executed: credentialProbeExecution{
			requestID: "probe-log-summary",
			result:    result,
			latency:   120 * time.Millisecond,
			protocol:  protocol.OpenAICompletions,
			attempts: []credentialProbeAttempt{{
				sequence: 1, routeMode: execution.RouteNative,
				completedAt: completedAt, duration: 120 * time.Millisecond, result: result,
			}},
		},
		evidence:    classifyCredentialProbeEvidence(result),
		completedAt: completedAt,
	}
}

// A 403 whose body names the real cause must reach the log verbatim: the reason
// code alone cannot tell "this token may not use that model" apart from an edge
// WAF block or an account ban.
func TestProbeRequestLogKeepsUpstreamErrorSummary(t *testing.T) {
	t.Parallel()
	const upstreamMessage = "该令牌无权使用模型 claude-opus-4-8"
	result := failedCredentialProbeResult(http.StatusForbidden, execution.ErrorKindHTTP, "")
	result.Body = []byte(`{"error":{"message":"` + upstreamMessage + `"}}`)
	result.Error.Summary = upstreamMessage

	event := buildProbeRequestEvent(probeLogObservationForSummary(
		result,
		time.Date(2026, time.September, 1, 10, 0, 0, 0, time.UTC),
	))

	if event.Status != telemetry.RequestStatusError || event.ErrorCode != "upstream_error" {
		t.Fatalf("event status/code = %q/%q, want error/upstream_error", event.Status, event.ErrorCode)
	}
	if event.ErrorSummary != upstreamMessage {
		t.Fatalf("event error summary = %q, want the upstream message", event.ErrorSummary)
	}
	if len(event.Attempts) != 1 {
		t.Fatalf("attempts = %d, want 1", len(event.Attempts))
	}
	if event.Attempts[0].ErrorSummary != upstreamMessage {
		t.Fatalf("attempt error summary = %q, want the upstream message", event.Attempts[0].ErrorSummary)
	}
	if event.Attempts[0].ErrorCode != "upstream_error" {
		t.Fatalf("attempt error code = %q, want upstream_error", event.Attempts[0].ErrorCode)
	}
}

// Without judgeable provider evidence the reason code stays the row summary, so
// the log list never renders an error with an empty message.
func TestProbeRequestLogFallsBackToReasonCodeWithoutUpstreamMessage(t *testing.T) {
	t.Parallel()
	result := execution.AttemptResult{
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusForbidden,
		Header:          http.Header{},
	}

	event := buildProbeRequestEvent(probeLogObservationForSummary(
		result,
		time.Date(2026, time.September, 1, 10, 0, 0, 0, time.UTC),
	))

	if event.ErrorCode != "unknown" {
		t.Fatalf("event error code = %q, want unknown", event.ErrorCode)
	}
	if event.ErrorSummary != "unknown" {
		t.Fatalf("event error summary = %q, want the reason code fallback", event.ErrorSummary)
	}
	if event.Attempts[0].ErrorSummary != "" {
		t.Fatalf("attempt error summary = %q, want empty", event.Attempts[0].ErrorSummary)
	}
}

// A group header rule can carry a literal credential that upstream echoes back
// inside its error body, so the probe log must redact it like the gateway does.
func TestProbeRequestLogRedactsHeaderRuleLiterals(t *testing.T) {
	t.Parallel()
	const headerSecret = "hb-live-9f3c2a7d5e1b4c60"
	result := failedCredentialProbeResult(http.StatusForbidden, execution.ErrorKindHTTP, "")
	result.Error.Summary = "invalid credential " + headerSecret
	observation := probeLogObservationForSummary(
		result,
		time.Date(2026, time.September, 1, 10, 0, 0, 0, time.UTC),
	)
	observation.group.HeaderRules = state.HeaderRules{
		Set: map[string]string{"x-probe-auth": headerSecret},
	}

	event := buildProbeRequestEvent(observation)

	if strings.Contains(event.ErrorSummary, headerSecret) {
		t.Fatalf("event error summary leaked a header rule literal: %q", event.ErrorSummary)
	}
	if strings.Contains(event.Attempts[0].ErrorSummary, headerSecret) {
		t.Fatalf("attempt error summary leaked a header rule literal: %q", event.Attempts[0].ErrorSummary)
	}
}

func TestModelProbeRouteContract(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	module := NewServer(&config.Config{AuthKey: "model-probe-route-auth"}, fixture.service).HTTPModule()
	found := false
	for _, route := range module.Routes {
		if route.Name != "control.model-probe.run" {
			continue
		}
		found = true
		if !reflect.DeepEqual(route.Methods, []string{http.MethodPost}) || route.Path != "/model-probe" {
			t.Fatalf("model probe route = %#v", route)
		}
	}
	if !found {
		t.Fatal("model probe route is missing")
	}
	// A probe spends real upstream requests, so it must stay an admin-only
	// control route: access keys may not reach it.
	if _, exists := accessKeyControlRoutes["/api/model-probe"]; exists {
		t.Fatal("accessKeyControlRoutes must not be widened to the model probe route")
	}
}
