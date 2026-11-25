package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	lmfpd "github.com/Ikaleio/lm-detector/sdk/go"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/pricing"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func TestModelTraceRunDeadline(t *testing.T) {
	if got, want := modelTraceRunDeadline, 10*time.Minute; got != want {
		t.Fatalf("ModelTrace run deadline = %s, want %s", got, want)
	}
}

// modelTraceChallengeExecutor answers each challenge with a distinct integer
// sequence, so a test can prove that every challenge was dispatched as its own
// formal model request.
type modelTraceChallengeExecutor struct {
	mu sync.Mutex

	prompts       []string
	outputBudgets []int
	sequence      int
	fail          func(index int) bool
	execute       func(spec execution.AttemptSpec) execution.AttemptResult
	// usage is the usage evidence a successful challenge reports, so a test can
	// assert that a challenge is priced like formal traffic.
	usage        *execution.UsageEvidence
	contextUsage *execution.UsageEvidence
}

func (executor *modelTraceChallengeExecutor) Execute(
	_ context.Context,
	spec execution.AttemptSpec,
) execution.AttemptResult {
	executor.mu.Lock()
	executor.prompts = append(executor.prompts, spec.PromptText)
	executor.outputBudgets = append(executor.outputBudgets, spec.ProbeMaxOutputTokens)
	execute := executor.execute
	reportedUsage := executor.usage
	if spec.PromptText == modelTraceContextPrompt {
		if executor.contextUsage != nil {
			reportedUsage = executor.contextUsage
		}
		executor.mu.Unlock()
		if execute != nil {
			return execute(spec)
		}
		return execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
			StatusCode: http.StatusOK, Header: http.Header{}, GeneratedText: "hello", Usage: reportedUsage}
	}
	index := executor.sequence
	executor.sequence++
	fail := executor.fail
	failChallenge := fail != nil && fail(index)
	executor.mu.Unlock()
	if execute != nil {
		return execute(spec)
	}
	if failChallenge {
		return execution.AttemptResult{
			DispatchState:   execution.DispatchMaybeSent,
			ResponseStarted: true,
			StatusCode:      http.StatusTooManyRequests,
			Header:          http.Header{},
			Error: &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, StatusCode: http.StatusTooManyRequests,
				Summary: "upstream rate limited the challenge",
			},
		}
	}
	numbers := make([]string, 0, 300)
	for value := 0; value < 300; value++ {
		numbers = append(numbers, fmt.Sprintf("%d", (value*7+index*13)%355+1))
	}
	return execution.AttemptResult{
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusOK,
		Header:          http.Header{},
		GeneratedText:   strings.Join(numbers, " "),
		Body: []byte(`{"choices":[{"message":{"content":"` +
			strings.Join(numbers, " ") + `"}}]}`),
		Usage: reportedUsage,
	}
}

func (*modelTraceChallengeExecutor) ExecuteStream(
	context.Context,
	execution.AttemptSpec,
	execution.StreamSink,
) execution.StreamResult {
	panic("unexpected stream execution")
}

func (executor *modelTraceChallengeExecutor) recordedPrompts() []string {
	executor.mu.Lock()
	defer executor.mu.Unlock()
	return append([]string(nil), executor.prompts...)
}

func (executor *modelTraceChallengeExecutor) recordedOutputBudgets() []int {
	executor.mu.Lock()
	defer executor.mu.Unlock()
	return append([]int(nil), executor.outputBudgets...)
}

// useModelTraceConfig installs a process configuration for one test.
func useModelTraceConfig(t *testing.T, cfg *ModelTraceConfig) {
	t.Helper()
	previous := loadModelTraceConfig()
	t.Cleanup(func() { storeModelTraceConfig(previous) })
	storeModelTraceConfig(cfg)
}

func modelTraceTestRequest(groupID uint) ModelTraceRequest {
	return ModelTraceRequest{GroupID: groupID, Model: probeTestModel}
}

func TestModelTraceProbeRecordsEveryChallengeAsFormalRequest(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-challenge-secret")
	executor := &modelTraceChallengeExecutor{contextUsage: &execution.UsageEvidence{Normalized: usage.Result{
		State:  usage.StateComplete,
		Tokens: usage.Tokens{UncachedInput: 10, CacheRead: 24, Output: 100},
	}}}
	fixture.service.executor = executor

	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	if len(response.Challenges) != modelTraceChallengeCount {
		t.Fatalf("challenges = %d, want %d", len(response.Challenges), modelTraceChallengeCount)
	}
	if response.Analysis == nil {
		t.Fatal("analysis is nil")
	}
	if response.Analysis.UsedOutputs != modelTraceChallengeCount {
		t.Fatalf("used_outputs = %d, want %d", response.Analysis.UsedOutputs, modelTraceChallengeCount)
	}

	// Every challenge is its own upstream request and its own request-log row.
	prompts := executor.recordedPrompts()
	if len(prompts) != modelTraceChallengeCount+1 {
		t.Fatalf("executed requests = %d, want %d including the context check", len(prompts), modelTraceChallengeCount+1)
	}
	for i, budget := range executor.recordedOutputBudgets() {
		want := 4096
		if prompts[i] == modelTraceContextPrompt {
			want = 256
		}
		if budget != want {
			t.Errorf("output budget = %d, want %d", budget, want)
		}
	}
	events := fixture.probeSink.Events()
	if len(events) != modelTraceChallengeCount+1 {
		t.Fatalf("request log events = %d, want one per upstream request", len(events))
	}
	logged := make(map[string]telemetry.RequestEvent, len(events))
	for _, event := range events {
		if event.Operation != execution.OperationModelTrace {
			t.Fatalf("log event operation = %q, want %q", event.Operation, execution.OperationModelTrace)
		}
		if event.Status != telemetry.RequestStatusSuccess || event.ErrorCode != "" {
			t.Fatalf("log event status = %q/%q, want success", event.Status, event.ErrorCode)
		}
		if event.AccessKeyID != 0 {
			t.Fatalf("log event access key = %d, want the control plane (0)", event.AccessKeyID)
		}
		if event.Usage.GroupID != groupID || event.Usage.CredentialID == 0 ||
			event.Usage.AttemptSequence == 0 {
			t.Fatalf("log event usage attribution = %#v, want the executed route", event.Usage)
		}
		if len(event.Attempts) != 1 {
			t.Fatalf("log event attempts = %d, want 1", len(event.Attempts))
		}
		if event.Attempts[0].FailureCategory != telemetry.FailureCategoryOK ||
			event.Attempts[0].Action != telemetry.ActionTerminate {
			t.Fatalf("log attempt = %#v, want a terminating ok attempt", event.Attempts[0])
		}
		logged[event.RequestID] = event
	}
	check := response.ContextCheck
	if check.Status != modelTraceContextSuspected || check.ExpectedInputTokens == nil || *check.ExpectedInputTokens != 6 ||
		check.ReportedInputTokens == nil || *check.ReportedInputTokens != 34 || check.Delta == nil || *check.Delta != 28 || check.LogID == nil {
		t.Fatalf("context check = %#v, want suspected extra context with a linked request", check)
	}
	contextEvent, ok := logged[*check.LogID]
	if !ok || contextEvent.Usage.Result != executor.contextUsage.Normalized {
		t.Fatalf("context request log = %#v, want its actual usage", contextEvent)
	}
	for _, challenge := range response.Challenges {
		if challenge.LogID == nil {
			t.Fatalf("challenge %q has no log id", challenge.ID)
		}
		event, ok := logged[*challenge.LogID]
		if !ok {
			t.Fatalf("challenge %q log id %q has no request-log row", challenge.ID, *challenge.LogID)
		}
		if event.RequestID == *check.LogID || event.Usage.CredentialID != contextEvent.Usage.CredentialID ||
			event.Protocol != contextEvent.Protocol || event.Attempts[0].RouteMode != contextEvent.Attempts[0].RouteMode {
			t.Fatalf("context check must have its own request ID and share the challenge route and credential")
		}
		if event.ClientModel != probeTestModel || event.UpstreamModel != probeTestModel {
			t.Fatalf("challenge %q logged model = %q/%q", challenge.ID, event.ClientModel, event.UpstreamModel)
		}
		if challenge.Outcome != modelTraceChallengeSucceeded {
			t.Fatalf("challenge %q outcome = %q, want succeeded", challenge.ID, challenge.Outcome)
		}
	}

}

func TestModelTraceProbeKeepsPartialOutputsAndLogsFailures(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-partial-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{
		fail: func(index int) bool { return index == 1 },
	}

	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	failed := 0
	for _, challenge := range response.Challenges {
		if challenge.Outcome == modelTraceChallengeFailed {
			failed++
			if challenge.LogID == nil {
				t.Fatalf("failed challenge %q must keep its request-log row", challenge.ID)
			}
			if challenge.Reason == nil {
				t.Fatalf("failed challenge %q has no reason", challenge.ID)
			}
		}
	}
	if failed != 1 {
		t.Fatalf("failed challenges = %d, want 1", failed)
	}
	events := fixture.probeSink.Events()
	if len(events) != modelTraceChallengeCount+1 {
		t.Fatalf("request log events = %d, want one per challenge", len(events))
	}
	upstreamFailures := 0
	for _, event := range events {
		if event.Status == telemetry.RequestStatusError {
			upstreamFailures++
		}
	}
	if upstreamFailures != 1 {
		t.Fatalf("error request-log rows = %d, want the failed challenge", upstreamFailures)
	}

	if response.Analysis == nil || response.Analysis.UsedOutputs != modelTraceChallengeCount-1 {
		t.Fatalf("analysis = %#v, want calibration over the accepted outputs", response.Analysis)
	}
}

func TestModelTraceDiagnosticsIdentifyOriginalChallenge(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "diagnostic-index-secret")
	previous := generateModelTraceChallenges
	t.Cleanup(func() { generateModelTraceChallenges = previous })
	generateModelTraceChallenges = func(int) ([]lmfpd.Challenge, error) {
		return []lmfpd.Challenge{{ID: "first", Prompt: "first", ExpectedCount: 300}, {ID: "second", Prompt: "second", ExpectedCount: 300}, {ID: "third", Prompt: "third", ExpectedCount: 300}}, nil
	}
	numbers := make([]string, 300)
	for i := range numbers {
		numbers[i] = fmt.Sprint(i + 1)
	}
	answer := strings.Join(numbers, " ")
	fixture.service.executor = &modelTraceChallengeExecutor{execute: func(spec execution.AttemptSpec) execution.AttemptResult {
		if spec.PromptText == "first" {
			return execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusTooManyRequests, Header: http.Header{}, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindHTTP, StatusCode: http.StatusTooManyRequests}}
		}
		return execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Header: http.Header{}, GeneratedText: answer}
	}}
	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatal(err)
	}
	if response.Challenges[0].Outcome != modelTraceChallengeFailed || response.Analysis == nil || response.Analysis.UsedOutputs != 2 || len(response.Analysis.Diagnostics) != 2 {
		t.Fatalf("response = %#v, want failed first challenge and two real outputs", response)
	}
	detector, err := getModelTraceDetector(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	expected, err := detector.Analyze([]lmfpd.Output{{Text: answer, ExpectedCount: 300}, {Text: answer, ExpectedCount: 300}}, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := range expected.Diagnostics {
		expected.Diagnostics[i].Index = i + 1
	}
	if !reflect.DeepEqual(response.Analysis, expected) {
		t.Fatalf("analysis differs from SDK with mapped indices: got %#v, want %#v", response.Analysis, expected)
	}
}

func TestModelTraceProbeFailsClearlyWithoutUsableOutputs(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-allfailed-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{
		fail: func(int) bool { return true },
		contextUsage: &execution.UsageEvidence{Normalized: usage.Result{
			State: usage.StateComplete, Tokens: usage.Tokens{UncachedInput: 10},
		}},
	}

	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v, want a reported probe result", err)
	}
	if response.Analysis != nil {
		t.Fatalf("analysis = %#v, want null when no challenge produced an answer", response.Analysis)
	}
	if response.Failure == nil || *response.Failure != modelTraceFailureNoOutputs {
		t.Fatalf("failure = %#v, want %q", response.Failure, modelTraceFailureNoOutputs)
	}
	if response.ContextCheck.Status != modelTraceContextNoAnomaly {
		t.Fatalf("context check = %#v, want a result even when attribution fails", response.ContextCheck)
	}
	if len(fixture.probeSink.Events()) != modelTraceChallengeCount+1 {
		t.Fatalf("request log events = %d, want one row per request",
			len(fixture.probeSink.Events()))
	}
}

// modelTracePriceTable builds the price table one channel/model route is
// priced against. Tests use it to give a challenge a real price and then move
// the published table underneath the run.
func modelTracePriceTable(t *testing.T, channelID string, input int64) *pricing.Table {
	t.Helper()
	table, err := pricing.NewTable([]pricing.Rule{{
		Identity: pricing.Identity{ChannelID: channelID, ModelID: probeTestModel},
		Prices: pricing.Prices{
			Input:  pricing.Price{NanoUSDPerMillion: pricing.NanoUSD(input), Set: true},
			Output: pricing.Price{NanoUSDPerMillion: pricing.NanoUSD(input), Set: true},
		},
	}})
	if err != nil {
		t.Fatalf("pricing.NewTable() error = %v", err)
	}
	return table
}

// TestModelTraceProbeFreezesPriceTableBeforeDispatch proves a challenge is
// priced like formal traffic: the table is captured before the request is
// dispatched, so a price publication that lands while the challenge is in
// flight cannot retroactively re-price it.
func TestModelTraceProbeFreezesPriceTableBeforeDispatch(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-frozen-price-secret")

	frozenInput := int64(2_000_000_000)
	inFlightInput := int64(9_000_000_000)
	fixture.priceRuntime.Publish(modelTracePriceTable(t, string(channel.OpenAI), frozenInput))
	executor := &modelTraceChallengeExecutor{
		usage: &execution.UsageEvidence{Normalized: usage.Result{
			State:  usage.StateComplete,
			Tokens: usage.Tokens{UncachedInput: 1_000_000, Output: 1_000_000},
		}},
		// The price publication happens after dispatch has started, exactly like
		// an operator editing prices while a request is in flight.
		execute: func(spec execution.AttemptSpec) execution.AttemptResult {
			fixture.priceRuntime.Publish(modelTracePriceTable(t, string(channel.OpenAI), inFlightInput))
			return execution.AttemptResult{
				DispatchState:   execution.DispatchMaybeSent,
				ResponseStarted: true,
				StatusCode:      http.StatusOK,
				Header:          http.Header{},
				GeneratedText:   "1 2 3 4",
				Usage: &execution.UsageEvidence{
					Normalized: usage.Result{
						State:  usage.StateComplete,
						Tokens: usage.Tokens{UncachedInput: 1_000_000, Output: 1_000_000},
					},
				},
			}
		},
	}
	fixture.service.executor = executor

	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	if len(response.Challenges) == 0 {
		t.Fatal("no challenges executed")
	}
	events := fixture.probeSink.Events()
	if len(events) != len(response.Challenges)+1 {
		t.Fatalf("request log events = %d, want one per request", len(events))
	}
	for _, event := range events {
		observation := event.Usage
		if observation.Result.State != usage.StateComplete {
			t.Fatalf("usage state = %q, want the captured provider usage", observation.Result.State)
		}
		wantCost := int64(frozenInput * 2)
		if observation.Pricing.EstimatedCostNanoUSD != wantCost {
			t.Fatalf("challenge priced against the in-flight table: cost = %d, want %d (pre-dispatch table)",
				observation.Pricing.EstimatedCostNanoUSD, wantCost)
		}
		if observation.Pricing.CostState != string(pricing.CostStatePriced) {
			t.Fatalf("cost state = %q, want priced", observation.Pricing.CostState)
		}
	}
}

// TestModelTraceProbePricesAgainstTablePublishedBeforeRun proves the frozen
// table is the one published when the run starts, not one captured at boot.
func TestModelTraceProbePricesAgainstTablePublishedBeforeRun(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-published-price-secret")

	fixture.priceRuntime.Publish(modelTracePriceTable(t, string(channel.OpenAI), 3_000_000_000))
	fixture.service.executor = &modelTraceChallengeExecutor{
		usage: &execution.UsageEvidence{Normalized: usage.Result{
			State:  usage.StateComplete,
			Tokens: usage.Tokens{UncachedInput: 1_000_000},
		}},
	}

	if _, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID)); err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	events := fixture.probeSink.Events()
	if len(events) == 0 {
		t.Fatal("no request log events")
	}
	for _, event := range events {
		if event.Usage.Pricing.EstimatedCostNanoUSD != 3_000_000_000 ||
			event.Usage.Pricing.CostState != string(pricing.CostStatePriced) {
			t.Fatalf("pricing = %#v, want the published table price", event.Usage.Pricing)
		}
	}
}

// TestModelTraceProbeSucceedsWithoutPublishedPriceTable proves that a missing
// table only makes the challenge unpriced: the challenge still runs, still
// logs, and never becomes free.
func TestModelTraceProbeSucceedsWithoutPublishedPriceTable(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-unpriced-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{
		usage: &execution.UsageEvidence{Normalized: usage.Result{
			State:  usage.StateComplete,
			Tokens: usage.Tokens{UncachedInput: 10, Output: 5},
		}},
	}

	if _, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID)); err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	events := fixture.probeSink.Events()
	if len(events) == 0 {
		t.Fatal("no request log events")
	}
	for _, event := range events {
		if event.Usage.Pricing.CostState != string(pricing.CostStateUnpriced) ||
			event.Usage.Pricing.PricingCompleteness != string(pricing.CompletenessUnavailable) ||
			event.Usage.Pricing.EstimatedCostNanoUSD != 0 {
			t.Fatalf("pricing without a table = %#v, want unpriced", event.Usage.Pricing)
		}
		if event.Usage.Result.State != usage.StateComplete {
			t.Fatalf("usage state = %q, want the captured provider usage", event.Usage.Result.State)
		}
	}
}

func TestModelTraceProbeDoesNotChangeRouteHealthOrRecover(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createProbeGroup(t, fixture, []string{probeTestModel})
	group := fixture.manager.Current().Groups[groupID]
	entryKey := state.RouteEntryKey{
		GroupID: groupID, EntryID: probeModelEntryID(group, probeTestModel),
	}
	if _, changed := fixture.registry.SetEntryBlacklistedWithChange(entryKey); !changed {
		t.Fatal("failed to seed route blacklist")
	}
	credential := takeGroupCredential(t, fixture, groupID)
	beforeEntries, err := fixture.registry.SnapshotGroupCredentialEntriesExact(groupID, []uint{credential.ID})
	if err != nil {
		t.Fatal(err)
	}
	observedAt := time.Now()
	beforeStats := fixture.stats.Snapshot(credential.ID, observedAt)
	fixture.service.executor = &modelTraceChallengeExecutor{}

	if _, err := fixture.service.RunModelTraceProbe(
		t.Context(), modelTraceTestRequest(groupID),
	); err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}

	afterEntries, err := fixture.registry.SnapshotGroupCredentialEntriesExact(groupID, []uint{credential.ID})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(afterEntries, beforeEntries) {
		t.Fatalf("fingerprint probe changed credential runtime:\nbefore=%#v\nafter=%#v",
			beforeEntries, afterEntries)
	}
	if after := fixture.stats.Snapshot(credential.ID, observedAt); !reflect.DeepEqual(after, beforeStats) {
		t.Fatalf("fingerprint probe changed health statistics: before=%#v after=%#v", beforeStats, after)
	}
	entry, ok := fixture.registry.EntryRuntime(entryKey, time.Now())
	if !ok || !entry.Blacklisted {
		t.Fatalf("route entry = %#v/%t, want the unrelated blacklist preserved", entry, ok)
	}
}

func TestModelTraceProbeRejectsUnsupportedTarget(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-unsupported-secret")
	executor := &modelTraceChallengeExecutor{}
	fixture.service.executor = executor

	tests := []struct {
		name    string
		request ModelTraceRequest
	}{
		{name: "unknown group", request: ModelTraceRequest{GroupID: 999_999, Model: probeTestModel}},
		{name: "unknown model", request: ModelTraceRequest{GroupID: groupID, Model: "not-configured"}},
		{name: "missing model", request: ModelTraceRequest{GroupID: groupID}},
		{name: "missing group", request: ModelTraceRequest{Model: probeTestModel}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := fixture.service.RunModelTraceProbe(
				t.Context(), test.request,
			); !errors.Is(err, app_errors.ErrBadRequest) {
				t.Fatalf("error = %v, want ErrBadRequest", err)
			}
		})
	}
	if prompts := executor.recordedPrompts(); len(prompts) != 0 {
		t.Fatalf("executed challenges = %d, want none for an unsupported target", len(prompts))
	}
}

func TestModelTraceProbeNeverExposesCredentialMaterial(t *testing.T) {
	const secret = "sk-modeltrace-secret-material"
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, secret)
	fixture.service.executor = &modelTraceChallengeExecutor{}

	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), secret) {
		t.Fatalf("probe response leaked credential material: %s", encoded)
	}

}

func TestModelTraceProbeResponseContract(t *testing.T) {
	useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-contract-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{}

	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &payload); err != nil {
		t.Fatal(err)
	}
	assertJSONKeys(t, payload, []string{
		"group_id", "group_name", "model", "challenges", "analysis", "failure",
		"context_check", "tested_at_ms", "run_id",
	})
	var check map[string]json.RawMessage
	if err := json.Unmarshal(payload["context_check"], &check); err != nil {
		t.Fatal(err)
	}
	assertJSONKeys(t, check, []string{"status", "expected_input_tokens", "reported_input_tokens", "delta", "log_id"})
	if response.ContextCheck.Status != modelTraceContextUnavailable || response.ContextCheck.LogID == nil {
		t.Fatalf("context check = %#v, want unavailable with a log for the request missing usage", response.ContextCheck)
	}
	var challenges []map[string]json.RawMessage
	if err := json.Unmarshal(payload["challenges"], &challenges); err != nil {
		t.Fatal(err)
	}
	for _, challenge := range challenges {
		assertJSONKeys(t, challenge, []string{
			"id", "expected_count", "outcome", "reason", "log_id", "latency_ms", "status_code", "retryable",
		})
	}

}

func TestModelTraceRoutesAndConfigContract(t *testing.T) {
	fixture := newServiceFixture(t)
	module := NewServer(&config.Config{AuthKey: "modeltrace-route-auth"}, fixture.service).HTTPModule()
	routes := make(map[string]httproute.Route, len(module.Routes))
	for _, route := range module.Routes {
		routes[route.Name] = route
	}
	probe, ok := routes["control.modeltrace-probe.run"]
	if !ok {
		t.Fatal("modeltrace probe route is missing")
	}
	if !reflect.DeepEqual(probe.Methods, []string{http.MethodPost}) ||
		probe.Path != "/modeltrace-probe" {
		t.Fatalf("modeltrace probe route = %#v", probe)
	}
	retry := routes["control.modeltrace-probe.retry"]
	if !reflect.DeepEqual(retry.Methods, []string{http.MethodPost}) || retry.Path != "/modeltrace-probe/retry" {
		t.Fatalf("modeltrace retry route = %#v", retry)
	}
	for _, route := range []string{probe.Path, retry.Path} {
		if _, exists := accessKeyControlRoutes["/api"+route]; exists {
			t.Fatalf("accessKeyControlRoutes must not be widened to %s", route)
		}
	}

	previous := loadModelTraceConfig()
	t.Cleanup(func() { storeModelTraceConfig(previous) })
	if err := loadModelTraceConfigFromEnvironment(func(key string) string {
		return map[string]string{
			"MODELTRACE_ENABLED":         "true",
			"MODELTRACE_URL":             "http://modeltrace:8080/",
			"MODELTRACE_TOKEN":           "ignored-service-token",
			"MODELTRACE_TIMEOUT_SECONDS": "ignored-invalid-timeout",
		}[key]
	}); err != nil {
		t.Fatalf("loadModelTraceConfigFromEnvironment() error = %v", err)
	}
	if loaded := loadModelTraceConfig(); loaded == nil || !loaded.Enabled {
		t.Fatalf("local detector config = %#v, want enabled despite retired service settings", loaded)
	}
}
