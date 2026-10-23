package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
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
	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/pricing"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

const modelTraceTestToken = "modeltrace-test-token-0123456789"

// modelTraceTestService is a faithful stub of the ModelTrace analysis contract:
// Bearer authentication, GET /api/challenges, POST /api/analyze.
type modelTraceTestService struct {
	mu sync.Mutex

	challenges    []modelTraceTestChallenge
	analyzeCalls  []map[string]any
	challengeHits int
	authHeaders   []string
	analyzeFn     func(map[string]any) (int, map[string]any)
}

type modelTraceTestChallenge struct {
	ID            string `json:"id"`
	ExpectedCount int    `json:"expected_count"`
	Prompt        string `json:"prompt"`
}

func newModelTraceTestService(t *testing.T) (*modelTraceTestService, *httptest.Server) {
	t.Helper()
	service := &modelTraceTestService{challenges: []modelTraceTestChallenge{
		{ID: "probe-1-aaa", ExpectedCount: 293, Prompt: "challenge-one"},
		{ID: "probe-2-bbb", ExpectedCount: 311, Prompt: "challenge-two"},
		{ID: "probe-3-ccc", ExpectedCount: 327, Prompt: "challenge-three"},
	}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		service.mu.Lock()
		service.authHeaders = append(service.authHeaders, r.Header.Get("Authorization"))
		service.mu.Unlock()
		if r.Header.Get("Authorization") != "Bearer "+modelTraceTestToken {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"unauthorized"}`))
			return
		}
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/api/challenges":
			service.mu.Lock()
			service.challengeHits++
			challenges := append([]modelTraceTestChallenge(nil), service.challenges...)
			service.mu.Unlock()
			writeModelTraceTestJSON(w, http.StatusOK, map[string]any{"challenges": challenges})
		case r.Method == http.MethodPost && r.URL.Path == "/api/analyze":
			var payload map[string]any
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":"invalid_json"}`))
				return
			}
			service.mu.Lock()
			service.analyzeCalls = append(service.analyzeCalls, payload)
			analyzeFn := service.analyzeFn
			service.mu.Unlock()
			if analyzeFn != nil {
				status, body := analyzeFn(payload)
				writeModelTraceTestJSON(w, status, body)
				return
			}
			writeModelTraceTestJSON(w, http.StatusOK, modelTraceTestAnalysis(payload))
		default:
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"not_found"}`))
		}
	}))
	t.Cleanup(server.Close)
	return service, server
}

func writeModelTraceTestJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

// modelTraceTestAnalysis answers with one candidate per submitted output so a
// test can assert that every accepted challenge counted toward the calibration.
func modelTraceTestAnalysis(payload map[string]any) map[string]any {
	outputs, _ := payload["outputs"].([]any)
	diagnostics := make([]any, 0, len(outputs))
	for index, raw := range outputs {
		output, _ := raw.(map[string]any)
		diagnostics = append(diagnostics, map[string]any{
			"prompt_id":       output["prompt_id"],
			"expected_count":  output["expected_count"],
			"parsed_numbers":  output["expected_count"],
			"minimum_numbers": 251,
			"accepted":        true,
		})
		_ = index
	}
	return map[string]any{
		"prediction":             "gpt-5.4",
		"prediction_name":        "GPT-5.4",
		"probability":            0.61,
		"family_prediction":      "gpt",
		"family_prediction_name": "GPT",
		"family_probability":     0.72,
		"used_outputs":           len(outputs),
		"method":                 "hellinger+ordered",
		"calibration":            "temperature",
		"candidates": []any{map[string]any{
			"model": "gpt-5.4", "display_name": "GPT-5.4", "family": "gpt",
			"family_name": "GPT", "probability": 0.61,
			"conditional_probability": 0.85, "profile_similarity": 0.93,
		}},
		"families":    map[string]any{"gpt": 0.72, "claude": 0.28},
		"diagnostics": diagnostics,
		"bank": map[string]any{
			"models": 18, "built_at": "2026-10-09T06:24:27.469312+00:00",
			"bank_sha256": "f267d0513aa06981d024ec5394746fceddf44e4e8a75fa6bf8304255f4b32d6d",
		},
		"upstream": map[string]any{
			"repository": "https://github.com/xqy2006/ModelTrace",
			"commit":     "6cd4a8ff0f3f4e49d6d6969dd2fbadfa1569fde2",
		},
	}
}

func (service *modelTraceTestService) observed() (challengeHits int, calls []map[string]any) {
	service.mu.Lock()
	defer service.mu.Unlock()
	calls = append([]map[string]any(nil), service.analyzeCalls...)
	return service.challengeHits, calls
}

// modelTraceChallengeExecutor answers each challenge with a distinct integer
// sequence, so a test can prove that every challenge was dispatched as its own
// formal model request and that all accepted answers reached /api/analyze.
type modelTraceChallengeExecutor struct {
	mu sync.Mutex

	prompts  []string
	sequence int
	fail     func(index int) bool
	execute  func(spec execution.AttemptSpec) execution.AttemptResult
	// usage is the usage evidence a successful challenge reports, so a test can
	// assert that a challenge is priced like formal traffic.
	usage *execution.UsageEvidence
}

func (executor *modelTraceChallengeExecutor) Execute(
	_ context.Context,
	spec execution.AttemptSpec,
) execution.AttemptResult {
	executor.mu.Lock()
	index := executor.sequence
	executor.sequence++
	var prompt string
	prompt = spec.PromptText
	executor.prompts = append(executor.prompts, prompt)
	execute := executor.execute
	fail := executor.fail
	reportedUsage := executor.usage
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

// newModelTraceFixture installs one test ModelTrace deployment for the
// duration of the test. The process configuration is restored on cleanup, and
// these tests stay serial because that configuration is process-global.
func newModelTraceFixture(
	t *testing.T,
	analyzeServer *httptest.Server,
) (serviceFixture, func()) {
	t.Helper()
	previous := loadModelTraceConfig()
	t.Cleanup(func() { storeModelTraceConfig(previous) })
	storeModelTraceConfig(&ModelTraceConfig{
		Enabled: true, BaseURL: analyzeServer.URL, Token: modelTraceTestToken,
	})
	return newServiceFixture(t), func() {}
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
	analysis, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-challenge-secret")
	executor := &modelTraceChallengeExecutor{}
	fixture.service.executor = executor

	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	if len(response.Challenges) != len(analysis.challenges) {
		t.Fatalf("challenges = %d, want %d", len(response.Challenges), len(analysis.challenges))
	}
	if response.Analysis == nil || response.Analysis.Prediction != "gpt-5.4" {
		t.Fatalf("analysis = %#v, want the ModelTrace attribution", response.Analysis)
	}
	if response.Analysis.UsedOutputs != len(analysis.challenges) {
		t.Fatalf("used_outputs = %d, want %d", response.Analysis.UsedOutputs, len(analysis.challenges))
	}
	if response.Service == nil || !response.Service.Enabled || response.Service.BaseURL == "" {
		t.Fatalf("service = %#v, want the enabled deployment identity", response.Service)
	}

	// Every challenge is its own upstream request and its own request-log row.
	prompts := executor.recordedPrompts()
	if len(prompts) != len(analysis.challenges) {
		t.Fatalf("executed challenges = %d, want %d", len(prompts), len(analysis.challenges))
	}
	events := fixture.probeSink.Events()
	if len(events) != len(analysis.challenges) {
		t.Fatalf("request log events = %d, want one per challenge", len(events))
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
	for _, challenge := range response.Challenges {
		if challenge.LogID == nil {
			t.Fatalf("challenge %q has no log id", challenge.ID)
		}
		event, ok := logged[*challenge.LogID]
		if !ok {
			t.Fatalf("challenge %q log id %q has no request-log row", challenge.ID, *challenge.LogID)
		}
		if event.ClientModel != probeTestModel || event.UpstreamModel != probeTestModel {
			t.Fatalf("challenge %q logged model = %q/%q", challenge.ID, event.ClientModel, event.UpstreamModel)
		}
		if challenge.Outcome != modelTraceChallengeSucceeded {
			t.Fatalf("challenge %q outcome = %q, want succeeded", challenge.ID, challenge.Outcome)
		}
	}

	// The whole accepted output set reaches ModelTrace in one calibration call.
	_, calls := analysis.observed()
	if len(calls) != 1 {
		t.Fatalf("analyze calls = %d, want 1", len(calls))
	}
	outputs, _ := calls[0]["outputs"].([]any)
	if len(outputs) != len(analysis.challenges) {
		t.Fatalf("analyze outputs = %d, want %d", len(outputs), len(analysis.challenges))
	}
	submitted := make([]string, 0, len(outputs))
	for _, raw := range outputs {
		output, _ := raw.(map[string]any)
		submitted = append(submitted, fmt.Sprint(output["prompt_id"]))
		if _, exists := output["text"]; !exists {
			t.Fatalf("output %v has no text", output["prompt_id"])
		}
	}
	wantIDs := make([]string, 0, len(analysis.challenges))
	for _, challenge := range analysis.challenges {
		wantIDs = append(wantIDs, challenge.ID)
	}
	sort.Strings(submitted)
	sort.Strings(wantIDs)
	if !reflect.DeepEqual(submitted, wantIDs) {
		t.Fatalf("analyze prompt ids = %v, want %v", submitted, wantIDs)
	}
}

func TestModelTraceProbeKeepsPartialOutputsAndLogsFailures(t *testing.T) {
	analysis, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
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
	if len(events) != len(analysis.challenges) {
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
	_, calls := analysis.observed()
	if len(calls) != 1 {
		t.Fatalf("analyze calls = %d, want 1", len(calls))
	}
	outputs, _ := calls[0]["outputs"].([]any)
	if len(outputs) != len(analysis.challenges)-1 {
		t.Fatalf("analyze outputs = %d, want the %d accepted challenges",
			len(outputs), len(analysis.challenges)-1)
	}
	if response.Analysis == nil || response.Analysis.UsedOutputs != len(analysis.challenges)-1 {
		t.Fatalf("analysis = %#v, want calibration over the accepted outputs", response.Analysis)
	}
}

func TestModelTraceProbeFailsClearlyWithoutUsableOutputs(t *testing.T) {
	analysis, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-allfailed-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{
		fail: func(int) bool { return true },
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
	_, calls := analysis.observed()
	if len(calls) != 0 {
		t.Fatalf("analyze calls = %d, want none without a usable output", len(calls))
	}
	if len(fixture.probeSink.Events()) != len(analysis.challenges) {
		t.Fatalf("request log events = %d, want one failed row per challenge",
			len(fixture.probeSink.Events()))
	}
}

func TestModelTraceProbeReportsAnalysisFailureWithoutDroppingLogs(t *testing.T) {
	analysis, server := newModelTraceTestService(t)
	analysis.analyzeFn = func(map[string]any) (int, map[string]any) {
		return http.StatusUnprocessableEntity, map[string]any{"error": "no_usable_outputs"}
	}
	fixture, _ := newModelTraceFixture(t, server)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-analysis-fail-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{}

	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("RunModelTraceProbe() error = %v", err)
	}
	if response.Analysis != nil {
		t.Fatalf("analysis = %#v, want null on an analysis error", response.Analysis)
	}
	if response.Failure == nil || *response.Failure != modelTraceFailureAnalysisRejected {
		t.Fatalf("failure = %#v, want %q", response.Failure, modelTraceFailureAnalysisRejected)
	}
	if len(fixture.probeSink.Events()) != len(analysis.challenges) {
		t.Fatalf("request log events = %d, want one formal row per challenge",
			len(fixture.probeSink.Events()))
	}
}

func TestModelTraceProbeRequiresEnabledConfiguration(t *testing.T) {
	fixture := newServiceFixture(t)
	useModelTraceConfig(t, nil)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-disabled-secret")
	executor := &modelTraceChallengeExecutor{}
	fixture.service.executor = executor

	_, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if !errors.Is(err, app_errors.ErrModelTraceDisabled) {
		t.Fatalf("RunModelTraceProbe() error = %v, want ErrModelTraceDisabled", err)
	}
	if len(executor.recordedPrompts()) != 0 {
		t.Fatal("a disabled ModelTrace must not spend upstream requests")
	}
	if len(fixture.probeSink.Events()) != 0 {
		t.Fatal("a disabled ModelTrace must not write request logs")
	}
}

func TestModelTraceProbeRejectsMalformedConfiguration(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-badconfig-secret")
	tests := []struct {
		name string
		cfg  *ModelTraceConfig
	}{
		{
			name: "no base url",
			cfg:  &ModelTraceConfig{Enabled: true, Token: modelTraceTestToken},
		},
		{
			name: "no token",
			cfg:  &ModelTraceConfig{Enabled: true, BaseURL: "http://127.0.0.1:1"},
		},
		{
			name: "unsupported scheme",
			cfg: &ModelTraceConfig{
				Enabled: true, BaseURL: "ftp://127.0.0.1:8080", Token: modelTraceTestToken,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			useModelTraceConfig(t, test.cfg)
			if _, err := fixture.service.RunModelTraceProbe(
				t.Context(), modelTraceTestRequest(groupID),
			); !errors.Is(err, app_errors.ErrModelTraceMisconfigured) {
				t.Fatalf("error = %v, want ErrModelTraceMisconfigured", err)
			}
		})
	}
}

func TestModelTraceProbeReportsUnavailableAndTimeout(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-unavailable-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{}

	closed := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	closedURL := closed.URL
	closed.Close()
	useModelTraceConfig(t, &ModelTraceConfig{
		Enabled: true, BaseURL: closedURL, Token: modelTraceTestToken,
	})
	if _, err := fixture.service.RunModelTraceProbe(
		t.Context(), modelTraceTestRequest(groupID),
	); !errors.Is(err, app_errors.ErrModelTraceUnavailable) {
		t.Fatalf("unreachable deployment error = %v, want ErrModelTraceUnavailable", err)
	}

	blocking := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	t.Cleanup(blocking.Close)
	useModelTraceConfig(t, &ModelTraceConfig{
		Enabled: true, BaseURL: blocking.URL, Token: modelTraceTestToken, RequestTimeout: 50 * time.Millisecond,
	})
	_, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if !errors.Is(err, app_errors.ErrModelTraceUnavailable) {
		t.Fatalf("timeout error = %v, want ErrModelTraceUnavailable", err)
	}
	if len(fixture.probeSink.Events()) != 0 {
		t.Fatal("a challenge that never ran must not write request logs")
	}
}

func TestModelTraceProbeRejectsUnauthorizedAnalysis(t *testing.T) {
	_, server := newModelTraceTestService(t)
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-unauthorized-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{}
	useModelTraceConfig(t, &ModelTraceConfig{
		Enabled: true, BaseURL: server.URL, Token: "wrong-token-value-0123456789",
	})

	_, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	var apiErr *app_errors.APIError
	if !errors.As(err, &apiErr) || apiErr.Code != app_errors.ErrModelTraceUnavailable.Code {
		t.Fatalf("unauthorized error = %v, want %s", err, app_errors.ErrModelTraceUnavailable.Code)
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
	_, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
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
	if len(events) != len(response.Challenges) {
		t.Fatalf("request log events = %d, want one per challenge", len(events))
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
	_, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
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
	_, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
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
	_, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
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
	analysis, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
	groupID := createGroupWithCredentials(t, fixture, "modeltrace-unsupported-secret")
	fixture.service.executor = &modelTraceChallengeExecutor{}

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
	if hits, _ := analysis.observed(); hits != 0 {
		t.Fatalf("challenge fetches = %d, want none for an unsupported target", hits)
	}
}

func TestModelTraceProbeNeverExposesCredentialMaterial(t *testing.T) {
	const secret = "sk-modeltrace-secret-material"
	analysis, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
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
	_, calls := analysis.observed()
	if len(calls) == 0 {
		t.Fatal("expected at least one analyze call")
	}
	rawCall, err := json.Marshal(calls)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(rawCall), secret) {
		t.Fatal("the analyze payload leaked credential material")
	}
}

func TestModelTraceProbeResponseContract(t *testing.T) {
	_, server := newModelTraceTestService(t)
	fixture, _ := newModelTraceFixture(t, server)
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
		"group_id", "group_name", "model", "service", "challenges", "analysis", "failure",
		"tested_at_ms",
	})
	var challenges []map[string]json.RawMessage
	if err := json.Unmarshal(payload["challenges"], &challenges); err != nil {
		t.Fatal(err)
	}
	for _, challenge := range challenges {
		assertJSONKeys(t, challenge, []string{
			"id", "expected_count", "outcome", "reason", "log_id", "latency_ms", "status_code",
		})
	}
	var service map[string]json.RawMessage
	if err := json.Unmarshal(payload["service"], &service); err != nil {
		t.Fatal(err)
	}
	assertJSONKeys(t, service, []string{"base_url", "enabled"})
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
	if _, exists := accessKeyControlRoutes["/api/modeltrace-probe"]; exists {
		t.Fatal("accessKeyControlRoutes must not be widened to the fingerprint probe route")
	}

	previous := loadModelTraceConfig()
	t.Cleanup(func() { storeModelTraceConfig(previous) })
	if err := loadModelTraceConfigFromEnvironment(func(key string) string {
		return map[string]string{
			"MODELTRACE_ENABLED":         "true",
			"MODELTRACE_URL":             "http://modeltrace:8080/",
			"MODELTRACE_TOKEN":           modelTraceTestToken,
			"MODELTRACE_TIMEOUT_SECONDS": "45",
		}[key]
	}); err != nil {
		t.Fatalf("loadModelTraceConfigFromEnvironment() error = %v", err)
	}
	loaded := loadModelTraceConfig()
	if loaded == nil || !loaded.Enabled || loaded.BaseURL != "http://modeltrace:8080" ||
		loaded.Token != modelTraceTestToken || loaded.RequestTimeout != 45*time.Second {
		t.Fatalf("loaded config = %#v", loaded)
	}
}
