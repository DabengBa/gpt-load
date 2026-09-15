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

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

type credentialProbeTestExecutor struct {
	mu      sync.Mutex
	result  execution.AttemptResult
	calls   []execution.AttemptSpec
	execute func(execution.AttemptSpec) execution.AttemptResult
}

func (executor *credentialProbeTestExecutor) Execute(
	_ context.Context,
	spec execution.AttemptSpec,
) execution.AttemptResult {
	executor.mu.Lock()
	executor.calls = append(executor.calls, spec.Clone())
	result := executor.result.Clone()
	execute := executor.execute
	executor.mu.Unlock()
	if execute != nil {
		return execute(spec.Clone())
	}
	return result
}

func (*credentialProbeTestExecutor) ExecuteStream(
	context.Context,
	execution.AttemptSpec,
	execution.StreamSink,
) execution.StreamResult {
	panic("unexpected stream execution")
}

func (executor *credentialProbeTestExecutor) recordedCalls() []execution.AttemptSpec {
	executor.mu.Lock()
	defer executor.mu.Unlock()
	result := make([]execution.AttemptSpec, len(executor.calls))
	for index := range executor.calls {
		result[index] = executor.calls[index].Clone()
	}
	return result
}

func successfulCredentialProbeResult() execution.AttemptResult {
	return execution.AttemptResult{
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusOK,
		Header:          http.Header{"Content-Type": []string{"application/json"}},
		Body:            []byte(`{"output":[{"content":[{"text":"4"}]}]}`),
	}
}

func TestGroupCredentialProbeHTTPRequiresAuthAndUsesOnlySpecifiedCredential(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	createGroupWithCredentials(t, fixture, "probe-first-secret")
	secondName := "credential-group-probe-second"
	secondResult, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: &secondName, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
		Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gpt-4o"}}},
		Credentials: "probe-second-secret", ConfirmSameTarget: true, ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatalf("CreateGroup(second) error = %v", err)
	}
	secondGroupID := secondResult.GroupID
	var credentials []models.Credential
	if err := fixture.db.Where("group_id = ?", secondGroupID).Order("id ASC").Find(&credentials).Error; err != nil {
		t.Fatal(err)
	}
	if len(credentials) != 1 {
		t.Fatalf("credentials = %#v, want one", credentials)
	}

	executor := &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}
	fixture.service.executor = executor
	fixture.service.now = func() time.Time {
		return time.Date(2026, time.August, 29, 12, 30, 0, 0, time.UTC)
	}

	const auth = "credential-probe-auth"
	engine := gin.New()
	NewServer(&config.Config{AuthKey: auth}, fixture.service).RegisterRoutes(engine)
	path := fmt.Sprintf("/api/groups/%d/credentials/%d/test", secondGroupID, credentials[0].ID)
	unauthorized := serveCredentialRequest(t, engine, http.MethodPost, path, "{}", "", "")
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized response = %d %s", unauthorized.Code, unauthorized.Body.String())
	}
	response := serveCredentialRequest(t, engine, http.MethodPost, path, "{}", auth, "")
	if response.Code != http.StatusOK {
		t.Fatalf("probe response = %d %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), `"reason":null`) {
		t.Fatalf("passed probe response omitted null reason: %s", response.Body.String())
	}
	var envelope struct {
		Code int                     `json:"code"`
		Data CredentialProbeResponse `json:"data"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Code != 0 || envelope.Data.Outcome != ProbeOutcomePassed ||
		envelope.Data.Model != "gpt-4o" || envelope.Data.Protocol != protocol.OpenAIResponses ||
		envelope.Data.Reason != nil || envelope.Data.Recovered ||
		envelope.Data.TestedAtMS != time.Date(2026, time.August, 29, 12, 30, 0, 0, time.UTC).UnixMilli() ||
		envelope.Data.LatencyMS < 0 {
		t.Fatalf("probe envelope = %#v", envelope)
	}
	calls := executor.recordedCalls()
	if len(calls) != 1 {
		t.Fatalf("probe calls = %d, want one", len(calls))
	}
	call := calls[0]
	if call.Credential.ID != credentials[0].ID || call.Operation != execution.OperationProbe ||
		call.ClientModel != "gpt-4o" || call.UpstreamModel != "gpt-4o" ||
		call.ClientProtocol != protocol.OpenAIResponses {
		t.Fatalf("probe attempt = %#v", call)
	}
	var canonical struct {
		APIKey string `json:"api_key"`
	}
	if err := json.Unmarshal(call.Credential.Data(), &canonical); err != nil {
		t.Fatal(err)
	}
	if canonical.APIKey != "probe-second-secret" {
		t.Fatalf("probed api key = %q, want specified second credential", canonical.APIKey)
	}
}

func TestGroupCredentialProbeUsesFirstConfiguredModel(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-model-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	executor := &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}
	fixture.service.executor = executor

	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil {
		t.Fatal(err)
	}
	if response.Model != "gpt-4o" || response.Outcome != ProbeOutcomePassed {
		t.Fatalf("probe response = %#v", response)
	}
	calls := executor.recordedCalls()
	if len(calls) != 1 || calls[0].UpstreamModel != "gpt-4o" ||
		calls[0].ClientProtocol != protocol.OpenAIResponses {
		t.Fatalf("probe calls = %#v", calls)
	}
}

func TestGroupCredentialProbeDoesNotFallbackAcrossProtocols(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-single-protocol-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	executor := &credentialProbeTestExecutor{execute: func(spec execution.AttemptSpec) execution.AttemptResult {
		result := failedCredentialProbeResult(
			http.StatusNotFound,
			execution.ErrorKindHTTP,
			execution.FailureHintModelUnavailable,
		)
		result.Error.OriginHint = execution.ErrorOriginUpstream
		return result
	}}
	fixture.service.executor = executor

	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil {
		t.Fatal(err)
	}
	if response.Outcome != ProbeOutcomeFailed || response.Protocol != protocol.OpenAIResponses {
		t.Fatalf("probe response = %#v", response)
	}
	calls := executor.recordedCalls()
	if len(calls) != 1 || calls[0].ClientProtocol != protocol.OpenAIResponses {
		t.Fatalf("probe calls = %#v", calls)
	}
}

func TestGroupCredentialProbeHTTPReturnsCompletedUpstreamFailureAsData(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-http-failure-secret")
	var credential models.Credential
	if err := fixture.db.Where("group_id = ?", groupID).Take(&credential).Error; err != nil {
		t.Fatal(err)
	}
	result := failedCredentialProbeResult(
		http.StatusUnauthorized,
		execution.ErrorKindHTTP,
		execution.FailureHintInvalidCredential,
	)
	result.Body = []byte("sensitive-upstream-response")
	result.Error.Summary = "sensitive-upstream-response"
	fixture.service.executor = &credentialProbeTestExecutor{result: result}
	const auth = "credential-probe-failure-auth"
	engine := gin.New()
	NewServer(&config.Config{AuthKey: auth}, fixture.service).RegisterRoutes(engine)

	response := serveCredentialRequest(
		t,
		engine,
		http.MethodPost,
		fmt.Sprintf("/api/groups/%d/credentials/%d/test", groupID, credential.ID),
		"{}",
		auth,
		"",
	)
	if response.Code != http.StatusOK ||
		!strings.Contains(response.Body.String(), `"code":0`) ||
		!strings.Contains(response.Body.String(), `"outcome":"failed"`) ||
		!strings.Contains(response.Body.String(), `"reason":"invalid_credential"`) ||
		strings.Contains(response.Body.String(), "sensitive-upstream-response") {
		t.Fatalf("probe failure response = %d %s", response.Code, response.Body.String())
	}
}

func TestGroupCredentialProbeRecoversBlacklistedCredentialImmediately(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name          string
		prepare       func(t *testing.T, fixture serviceFixture, credentialID uint)
		wantRecovered bool
		wantMutated   bool
	}{
		{
			name: "cooldown",
			prepare: func(t *testing.T, fixture serviceFixture, credentialID uint) {
				t.Helper()
				if !fixture.registry.SetCooldown(credentialID, time.Now().Add(time.Hour)) {
					t.Fatal("SetCooldown() = false")
				}
			},
		},
		{
			name: "blacklisted",
			prepare: func(t *testing.T, fixture serviceFixture, credentialID uint) {
				t.Helper()
				if _, ok := fixture.registry.IncrFailure(credentialID); !ok || !fixture.registry.SetBlacklisted(credentialID) {
					t.Fatal("failed to blacklist credential")
				}
				fixture.stats.RecordFailure(credentialID, health.FailureCategoryInvalidKey, http.StatusUnauthorized, time.Now())
			},
			wantRecovered: true,
			wantMutated:   true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			fixture := newServiceFixture(t)
			groupID := createGroupWithCredentials(t, fixture, "probe-state-secret")
			credential := takeGroupCredential(t, fixture, groupID)
			test.prepare(t, fixture, credential.ID)
			beforeEntries, err := fixture.registry.SnapshotGroupCredentialEntriesExact(groupID, []uint{credential.ID})
			if err != nil {
				t.Fatal(err)
			}
			beforeStats := fixture.stats.Snapshot(credential.ID, time.Now())
			fixture.service.executor = &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}

			response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(response)
			if err != nil {
				t.Fatal(err)
			}
			var payload map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &payload); err != nil {
				t.Fatal(err)
			}
			var recovered bool
			if raw, exists := payload["recovered"]; exists {
				if err := json.Unmarshal(raw, &recovered); err != nil {
					t.Fatal(err)
				}
			}
			if response.Outcome != ProbeOutcomePassed || recovered != test.wantRecovered {
				t.Fatalf("probe response = %#v, recovered=%t", response, recovered)
			}
			afterEntries, err := fixture.registry.SnapshotGroupCredentialEntriesExact(groupID, []uint{credential.ID})
			if err != nil {
				t.Fatal(err)
			}
			if test.wantMutated {
				if afterEntries[0].Blacklisted || afterEntries[0].FailureCount != 0 {
					t.Fatalf("recovered credential = %#v", afterEntries[0])
				}
			} else if !reflect.DeepEqual(afterEntries, beforeEntries) {
				t.Fatalf("non-recovering probe mutated registry: before=%#v after=%#v", beforeEntries, afterEntries)
			}
			if test.wantMutated {
				stats := fixture.stats.Snapshot(credential.ID, time.Now())
				if stats.ConsecutiveFailure != 0 || stats.ConsecutiveProblem != 0 {
					t.Fatalf("recovered stats = %#v", stats)
				}
			} else if afterStats := fixture.stats.Snapshot(credential.ID, time.Now()); !reflect.DeepEqual(afterStats, beforeStats) {
				t.Fatalf("non-recovering probe mutated stats: before=%#v after=%#v", beforeStats, afterStats)
			}
		})
	}
}

func TestGroupCredentialProbeRevokesRestoreEligibilityWhenTargetChangesDuringProbe(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		mutate func(t *testing.T, fixture serviceFixture, groupID uint)
	}{
		{
			name: "header rules",
			mutate: func(t *testing.T, fixture serviceFixture, groupID uint) {
				t.Helper()
				_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
					Overrides: optionalField[config.Settings]{Set: true, Value: config.Settings{
						state.SettingHeaderRules: map[string]any{
							"set": map[string]any{"X-Probe-Changed": "yes"},
						},
					}},
				})
				if err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "group proxy",
			mutate: func(t *testing.T, fixture serviceFixture, groupID uint) {
				t.Helper()
				_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
					Proxy: optionalField[outboundproxy.Config]{Set: true, Value: outboundproxy.Config{
						Mode: outboundproxy.ModeCustom,
						URL:  "http://changed-probe-proxy.example:8080",
					}},
				})
				if err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "credential auth state",
			mutate: func(t *testing.T, fixture serviceFixture, groupID uint) {
				t.Helper()
				credential := takeGroupCredential(t, fixture, groupID)
				if !fixture.registry.SetCredentialAuthState(credential.ID, state.CredentialAuthStateReauthorizationRequired) {
					t.Fatal("SetCredentialAuthState() = false")
				}
			},
		},
		{
			name: "tested model removed",
			mutate: func(t *testing.T, fixture serviceFixture, groupID uint) {
				t.Helper()
				_, err := fixture.service.UpdateGroupModels(t.Context(), groupID, GroupModelsUpdateRequest{
					Models: optionalGroupModels{Set: true, Values: []GroupModel{}},
				})
				if err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "resolved target",
			mutate: func(t *testing.T, fixture serviceFixture, groupID uint) {
				t.Helper()
				_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
					Params: optionalField[json.RawMessage]{Set: true, Value: json.RawMessage(
						`{"base_url":"https://changed-probe-target.example/v1"}`,
					)},
				})
				if err != nil {
					t.Fatal(err)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			fixture := newServiceFixture(t)
			created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
				Name:        stringPointer("probe target change " + test.name),
				ChannelID:   channel.OpenAICompatible,
				Params:      json.RawMessage(`{"base_url":"https://initial-probe-target.example/v1"}`),
				Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gpt-4o"}}},
				Credentials: "probe-target-change-secret", ConnectionType: "api_key",
			})
			if err != nil {
				t.Fatal(err)
			}
			var credential models.Credential
			if err := fixture.db.Where("group_id = ?", created.GroupID).Take(&credential).Error; err != nil {
				t.Fatal(err)
			}
			if _, ok := fixture.registry.IncrFailure(credential.ID); !ok ||
				!fixture.registry.SetBlacklisted(credential.ID) {
				t.Fatal("failed to blacklist credential")
			}
			fixture.service.executor = &credentialProbeTestExecutor{
				execute: func(execution.AttemptSpec) execution.AttemptResult {
					test.mutate(t, fixture, created.GroupID)
					return successfulCredentialProbeResult()
				},
			}

			response, err := fixture.service.TestGroupCredential(t.Context(), created.GroupID, credential.ID)
			if err != nil {
				t.Fatal(err)
			}
			if response.Outcome != ProbeOutcomePassed || response.Recovered {
				t.Fatalf("probe response after target change = %#v", response)
			}
		})
	}
}

func TestGroupCredentialProbeRejectsSubscriptionGroup(t *testing.T) {
	t.Parallel()
	fixture, groupID, credentialID := newSubscriptionCredentialFixture(t)
	executor := &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}
	fixture.service.executor = executor

	_, err := fixture.service.TestGroupCredential(t.Context(), groupID, credentialID)
	if !errors.Is(err, app_errors.ErrForbidden) {
		t.Fatalf("TestGroupCredential() error = %v, want forbidden", err)
	}
	if calls := executor.recordedCalls(); len(calls) != 0 {
		t.Fatalf("subscription probe calls = %#v, want none", calls)
	}
	initControlI18n(t)
	const auth = "subscription-probe-auth"
	engine := gin.New()
	NewServer(&config.Config{AuthKey: auth}, fixture.service).RegisterRoutes(engine)
	response := serveCredentialRequest(
		t,
		engine,
		http.MethodPost,
		fmt.Sprintf("/api/groups/%d/credentials/%d/test", groupID, credentialID),
		"{}",
		auth,
		"",
	)
	if response.Code != http.StatusForbidden ||
		!strings.Contains(response.Body.String(), `"code":"FORBIDDEN"`) {
		t.Fatalf("subscription probe response = %d %s", response.Code, response.Body.String())
	}
}

func TestClassifyCredentialProbeResultUsesStableSafeOutcomes(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		result      execution.AttemptResult
		wantOutcome ProbeOutcome
		wantReason  *ProbeReason
	}{
		{name: "passed", result: successfulCredentialProbeResult(), wantOutcome: ProbeOutcomePassed},
		{
			name:        "invalid credential hint",
			result:      failedCredentialProbeResult(http.StatusForbidden, execution.ErrorKindHTTP, execution.FailureHintInvalidCredential),
			wantOutcome: ProbeOutcomeFailed, wantReason: credentialProbeReasonPointer(ProbeReasonInvalidCredential),
		},
		{
			name:        "unauthorized",
			result:      failedCredentialProbeResult(http.StatusUnauthorized, execution.ErrorKindHTTP, ""),
			wantOutcome: ProbeOutcomeFailed, wantReason: credentialProbeReasonPointer(ProbeReasonInvalidCredential),
		},
		{
			name: "unauthorized without required error evidence is unknown",
			result: execution.AttemptResult{
				DispatchState:   execution.DispatchMaybeSent,
				ResponseStarted: true,
				StatusCode:      http.StatusUnauthorized,
			},
			wantOutcome: ProbeOutcomeInconclusive, wantReason: credentialProbeReasonPointer(ProbeReasonUnknown),
		},
		{
			name:        "model unavailable",
			result:      failedCredentialProbeResult(http.StatusNotFound, execution.ErrorKindHTTP, execution.FailureHintModelUnavailable),
			wantOutcome: ProbeOutcomeFailed, wantReason: credentialProbeReasonPointer(ProbeReasonModelUnavailable),
		},
		{
			name:        "model hint wins over local request kind",
			result:      failedCredentialProbeResult(0, execution.ErrorKindInvalidRequest, execution.FailureHintModelUnavailable),
			wantOutcome: ProbeOutcomeFailed, wantReason: credentialProbeReasonPointer(ProbeReasonModelUnavailable),
		},
		{
			name:        "rate limited",
			result:      failedCredentialProbeResult(http.StatusTooManyRequests, execution.ErrorKindHTTP, ""),
			wantOutcome: ProbeOutcomeInconclusive, wantReason: credentialProbeReasonPointer(ProbeReasonRateLimited),
		},
		{
			name:        "timeout",
			result:      failedCredentialProbeResult(0, execution.ErrorKindTimeout, ""),
			wantOutcome: ProbeOutcomeInconclusive, wantReason: credentialProbeReasonPointer(ProbeReasonTimeout),
		},
		{
			name:        "incompatible",
			result:      failedCredentialProbeResult(0, execution.ErrorKindConversionUnsupported, ""),
			wantOutcome: ProbeOutcomeInconclusive, wantReason: credentialProbeReasonPointer(ProbeReasonIncompatible),
		},
		{
			name:        "upstream error",
			result:      failedCredentialProbeResult(http.StatusServiceUnavailable, execution.ErrorKindHTTP, ""),
			wantOutcome: ProbeOutcomeInconclusive, wantReason: credentialProbeReasonPointer(ProbeReasonUpstreamError),
		},
		{
			name:        "unknown",
			result:      execution.AttemptResult{},
			wantOutcome: ProbeOutcomeInconclusive, wantReason: credentialProbeReasonPointer(ProbeReasonUnknown),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			outcome, reason := classifyCredentialProbeResult(test.result)
			if outcome != test.wantOutcome || !reflect.DeepEqual(reason, test.wantReason) {
				t.Fatalf("classifyCredentialProbeResult() = %q/%v, want %q/%v", outcome, reason, test.wantOutcome, test.wantReason)
			}
		})
	}

	secret := "credential-probe-sensitive-value"
	result := failedCredentialProbeResult(http.StatusUnauthorized, execution.ErrorKindHTTP, execution.FailureHintInvalidCredential)
	result.Body = []byte(secret)
	result.Error.Summary = secret
	outcome, reason := classifyCredentialProbeResult(result)
	encoded, err := json.Marshal(CredentialProbeResponse{Outcome: outcome, Reason: reason})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), secret) {
		t.Fatalf("probe response leaked sensitive upstream content: %s", encoded)
	}
}

func failedCredentialProbeResult(
	statusCode int,
	kind execution.ErrorKind,
	hint execution.FailureHint,
) execution.AttemptResult {
	return execution.AttemptResult{
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: statusCode != 0,
		StatusCode:      statusCode,
		Header:          http.Header{},
		Error: &execution.ErrorEvidence{
			Kind: kind, Hint: hint, StatusCode: statusCode,
			Summary: "sanitized test failure",
		},
	}
}

func credentialProbeReasonPointer(reason ProbeReason) *ProbeReason {
	return &reason
}

func TestGroupCredentialProbeRouteContract(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	module := NewServer(&config.Config{AuthKey: "credential-probe-route-auth"}, fixture.service).HTTPModule()
	for _, route := range module.Routes {
		if route.Name != "control.group-credentials.test" {
			continue
		}
		if !reflect.DeepEqual(route.Methods, []string{http.MethodPost}) ||
			route.Path != "/groups/:group_id/credentials/:credential_id/test" {
			t.Fatalf("credential probe route = %#v", route)
		}
		return
	}
	t.Fatal("credential probe route is missing")
}

var _ execution.Executor = (*credentialProbeTestExecutor)(nil)
