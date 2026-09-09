package control

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/protocol"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

func performRouteInspectRequest(
	engine *gin.Engine,
	authKey string,
	body string,
) *httptest.ResponseRecorder {
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/route/inspect",
		strings.NewReader(body),
	)
	request.Header.Set("Content-Type", "application/json")
	if authKey != "" {
		request.Header.Set("Authorization", "Bearer "+authKey)
	}
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	return recorder
}

func decodeRouteInspectSuccess(
	t *testing.T,
	recorder *httptest.ResponseRecorder,
) routeInspectResponse {
	t.Helper()
	if recorder.Code != http.StatusOK {
		t.Fatalf("response = %d %s, want 200", recorder.Code, recorder.Body.String())
	}
	var envelope struct {
		Code int                  `json:"code"`
		Data routeInspectResponse `json:"data"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if envelope.Code != 0 {
		t.Fatalf("code = %d, want 0: %s", envelope.Code, recorder.Body.String())
	}
	return envelope.Data
}

func assertRouteReason(
	t *testing.T,
	got *scheduler.ReasonCode,
	want scheduler.ReasonCode,
) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("reason = %v, want %q", got, want)
	}
}

func routeModelValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func TestRouteInspectShowsBenchmarkEntryRowsSharesAndEntryCooldown(t *testing.T) {
	t.Parallel()

	fixture := newServiceFixture(t)
	now := healthNow()
	fixture.service.now = func() time.Time { return now }
	weightA, weightB, weightC := 30, 50, 20
	weightFull := 100
	twoPriority := 2
	if _, err := fixture.manager.Publish(state.CompileInput{
		ChannelRegistry: fixture.channelRegistry,
		Groups: []state.GroupConfig{
			{ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`), Enabled: true,
				Models: []state.ModelConfig{
					{ID: "up-a", Alias: "pub", EntryID: "e000000000001", Weight: &weightA},
					{ID: "up-b", Alias: "pub", EntryID: "e000000000002", Weight: &weightB},
					{ID: "up-c", Alias: "pub", EntryID: "e000000000003", Weight: &weightC, Priority: &twoPriority},
				},
			},
			{ConnectionType: "api_key", ID: 2, Name: "two", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`), Enabled: true,
				Models: []state.ModelConfig{{ID: "up-b", Alias: "pub", EntryID: "e000000000004", Weight: &weightFull}},
			},
			{ConnectionType: "api_key", ID: 3, Name: "three", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`), Enabled: true,
				Models: []state.ModelConfig{{ID: "up-d", Alias: "pub", EntryID: "e000000000005", Weight: &weightFull}},
			},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 10, Name: "client", KeyHash: "hash", Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	if err := fixture.registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 11, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "k1", AuthState: state.CredentialAuthStateReady, EncryptedValue: "c1"},
		{ID: 12, GroupID: 2, Version: 1, IdentityGeneration: 2, Fingerprint: "k2", AuthState: state.CredentialAuthStateReady, EncryptedValue: "c2"},
		{ID: 13, GroupID: 3, Version: 1, IdentityGeneration: 3, Fingerprint: "k3", AuthState: state.CredentialAuthStateReady, EncryptedValue: "c3"},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}

	inspect := func() routeInspectResponse {
		result, err := fixture.service.InspectRoute(routeInspectRequest{
			Protocol: protocol.OpenAICompletions, ExternalModel: "pub", AccessKeyID: 10,
		})
		if err != nil {
			t.Fatalf("InspectRoute() error = %v", err)
		}
		return result
	}

	result := inspect()
	if !result.Routable || len(result.Groups) != 5 {
		t.Fatalf("routable/groups = %t/%d, want routable with 5 entry rows", result.Routable, len(result.Groups))
	}
	order := []struct {
		groupID        uint
		upstream       string
		weight         int
		priority       int
		fallback       bool
		wantShare      float64
		wantConfigured float64
	}{
		{1, "up-a", 30, 1, false, 30.0 / 280.0, 30.0 / 280.0},
		{1, "up-b", 50, 1, false, 50.0 / 280.0, 50.0 / 280.0},
		{2, "up-b", 100, 1, false, 100.0 / 280.0, 100.0 / 280.0},
		{3, "up-d", 100, 1, false, 100.0 / 280.0, 100.0 / 280.0},
		{1, "up-c", 20, 2, true, 0, 1},
	}
	for index, want := range order {
		row := result.Groups[index]
		if row.GroupID != want.groupID || routeModelValue(row.UpstreamModel) != want.upstream ||
			row.EntryWeight != want.weight || row.Priority != want.priority || row.Fallback != want.fallback {
			t.Fatalf("row %d = %#v, want %v", index, row, want)
		}
		if diff := row.EffectiveShare - want.wantShare; diff < -1e-9 || diff > 1e-9 {
			t.Fatalf("row %d effective_share = %v, want %v", index, row.EffectiveShare, want.wantShare)
		}
		if diff := row.ConfiguredShare - want.wantConfigured; diff < -1e-9 || diff > 1e-9 {
			t.Fatalf("row %d configured_share = %v, want %v", index, row.ConfiguredShare, want.wantConfigured)
		}
		if row.EntryCooldownUntilMS != nil {
			t.Fatalf("row %d entry cooldown = %v, want nil", index, row.EntryCooldownUntilMS)
		}
	}
	assertRouteReason(t, result.Groups[4].ReasonCode, scheduler.ReasonTierDemoted)

	if exists, _ := fixture.registry.SetEntryCooldownForEntry(1, "e000000000002", now.Add(30*time.Minute)); !exists {
		t.Fatal("SetEntryCooldownForEntry() exists = false")
	}
	cooled := inspect()
	var cooledRow *routeInspectGroupResponse
	for index := range cooled.Groups {
		row := &cooled.Groups[index]
		if row.GroupID == 1 && routeModelValue(row.UpstreamModel) == "up-b" {
			cooledRow = row
		}
	}
	if cooledRow == nil ||
		cooledRow.EntryCooldownUntilMS == nil ||
		*cooledRow.EntryCooldownUntilMS != now.Add(30*time.Minute).UnixMilli() {
		t.Fatalf("cooled row = %#v", cooledRow)
	}
	assertRouteReason(t, cooledRow.ReasonCode, scheduler.ReasonEntryCooldown)
	if cooledRow.EffectiveShare != 0 {
		t.Fatalf("cooled row share = %v, want 0", cooledRow.EffectiveShare)
	}
	if diff := cooledRow.ConfiguredShare - 50.0/280.0; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("cooled row configured_share = %v, want %v", cooledRow.ConfiguredShare, 50.0/280.0)
	}
	// P1 renormalizes over the remaining routable entries: A 30, G2B 100, D 100.
	remaining := map[int]float64{0: 30.0 / 230.0, 2: 100.0 / 230.0, 3: 100.0 / 230.0}
	for index, want := range remaining {
		if diff := cooled.Groups[index].EffectiveShare - want; diff < -1e-9 || diff > 1e-9 {
			t.Fatalf("row %d share after cooldown = %v, want %v", index, cooled.Groups[index].EffectiveShare, want)
		}
	}

	// When every P1 entry is cooled, the P2 entry becomes the active tier.
	for _, entryID := range []string{"e000000000001", "e000000000002"} {
		if exists, _ := fixture.registry.SetEntryCooldownForEntry(1, entryID, now.Add(time.Hour)); !exists {
			t.Fatalf("SetEntryCooldownForEntry(1, %q) exists = false", entryID)
		}
	}
	for _, item := range []struct {
		group   uint
		entryID string
	}{
		{group: 2, entryID: "e000000000004"},
		{group: 3, entryID: "e000000000005"},
	} {
		if exists, _ := fixture.registry.SetEntryCooldownForEntry(item.group, item.entryID, now.Add(time.Hour)); !exists {
			t.Fatalf("SetEntryCooldownForEntry(%d, %q) exists = false", item.group, item.entryID)
		}
	}
	allP1Cooled := inspect()
	if allP1Cooled.Groups[4].EffectiveShare != 1.0 {
		t.Fatalf("P2 share with all P1 cooled = %v, want 1", allP1Cooled.Groups[4].EffectiveShare)
	}
	for index := range allP1Cooled.Groups[:4] {
		if allP1Cooled.Groups[index].EffectiveShare != 0 {
			t.Fatalf("P1 row %d share with all P1 cooled = %v, want 0", index, allP1Cooled.Groups[index].EffectiveShare)
		}
	}
	if allP1Cooled.Groups[4].ConfiguredShare != 1 {
		t.Fatalf("P2 configured_share with all P1 cooled = %v, want 1", allP1Cooled.Groups[4].ConfiguredShare)
	}
}

func TestConfiguredEntrySharesNormalizeEachPriorityIncludingUnavailableAndZeroWeight(t *testing.T) {
	groups := []scheduler.GroupInspection{
		{Priority: 1, EntryWeight: 25, Routable: false},
		{Priority: 1, EntryWeight: 75, Routable: true},
		{Priority: 2, EntryWeight: 40, Routable: false},
		{Priority: 2, EntryWeight: 0, Routable: true},
		{Priority: 3, EntryWeight: 0, Routable: false},
	}
	got := configuredEntryShares(groups)
	want := []float64{0.25, 0.75, 1, 0, 0}
	for index := range want {
		if diff := got[index] - want[index]; diff < -1e-9 || diff > 1e-9 {
			t.Fatalf("configured share %d = %v, want %v", index, got[index], want[index])
		}
	}
}

func TestRouteInspectReportsSnapshotRouteStrategy(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)

	for _, strategy := range []string{"native_first", "weighted_mix"} {
		t.Run(strategy, func(t *testing.T) {
			settings := config.Settings{}
			if strategy != "native_first" {
				settings["route_strategy"] = strategy
			}
			snapshot, err := fixture.manager.Publish(state.CompileInput{
				ChannelRegistry: fixture.channelRegistry,
				SystemSettings:  settings,
				Groups: []state.GroupConfig{{
					ID: 1, Name: "openai", ChannelID: channel.OpenAI, ConnectionType: "api_key",
					Params: json.RawMessage(`{}`), Models: []state.ModelConfig{{ID: "model"}}, Enabled: true,
				}},
				AccessKeys: []state.AccessKeyConfig{{
					ID: 10, Name: "client", KeyHash: "hash", Status: state.AccessKeyStatusActive,
				}},
			})
			if err != nil {
				t.Fatalf("publish route strategy: %v", err)
			}
			recorder := performRouteInspectRequest(engine, "test-auth-key",
				`{"protocol":"openai-completions","external_model":"model","access_key_id":10}`)
			got := decodeRouteInspectSuccess(t, recorder)
			if got.SnapshotRevision != snapshot.Revision {
				t.Fatalf("snapshot revision = %d, want %d", got.SnapshotRevision, snapshot.Revision)
			}
			var envelope struct {
				Data struct {
					RouteStrategy string `json:"route_strategy"`
				} `json:"data"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
				t.Fatal(err)
			}
			if envelope.Data.RouteStrategy != strategy {
				t.Fatalf("route strategy = %q, want %q", envelope.Data.RouteStrategy, strategy)
			}
		})
	}
}

func TestRouteInspectEndpointRejectsMalformedAndInvalidRequests(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	tests := []struct {
		name     string
		body     string
		wantCode string
	}{
		{
			name:     "unknown field",
			body:     `{"protocol":"openai-completions","external_model":"model","access_key_id":1,"extra":true}`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
		{
			name:     "plaintext access key rejected",
			body:     `{"protocol":"openai-completions","external_model":"model","access_key_id":1,"access_key":"secret"}`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
		{
			name:     "multiple values",
			body:     `{"protocol":"openai-completions","external_model":"model","access_key_id":1}{}`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
		{
			name:     "trailing malformed value",
			body:     `{"protocol":"openai-completions","external_model":"model","access_key_id":1}x`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
		{
			name:     "malformed JSON",
			body:     `{"protocol":"openai-completions"`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
		{
			name:     "invalid protocol",
			body:     `{"protocol":"invalid","external_model":"model","access_key_id":1}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name:     "reserved protocol",
			body:     `{"protocol":"openai-response","external_model":"model","access_key_id":1}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name:     "replaced completions protocol",
			body:     `{"protocol":"openai-chat-completions","external_model":"model","access_key_id":1}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name:     "legacy required features field",
			body:     `{"protocol":"openai-completions","required_features":[],"external_model":"model","access_key_id":1}`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
		{
			name:     "missing model",
			body:     `{"protocol":"openai-completions","access_key_id":1}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name:     "empty model",
			body:     `{"protocol":"openai-completions","external_model":"","access_key_id":1}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name:     "trimmed model required",
			body:     `{"protocol":"openai-completions","external_model":" model ","access_key_id":1}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name:     "control character in model",
			body:     `{"protocol":"openai-completions","external_model":"model\u0085id","access_key_id":1}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name: "model exceeds UTF-8 byte limit",
			body: `{"protocol":"openai-completions","external_model":"` +
				strings.Repeat("a", 253) + `猫","access_key_id":1}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name:     "zero access key id",
			body:     `{"protocol":"openai-completions","external_model":"model","access_key_id":0}`,
			wantCode: app_errors.ErrValidation.Code,
		},
		{
			name:     "protocol has wrong JSON type",
			body:     `{"protocol":1,"external_model":"model","access_key_id":1}`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
		{
			name:     "external model has wrong JSON type",
			body:     `{"protocol":"openai-completions","external_model":false,"access_key_id":1}`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
		{
			name:     "access key id has wrong JSON type",
			body:     `{"protocol":"openai-completions","external_model":"model","access_key_id":"1"}`,
			wantCode: app_errors.ErrInvalidJSON.Code,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := performRouteInspectRequest(engine, "test-auth-key", test.body)
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("response = %d %s, want 400", recorder.Code, recorder.Body.String())
			}
			var envelope struct {
				Code string `json:"code"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if envelope.Code != test.wantCode {
				t.Fatalf("code = %q, want %q", envelope.Code, test.wantCode)
			}
		})
	}
}

func TestRouteInspectDerivesStandardRequestMetadataFromProtocol(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	if _, err := fixture.manager.Publish(state.CompileInput{
		ChannelRegistry: fixture.channelRegistry,
		Groups: []state.GroupConfig{{
			ID: 1, Name: "openai", ChannelID: channel.OpenAI, ConnectionType: "api_key",
			Params: json.RawMessage(`{}`), Models: []state.ModelConfig{{ID: "provider-model", Alias: "public"}},
			Enabled: true,
		}},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 10, Name: "client", KeyHash: "hash", Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	if err := fixture.registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 100, GroupID: 1, AuthState: state.CredentialAuthStateReady,
		Version: 1, IdentityGeneration: 1, Fingerprint: "credential", EncryptedValue: "encrypted",
	}}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)

	tests := []struct {
		protocol         protocol.Protocol
		operation        execution.Operation
		routeMode        execution.RouteMode
		routeRequirement execution.RouteRequirement
	}{
		{protocol: protocol.OpenAICompletions, operation: execution.OperationChatCompletion, routeMode: execution.RouteNative, routeRequirement: execution.RouteRequirementAny},
		{protocol: protocol.OpenAIResponses, operation: execution.OperationResponsesCreate, routeMode: execution.RouteNative, routeRequirement: execution.RouteRequirementAny},
		{protocol: protocol.OpenAIImages, operation: execution.OperationImagesGenerate, routeMode: execution.RouteNative, routeRequirement: execution.RouteRequirementNative},
		{protocol: protocol.OpenAIEmbeddings, operation: execution.OperationEmbeddingsCreate, routeMode: execution.RouteNative, routeRequirement: execution.RouteRequirementNative},
		{protocol: protocol.Anthropic, operation: execution.OperationChatCompletion, routeMode: execution.RouteConverted, routeRequirement: execution.RouteRequirementAny},
		{protocol: protocol.Gemini, operation: execution.OperationChatCompletion, routeMode: execution.RouteConverted, routeRequirement: execution.RouteRequirementAny},
	}
	for _, test := range tests {
		t.Run(string(test.protocol), func(t *testing.T) {
			recorder := performRouteInspectRequest(
				engine,
				"test-auth-key",
				`{"protocol":"`+string(test.protocol)+`","external_model":"public","access_key_id":10}`,
			)
			got := decodeRouteInspectSuccess(t, recorder)
			if got.Protocol != test.protocol || got.Operation != test.operation ||
				got.RouteRequirement != test.routeRequirement ||
				routeModelValue(got.ExternalModel) != "public" ||
				len(got.Groups) != 1 || got.Groups[0].RouteMode != test.routeMode {
				t.Fatalf("route inspection = %#v", got)
			}
		})
	}
}

func TestRouteInspectRejectsLegacyDerivedFields(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	recorder := performRouteInspectRequest(
		engine,
		"test-auth-key",
		`{"protocol":"openai-completions","operation":"chat_completion","route_requirement":"any","external_model":"public","access_key_id":10}`,
	)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("response = %d %s, want 400", recorder.Code, recorder.Body.String())
	}
	var envelope struct {
		Code string `json:"code"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if envelope.Code != app_errors.ErrInvalidJSON.Code {
		t.Fatalf("code = %q, want %q", envelope.Code, app_errors.ErrInvalidJSON.Code)
	}
}

func TestRouteInspectStandardRequestIncludesNativeAndConvertedTargets(t *testing.T) {
	t.Parallel()

	fixture := newServiceFixture(t)
	if _, err := fixture.manager.Publish(state.CompileInput{
		ChannelRegistry: fixture.channelRegistry,
		Groups: []state.GroupConfig{
			{ConnectionType: "api_key", ID: 1, Name: "native", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "native-model", Alias: "public-model"}}, Enabled: true,
			},
			{ConnectionType: "api_key", ID: 2, Name: "converted", ChannelID: channel.OpenAICompatible,
				Params: json.RawMessage(`{"base_url":"https://compatible.example/v1"}`),
				Models: []state.ModelConfig{{ID: "converted-model", Alias: "public-model"}}, Enabled: true,
			},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 10, Name: "client", KeyHash: "hash", Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	if err := fixture.registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 11, GroupID: 1, Version: 1, IdentityGeneration: 11, Fingerprint: "native", AuthState: state.CredentialAuthStateReady, EncryptedValue: "native"},
		{ID: 21, GroupID: 2, Version: 1, IdentityGeneration: 21, Fingerprint: "converted", AuthState: state.CredentialAuthStateReady, EncryptedValue: "converted"},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}

	request := routeInspectRequest{
		Protocol: protocol.OpenAIResponses, ExternalModel: "public-model", AccessKeyID: 10,
	}
	result, err := fixture.service.InspectRoute(request)
	if err != nil {
		t.Fatalf("InspectRoute() error = %v", err)
	}
	if !result.Routable || result.RouteRequirement != execution.RouteRequirementAny ||
		len(result.Groups) != 2 || !result.Groups[0].RouteRequirementSatisfied ||
		!result.Groups[1].RouteRequirementSatisfied || !result.Groups[1].Included {
		t.Fatalf("result = %#v", result)
	}
}

func TestRouteInspectEndpointReturnsCurrentSafeExplanation(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	now := healthNow()
	fixture.service.now = func() time.Time { return now }
	if _, err := fixture.manager.Publish(state.CompileInput{
		ChannelRegistry: fixture.channelRegistry,
		Groups: []state.GroupConfig{
			{ConnectionType: "api_key", ID: 2, Name: "backup", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "provider-backup", Alias: "public-model"}},
			},
			{ConnectionType: "api_key", ID: 1, Name: "primary", ChannelID: channel.OpenAI,
				Params:  json.RawMessage(`{}`),
				Models:  []state.ModelConfig{{ID: "provider-model", Alias: "public-model"}},
				Enabled: true,
			},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 10, Name: "production", KeyHash: "active-hash",
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	if err := fixture.registry.ReplaceCredentials([]state.CredentialEntry{
		{
			ID: 31, GroupID: 2, Version: 1, IdentityGeneration: 31, Fingerprint: "test-31", AuthState: state.CredentialAuthStateReady,
		},
		{
			ID: 22, GroupID: 1, Version: 1, IdentityGeneration: 22, Fingerprint: "test-22", AuthState: state.CredentialAuthStateReady,
			CooldownUntil:  now.Add(time.Minute),
			EncryptedValue: "cipher-two",
		},
		{
			ID: 21, GroupID: 1, Version: 1, IdentityGeneration: 21, Fingerprint: "test-21", AuthState: state.CredentialAuthStateReady,
			EncryptedValue: "cipher-one",
		},
	}); err != nil {
		t.Fatalf("Replace() error = %v", err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	recorder := performRouteInspectRequest(
		engine,
		"test-auth-key",
		`{"protocol":"openai-completions","external_model":"public-model","access_key_id":10}`,
	)
	got := decodeRouteInspectSuccess(t, recorder)
	if got.ObservedAtMS != now.UnixMilli() ||
		got.SnapshotRevision != fixture.manager.Current().Revision ||
		got.Protocol != protocol.OpenAICompletions ||
		got.Operation != execution.OperationChatCompletion ||
		got.RouteRequirement != execution.RouteRequirementAny ||
		routeModelValue(got.ExternalModel) != "public-model" ||
		got.AccessKey != (routeInspectAccessKeyResponse{
			ID: 10, Name: "production", Status: state.AccessKeyStatusActive,
		}) ||
		!got.Routable || got.ReasonCode != nil {
		t.Fatalf("route response = %#v", got)
	}
	if len(got.Groups) != 2 || got.Groups[0].GroupID != 1 ||
		got.Groups[1].GroupID != 2 {
		t.Fatalf("group order = %#v", got.Groups)
	}
	primary := got.Groups[0]
	if primary.GroupName != "primary" ||
		primary.ChannelID != channel.OpenAI ||
		primary.RouteMode != execution.RouteNative ||
		!primary.RouteRequirementSatisfied ||
		routeModelValue(primary.UpstreamModel) != "provider-model" ||
		!primary.Included ||
		!primary.Routable || primary.ReasonCode != nil ||
		len(primary.Credentials) != 2 ||
		primary.Credentials[0].CredentialID != 21 || primary.Credentials[1].CredentialID != 22 {
		t.Fatalf("primary group = %#v", primary)
	}
	available := primary.Credentials[0]
	if !available.Available || available.ReasonCode != nil ||
		available.CooldownUntilMS != nil {
		t.Fatalf("available key = %#v", available)
	}
	cooldown := primary.Credentials[1]
	if cooldown.Available ||
		cooldown.CooldownUntilMS == nil ||
		*cooldown.CooldownUntilMS != now.Add(time.Minute).UnixMilli() {
		t.Fatalf("cooldown key = %#v", cooldown)
	}
	assertRouteReason(t, cooldown.ReasonCode, scheduler.ReasonCredentialCooldown)
	backup := got.Groups[1]
	if backup.GroupName != "backup" ||
		routeModelValue(backup.UpstreamModel) != "provider-backup" ||
		!backup.Included || !backup.Routable || backup.ReasonCode != nil ||
		len(backup.Credentials) != 1 || backup.Credentials[0].CredentialID != 31 ||
		!backup.Credentials[0].Available || backup.Credentials[0].ReasonCode != nil ||
		backup.Credentials[0].CooldownUntilMS != nil {
		t.Fatalf("backup group = %#v", backup)
	}
	body := recorder.Body.String()
	if strings.Count(body, `"reason_code":null`) != 5 ||
		strings.Count(body, `"cooldown_until_ms":null`) != 2 {
		t.Fatalf("success response must preserve explicit nulls: %s", body)
	}
	lower := strings.ToLower(body)
	for _, forbidden := range []string{
		"cipher-one", "cipher-two", "cipher-three", "active-hash", "upstream_url",
		"header_rules", "filters",
	} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("response exposes %q: %s", forbidden, body)
		}
	}
}

func TestRouteInspectEndpointReturnsFilterExplanations(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	tests := []struct {
		name        string
		filters     state.FilterSet
		wantReason  scheduler.ReasonCode
		assertGroup bool
	}{
		{
			name: "protocol filter",
			filters: state.FilterSet{
				Protocols: map[protocol.Protocol]struct{}{protocol.Anthropic: {}},
			},
			wantReason: scheduler.ReasonProtocolFiltered,
		},
		{
			name: "model filter",
			filters: state.FilterSet{
				Models: map[string]struct{}{"other-model": {}},
			},
			wantReason: scheduler.ReasonModelFiltered,
		},
		{
			name: "group filter",
			filters: state.FilterSet{
				Groups: map[uint]struct{}{99: {}},
			},
			wantReason: scheduler.ReasonGroupFiltered, assertGroup: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newServiceFixture(t)
			now := healthNow()
			fixture.service.now = func() time.Time { return now }
			if _, err := fixture.manager.Publish(state.CompileInput{
				ChannelRegistry: fixture.channelRegistry,
				Groups: []state.GroupConfig{
					{ConnectionType: "api_key", ID: 2, Name: "second", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
						Models: []state.ModelConfig{{ID: "provider-two", Alias: "public-model"}},
					},
					{ConnectionType: "api_key", ID: 1, Name: "first", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
						Models:  []state.ModelConfig{{ID: "provider-one", Alias: "public-model"}},
						Enabled: true,
					},
				},
				AccessKeys: []state.AccessKeyConfig{{
					ID: 10, Name: "filtered", KeyHash: "filtered-hash",
					Status: state.AccessKeyStatusActive, Filters: test.filters,
				}},
			}); err != nil {
				t.Fatalf("Publish() error = %v", err)
			}
			if err := fixture.registry.ReplaceCredentials([]state.CredentialEntry{
				{ID: 22, GroupID: 2, Version: 1, IdentityGeneration: 22, Fingerprint: "test-22", AuthState: state.CredentialAuthStateReady, EncryptedValue: "two"},
				{ID: 11, GroupID: 1, Version: 1, IdentityGeneration: 11, Fingerprint: "test-11", AuthState: state.CredentialAuthStateReady, EncryptedValue: "one"},
			}); err != nil {
				t.Fatalf("Replace() error = %v", err)
			}
			engine := gin.New()
			NewServer(
				&config.Config{AuthKey: "test-auth-key"},
				fixture.service,
			).RegisterRoutes(engine)
			recorder := performRouteInspectRequest(
				engine,
				"test-auth-key",
				`{"protocol":"openai-completions","external_model":"public-model","access_key_id":10}`,
			)
			got := decodeRouteInspectSuccess(t, recorder)
			if got.ObservedAtMS != now.UnixMilli() ||
				got.SnapshotRevision != fixture.manager.Current().Revision ||
				got.Routable {
				t.Fatalf("filter response = %#v", got)
			}
			assertRouteReason(t, got.ReasonCode, test.wantReason)
			if !test.assertGroup {
				if got.Groups == nil || len(got.Groups) != 0 ||
					!strings.Contains(recorder.Body.String(), `"groups":[]`) {
					t.Fatalf("top-level filter groups = %#v", got.Groups)
				}
				return
			}
			if len(got.Groups) != 2 ||
				got.Groups[0].GroupID != 1 || got.Groups[1].GroupID != 2 {
				t.Fatalf("group order = %#v", got.Groups)
			}
			for index, group := range got.Groups {
				if group.Included || group.Routable ||
					group.Credentials == nil || len(group.Credentials) != 0 {
					t.Fatalf("filtered group %d = %#v", index, group)
				}
				assertRouteReason(t, group.ReasonCode, scheduler.ReasonGroupFiltered)
			}
			if routeModelValue(got.Groups[0].UpstreamModel) != "provider-one" ||
				routeModelValue(got.Groups[1].UpstreamModel) != "provider-two" {
				t.Fatalf("filtered group mapping = %#v", got.Groups)
			}
		})
	}
}

func TestRouteInspectEndpointReturnsNoRouteTargetExplanation(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	now := healthNow()
	fixture.service.now = func() time.Time { return now }
	if _, err := fixture.manager.Publish(state.CompileInput{
		ChannelRegistry: fixture.channelRegistry,
		Groups: []state.GroupConfig{{ConnectionType: "api_key", ID: 1, Name: "primary", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
			Models: []state.ModelConfig{{ID: "configured-model"}}, Enabled: true,
		}},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 10, Name: "production", KeyHash: "active-hash",
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	recorder := performRouteInspectRequest(
		engine,
		"test-auth-key",
		`{"protocol":"openai-completions","external_model":"missing-model","access_key_id":10}`,
	)
	got := decodeRouteInspectSuccess(t, recorder)
	if got.ObservedAtMS != now.UnixMilli() ||
		got.SnapshotRevision != fixture.manager.Current().Revision ||
		got.Routable || got.Groups == nil || len(got.Groups) != 0 {
		t.Fatalf("no-target response = %#v", got)
	}
	assertRouteReason(t, got.ReasonCode, scheduler.ReasonNoRouteTarget)
	if !strings.Contains(recorder.Body.String(), `"groups":[]`) {
		t.Fatalf("no-target groups must be []: %s", recorder.Body.String())
	}
}

func TestRouteInspectEndpointReturnsNoAvailableKeyExplanation(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	now := healthNow()
	fixture.service.now = func() time.Time { return now }
	if _, err := fixture.manager.Publish(state.CompileInput{
		ChannelRegistry: fixture.channelRegistry,
		Groups: []state.GroupConfig{{ConnectionType: "api_key", ID: 1, Name: "primary", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
			Models: []state.ModelConfig{{ID: "provider-model", Alias: "public-model"}},
		}},
		AccessKeys: []state.AccessKeyConfig{{ID: 10, Name: "production", KeyHash: "active-hash", Status: state.AccessKeyStatusActive}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	if err := fixture.registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 11, GroupID: 1, Version: 1, IdentityGeneration: 11, Fingerprint: "test-11", AuthState: state.CredentialAuthStateReauthorizationRequired},
		{ID: 12, GroupID: 1, Version: 1, IdentityGeneration: 12, Fingerprint: "test-12", AuthState: state.CredentialAuthStateReauthorizationRequired},
		{ID: 13, GroupID: 1, Version: 1, IdentityGeneration: 13, Fingerprint: "test-13", AuthState: state.CredentialAuthStateReauthorizationRequired},
	}); err != nil {
		t.Fatalf("Replace() error = %v", err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	recorder := performRouteInspectRequest(engine, "test-auth-key", `{"protocol":"openai-completions","external_model":"public-model","access_key_id":10}`)
	got := decodeRouteInspectSuccess(t, recorder)
	if got.Routable || len(got.Groups) != 1 || len(got.Groups[0].Credentials) != 3 {
		t.Fatalf("unavailable response = %#v", got)
	}
	assertRouteReason(t, got.ReasonCode, scheduler.ReasonNoAvailableCredential)
	for _, credential := range got.Groups[0].Credentials {
		if credential.Available || credential.ReasonCode == nil || credential.CooldownUntilMS != nil {
			t.Fatalf("unavailable credential = %#v", credential)
		}
	}
}

func TestRouteInspectReturnsDisabledAccessKeyAsExplanation(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	if _, err := fixture.manager.Publish(state.CompileInput{
		AccessKeys: []state.AccessKeyConfig{{
			ID: 12, Name: "disabled", KeyHash: "disabled-hash",
			Status: state.AccessKeyStatusDisabled,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	result, err := fixture.service.InspectRoute(routeInspectRequest{
		Protocol: protocol.OpenAICompletions, ExternalModel: "model", AccessKeyID: 12,
	})
	if err != nil {
		t.Fatalf("InspectRoute() error = %v", err)
	}
	if result.Routable || result.ReasonCode == nil ||
		*result.ReasonCode != scheduler.ReasonAccessKeyDisabled ||
		result.Groups == nil || len(result.Groups) != 0 {
		t.Fatalf("disabled result = %#v", result)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	recorder := performRouteInspectRequest(
		engine,
		"test-auth-key",
		`{"protocol":"openai-completions","external_model":"model","access_key_id":12}`,
	)
	if recorder.Code != http.StatusOK ||
		!strings.Contains(recorder.Body.String(), `"reason_code":"access_key_disabled"`) ||
		!strings.Contains(recorder.Body.String(), `"groups":[]`) {
		t.Fatalf("disabled HTTP response = %d %s", recorder.Code, recorder.Body.String())
	}
}

func TestRouteInspectReturnsExpiredAccessKeyAsExplanation(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	now := time.Date(2026, time.August, 30, 12, 0, 0, 0, time.UTC)
	fixture.service.now = func() time.Time { return now }
	expiresAtMS := now.UnixMilli()
	if _, err := fixture.manager.Publish(state.CompileInput{
		AccessKeys: []state.AccessKeyConfig{{
			ID: 13, Name: "expired", KeyHash: "expired-hash",
			Status: state.AccessKeyStatusActive, ExpiresAtMS: &expiresAtMS,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	result, err := fixture.service.InspectRoute(routeInspectRequest{
		Protocol: protocol.OpenAICompletions, ExternalModel: "model", AccessKeyID: 13,
	})
	if err != nil {
		t.Fatalf("InspectRoute() error = %v", err)
	}
	if result.Routable || result.ReasonCode == nil ||
		*result.ReasonCode != scheduler.ReasonAccessKeyExpired ||
		result.Groups == nil || len(result.Groups) != 0 {
		t.Fatalf("expired result = %#v", result)
	}
}

func TestRouteInspectMissingAccessKeyReturnsNotFound(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	_, err := fixture.service.InspectRoute(routeInspectRequest{
		Protocol: protocol.OpenAICompletions, ExternalModel: "model", AccessKeyID: 404,
	})
	if !errors.Is(err, app_errors.ErrResourceNotFound) {
		t.Fatalf("InspectRoute() error = %v, want NOT_FOUND", err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	recorder := performRouteInspectRequest(
		engine,
		"test-auth-key",
		`{"protocol":"openai-completions","external_model":"model","access_key_id":404}`,
	)
	var envelope struct {
		Code string `json:"code"`
	}
	if decodeErr := json.Unmarshal(recorder.Body.Bytes(), &envelope); decodeErr != nil {
		t.Fatalf("decode missing-key response: %v", decodeErr)
	}
	if recorder.Code != http.StatusNotFound ||
		envelope.Code != app_errors.ErrResourceNotFound.Code {
		t.Fatalf("missing-key response = %d %#v", recorder.Code, envelope)
	}
}

type routeInspectEncryptionSpy struct {
	calls atomic.Int64
}

func (spy *routeInspectEncryptionSpy) Encrypt(string) (string, error) {
	spy.calls.Add(1)
	return "", nil
}

func (spy *routeInspectEncryptionSpy) Decrypt(string) (string, error) {
	spy.calls.Add(1)
	return "", nil
}

func (spy *routeInspectEncryptionSpy) Hash(string) string {
	spy.calls.Add(1)
	return ""
}

func TestRouteInspectNeverCallsUpstreamOrMutatesRuntime(t *testing.T) {
	t.Parallel()
	var upstreamCalls atomic.Int64
	fixture := newServiceFixture(t)
	encryptionSpy := &routeInspectEncryptionSpy{}
	fixture.service.encryption = encryptionSpy
	dialectCalls := 0
	fixture.service.executor = newRecordingDiscoveryExecutor(&recordingDiscoveryExecutorTarget{
		value: protocol.OpenAICompletions,
		listFn: func(
			context.Context,
			string,
			string,
			state.HeaderRules,
		) ([]string, error) {
			dialectCalls++
			return nil, nil
		},
	})
	if _, err := fixture.manager.Publish(state.CompileInput{
		ChannelRegistry: fixture.channelRegistry,
		Groups: []state.GroupConfig{{ConnectionType: "api_key", ID: 1, Name: "upstream", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
			Models: []state.ModelConfig{{ID: "model"}}, Enabled: true,
		}},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: "hash",
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	if err := fixture.registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 1, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "test-1", AuthState: state.CredentialAuthStateReady,
		EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatalf("Replace() error = %v", err)
	}
	beforeSnapshot := fixture.manager.Current()
	beforeKeys := fixture.registry.Snapshot()
	beforeStats := fixture.stats.Snapshot(1, healthNow())
	requestLogStatsCalls := 0
	fixture.requestLogStats.fn = func() requestlog.Stats {
		requestLogStatsCalls++
		return requestlog.Stats{}
	}

	if _, err := fixture.service.InspectRoute(routeInspectRequest{
		Protocol: protocol.OpenAICompletions, ExternalModel: "model", AccessKeyID: 1,
	}); err != nil {
		t.Fatalf("InspectRoute() error = %v", err)
	}
	if upstreamCalls.Load() != 0 || dialectCalls != 0 || encryptionSpy.calls.Load() != 0 {
		t.Fatalf(
			"upstream calls = %d, Dialect calls = %d, encryption calls = %d",
			upstreamCalls.Load(),
			dialectCalls,
			encryptionSpy.calls.Load(),
		)
	}
	if requestLogStatsCalls != 0 {
		t.Fatalf("RequestLog stats calls = %d, want 0", requestLogStatsCalls)
	}
	if fixture.manager.Current() != beforeSnapshot ||
		!reflect.DeepEqual(fixture.registry.Snapshot(), beforeKeys) ||
		fixture.stats.Snapshot(1, healthNow()) != beforeStats {
		t.Fatal("InspectRoute() mutated runtime state")
	}
}

func TestRouteInspectEndpointRequiresManagementAuthentication(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	beforeSnapshot := fixture.manager.Current()
	beforeKeys := fixture.registry.Snapshot()
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	recorder := performRouteInspectRequest(
		engine,
		"",
		`{"protocol":"openai-completions","external_model":"model","access_key_id":1}`,
	)
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("response = %d %s, want 401", recorder.Code, recorder.Body.String())
	}
	if fixture.manager.Current() != beforeSnapshot ||
		!reflect.DeepEqual(fixture.registry.Snapshot(), beforeKeys) {
		t.Fatal("unauthenticated Inspector request reached runtime state")
	}
}

func TestRouteInspectCatalogMismatchReturnsInternalServerError(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	if _, err := fixture.manager.Publish(state.CompileInput{
		ChannelRegistry: fixture.channelRegistry,
		Groups: []state.GroupConfig{{ConnectionType: "api_key", ID: 1, Name: "primary", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
			Models: []state.ModelConfig{{ID: "model"}}, Enabled: true,
		}},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: "hash", Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	fixture.manager.Current().ExecutionRouteCatalog[protocol.OpenAICompletions][execution.OperationChatCompletion]["model"] = []state.RouteTarget{{
		GroupID: 999, UpstreamModelID: "model",
	}}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	recorder := performRouteInspectRequest(
		engine,
		"test-auth-key",
		`{"protocol":"openai-completions","external_model":"model","access_key_id":1}`,
	)
	var envelope struct {
		Code string `json:"code"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if recorder.Code != http.StatusInternalServerError ||
		envelope.Code != app_errors.ErrInternalServer.Code {
		t.Fatalf("catalog mismatch response = %d %#v", recorder.Code, envelope)
	}
}
