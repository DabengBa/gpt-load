package gateway

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/state"
	"gpt-load/internal/testutil/encryptiontest"
)

// newModelRewriteTestRuntime publishes a group whose external model "pub" maps
// to the given route entries and credentials, so retry chains can cross both.
func newModelRewriteTestRuntime(
	t *testing.T,
	forwarder AttemptForwarder,
	models []state.ModelConfig,
	upstreamKeys ...string,
) (*gin.Engine, *Handler, *state.CredentialRegistry) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	keyService := encryptiontest.Service(t, "model-rewrite-test-master-key")
	manager := state.NewManager()
	credentialConfigs := make([]state.CredentialConfig, 0, len(upstreamKeys))
	for index := range upstreamKeys {
		credentialConfigs = append(credentialConfigs, state.CredentialConfig{
			ID: uint(index + 1), GroupID: 1, Status: state.CredentialStatusActive,
			Version: 1, IdentityGeneration: uint64(index + 1),
			Fingerprint: "credential-" + string(rune('a'+index)),
		})
	}
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{{
			ConnectionType: "api_key", ID: 1, Name: "openai", ChannelID: channel.OpenAI,
			Params: json.RawMessage(`{}`), Models: models, Enabled: true,
		}},
		Credentials: credentialConfigs,
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: keyService.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	registry := state.NewCredentialRegistry()
	entries := make([]state.CredentialEntry, 0, len(upstreamKeys))
	for index, plaintext := range upstreamKeys {
		credential, err := json.Marshal(map[string]string{"api_key": plaintext})
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		encrypted, err := keyService.Encrypt(string(credential))
		if err != nil {
			t.Fatalf("Encrypt() error = %v", err)
		}
		entries = append(entries, state.CredentialEntry{
			ID: uint(index + 1), GroupID: 1,
			Version: 1, IdentityGeneration: uint64(index + 1),
			Fingerprint: "credential-" + string(rune('a'+index)),
			Status:      state.CredentialStatusActive, EncryptedValue: encrypted,
		})
	}
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	handler := NewHandler(
		manager, registry, keyService, forwarder, dialect.NewSet(dialect.NewOpenAI()),
		health.NewStatsStore(), health.NewMutationCoordinator(),
		nil, nil, nil,
	)
	handler.newRandom = func() *rand.Rand { return rand.New(rand.NewSource(1)) }
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	return engine, handler, registry
}

func modelUnavailableScriptedResult() UpstreamResult {
	return UpstreamResult{
		DispatchState:      execution.DispatchMaybeSent,
		ResponseStarted:    true,
		StatusCode:         http.StatusNotFound,
		Header:             make(http.Header),
		Body:               []byte(`{"error":{"code":"model_not_found"}}`),
		ClassificationBody: []byte(`{"error":{"code":"model_not_found"}}`),
		RequestWritten:     true,
		ExecutionError: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			ScopeHint:  execution.ErrorScopeModel,
			StatusCode: http.StatusNotFound,
			Code:       "model_not_found",
			Summary:    "model unavailable",
		},
	}
}

func successScriptedResult() UpstreamResult {
	return UpstreamResult{
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusOK,
		Header:          http.Header{"Content-Type": {"application/json"}},
		Body:            []byte(`{"ok":true}`),
		RequestWritten:  true,
	}
}

func TestHandlerModelRewriteSwitchesRouteEntryAfterModelFailure(t *testing.T) {
	now := time.Date(2026, time.September, 6, 10, 0, 0, 0, time.UTC)
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		modelUnavailableScriptedResult(),
		successScriptedResult(),
	}}
	engine, handler, _ := newModelRewriteTestRuntime(t, forwarder, []state.ModelConfig{
		{ID: "up-a", Alias: "pub"}, {ID: "up-b", Alias: "pub"},
	}, "sk-one")
	handler.now = func() time.Time { return now }

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"pub","messages":[{"role":"user","content":"hi"}]}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)

	if response.Code != http.StatusOK || len(forwarder.inputs) != 2 {
		t.Fatalf("status/attempts = %d/%d body=%s", response.Code, len(forwarder.inputs), response.Body.String())
	}
	first := forwarder.inputs[0].UpstreamModelID
	second := forwarder.inputs[1].UpstreamModelID
	if first == "" || second == "" || first == second {
		t.Fatalf("upstream models = %q then %q, want the retry to switch route entry", first, second)
	}
	for index, input := range forwarder.inputs {
		if input.ExternalModel != "pub" {
			t.Fatalf("attempt %d external model = %q, want pub", index, input.ExternalModel)
		}
		if !bytes.Contains(input.Request.Body, []byte(`"model":"pub"`)) {
			t.Fatalf("attempt %d request body keeps the client alias, got %s", index, input.Request.Body)
		}
	}
}

func TestHandlerModelRewriteStreamSwitchesRouteEntryAfterModelFailure(t *testing.T) {
	now := time.Date(2026, time.September, 6, 10, 0, 0, 0, time.UTC)
	forwarder := &scriptedForwarder{
		invokeStreamReady: true,
		streamResults: []UpstreamResult{
			modelUnavailableScriptedResult(),
			{
				DispatchState:   execution.DispatchMaybeSent,
				ResponseStarted: true,
				StatusCode:      http.StatusOK,
				Header:          http.Header{"Content-Type": {"text/event-stream"}},
				RequestWritten:  true,
				Committed:       true,
				Stream:          StreamObservation{EndReason: StreamEndCleanEOF},
			},
		},
	}
	engine, handler, _ := newModelRewriteTestRuntime(t, forwarder, []state.ModelConfig{
		{ID: "up-a", Alias: "pub"}, {ID: "up-b", Alias: "pub"},
	}, "sk-one")
	handler.now = func() time.Time { return now }

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"pub","stream":true,"messages":[{"role":"user","content":"hi"}]}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	engine.ServeHTTP(httptest.NewRecorder(), request)

	if len(forwarder.streamInputs) != 2 {
		t.Fatalf("stream attempts = %d, want 2", len(forwarder.streamInputs))
	}
	first := forwarder.streamInputs[0].UpstreamModelID
	second := forwarder.streamInputs[1].UpstreamModelID
	if first == "" || second == "" || first == second {
		t.Fatalf("stream upstream models = %q then %q, want the retry to switch route entry", first, second)
	}
	for index, input := range forwarder.streamInputs {
		if input.ExternalModel != "pub" {
			t.Fatalf("stream attempt %d external model = %q, want pub", index, input.ExternalModel)
		}
	}
}

func TestHandlerModelRewriteKeepsUpstreamModelAcrossCredentialRetry(t *testing.T) {
	now := time.Date(2026, time.September, 6, 10, 0, 0, 0, time.UTC)
	unauthorized := UpstreamResult{
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusUnauthorized,
		Header:          make(http.Header),
		Body:            []byte(`{"error":{"code":"invalid_api_key"}}`),
		RequestWritten:  true,
		ExecutionError: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			ScopeHint:  execution.ErrorScopeCredential,
			StatusCode: http.StatusUnauthorized,
			Code:       "invalid_api_key",
			Summary:    "invalid credential",
		},
	}
	forwarder := &scriptedForwarder{results: []UpstreamResult{unauthorized, successScriptedResult()}}
	engine, handler, _ := newModelRewriteTestRuntime(t, forwarder, []state.ModelConfig{
		{ID: "up-a", Alias: "pub"},
	}, "sk-one", "sk-two")
	handler.now = func() time.Time { return now }

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"pub","messages":[{"role":"user","content":"hi"}]}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)

	if response.Code != http.StatusOK || len(forwarder.inputs) != 2 {
		t.Fatalf("status/attempts = %d/%d body=%s", response.Code, len(forwarder.inputs), response.Body.String())
	}
	if forwarder.inputs[0].APIKey == forwarder.inputs[1].APIKey {
		t.Fatal("credential retry reused the same credential")
	}
	for index, input := range forwarder.inputs {
		if input.UpstreamModelID != "up-a" || input.ExternalModel != "pub" {
			t.Fatalf("attempt %d models = %q/%q, want pub→up-a", index, input.ExternalModel, input.UpstreamModelID)
		}
	}
}
