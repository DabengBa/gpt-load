package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func TestHandlerReportsAnthropicEnabledNoneConflict(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{successScriptedResult()}}
	engine, handler, manager, _ := newRequestLogHandlerTestRuntime(t, forwarder, &recordingAccessKeyRPMLimiter{}, &recordingRequestLogSink{}, "sk-one")
	handler.dialects = dialect.NewSet(dialect.NewAnthropic())
	_, err := manager.Publish(state.CompileInput{
		SystemSettings: config.Settings{state.SettingRetryCount: testDefaultRetryBudget}, ChannelRegistry: channel.NewRegistry(),
		Groups:      []state.GroupConfig{{ID: 1, Name: "anthropic", ChannelID: channel.Anthropic, Params: json.RawMessage(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "claude", Alias: "pub", EntryID: "e000000000001", ReasoningEffort: "none"}}}},
		Credentials: []state.CredentialConfig{{ID: 1, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "credential-1"}},
		AccessKeys:  []state.AccessKeyConfig{{ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"), Status: state.AccessKeyStatusActive}},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"pub","max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":2048},"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("x-api-key", "gl-client")
	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "reasoning_policy_conflict") || len(forwarder.inputs) != 0 {
		t.Fatalf("status=%d body=%s forwards=%d", rec.Code, rec.Body.String(), len(forwarder.inputs))
	}
}
