package bifrost

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/provideradapter"
)

func TestMultiProtocolGatewayDispatchAndChatProbe(t *testing.T) {
	t.Parallel()
	const id = channel.ID("multi_protocol")
	tests := []struct {
		name, path, wantPath, body, response, header, credential string
		protocol                                                 protocol.Protocol
		operation                                                execution.Operation
	}{
		{"chat", "/v1/chat/completions", "/team/v1/chat/completions", `{"model":"client","messages":[{"role":"user","content":"hi"}]}`, `{"id":"chat_1","object":"chat.completion","created":1,"model":"upstream","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, "Authorization", "Bearer " + testAPIKey, protocol.OpenAICompletions, execution.OperationChatCompletion},
		{"responses", "/v1/responses", "/team/v1/responses", `{"model":"client","input":"hi","store":false}`, `{"id":"resp_1","object":"response","status":"completed","model":"upstream","output":[]}`, "Authorization", "Bearer " + testAPIKey, protocol.OpenAIResponses, execution.OperationResponsesCreate},
		{"anthropic", "/v1/messages", "/team/v1/messages", `{"model":"client","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`, `{"id":"msg_1","type":"message","role":"assistant","model":"upstream","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`, "X-Api-Key", testAPIKey, protocol.Anthropic, execution.OperationChatCompletion},
		{"gemini", "/v1beta/models/client:generateContent", "/team/v1beta/models/upstream:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"modelVersion":"upstream"}`, "X-Goog-Api-Key", testAPIKey, protocol.Gemini, execution.OperationChatCompletion},
		{"probe", "", "/team/v1/chat/completions", "", `{"id":"chat_1","object":"chat.completion","created":1,"model":"upstream","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, "Authorization", "Bearer " + testAPIKey, protocol.OpenAICompletions, execution.OperationProbe},
		{"chat-version-prefix", "/v1/chat/completions", "/v1/v1/chat/completions", `{"model":"client","messages":[{"role":"user","content":"hi"}]}`, `{"id":"chat_1","object":"chat.completion","created":1,"model":"upstream","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, "Authorization", "Bearer " + testAPIKey, protocol.OpenAICompletions, execution.OperationChatCompletion},
		{"gemini-version-prefix", "/v1beta/models/client:generateContent", "/team/v1beta/v1beta/models/upstream:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"modelVersion":"upstream"}`, "X-Goog-Api-Key", testAPIKey, protocol.Gemini, execution.OperationChatCompletion},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseSuffix := "/team"
			if tc.name == "chat-version-prefix" {
				baseSuffix = "/v1"
			}
			if tc.name == "gemini-version-prefix" {
				baseSuffix = "/team/v1beta"
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tc.wantPath || r.Header.Get(tc.header) != tc.credential {
					t.Errorf("upstream path or credential mismatch: path=%q header=%s", r.URL.Path, tc.header)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.response)
			}))
			defer server.Close()
			registry := channel.NewRegistry()
			resolved, err := registry.Resolve(id, json.RawMessage(`{"base_url":"`+server.URL+baseSuffix+`"}`))
			if err != nil {
				t.Fatal(err)
			}
			manager, err := newRuntimeManager(runtimeOptions{allowPrivateNetwork: true}, registry)
			if err != nil {
				t.Fatal(err)
			}
			if err := manager.Start(t.Context()); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(manager.Shutdown)
			if err := manager.Reconcile([]provideradapter.RuntimeTarget{{Target: resolved}}); err != nil {
				t.Fatal(err)
			}
			spec := execution.AttemptSpec{
				RequestID: "multi-request", AttemptID: "multi-attempt", Sequence: 1,
				ChannelID: string(id), ClientProtocol: tc.protocol, Operation: tc.operation,
				ClientModel: "client", UpstreamModel: "upstream", Method: http.MethodPost, Path: tc.path,
				Header: http.Header{"Content-Type": {"application/json"}}, Body: []byte(tc.body), TargetConfig: resolved.TargetConfig,
				Credential: execution.NewCredentialSnapshot(17, 1, 1, []byte(`{"api_key":"`+testAPIKey+`"}`)),
			}
			if tc.operation == execution.OperationProbe {
				spec = utilitySpec(id, tc.protocol, tc.operation, "", "", nil)
				spec.ClientModel, spec.UpstreamModel = "client", "upstream"
				spec.TargetConfig = resolved.TargetConfig
			}
			spec = freezeTestAttempt(spec)
			result := manager.Execute(t.Context(), spec)
			if err := result.Validate(); err != nil || result.Error != nil || result.StatusCode != http.StatusOK || result.UpstreamProtocol != tc.protocol {
				t.Fatalf("result = %+v, execution error = %+v, validation = %v", result, result.Error, err)
			}
		})
	}
}
