package bifrost

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestNativeResponsesHistoryConvertsToChat(t *testing.T) {
	for _, test := range []struct {
		name, reasoning  string
		wantText, reject bool
	}{
		{"raw reasoning", `{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"original plan"}]}`, true, false},
		{"summary and encrypted", `{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"brief summary"}],"encrypted_content":"opaque"}`, false, false},
		{"strict target rejects reasoning", `{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"original plan"}]}`, true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			captured := make(chan []byte, 4)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/relay/v1/chat/completions" {
					t.Errorf("unexpected path %s", r.URL.Path)
				}
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				captured <- body
				if test.reject {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusBadRequest)
					_, _ = io.WriteString(w, `{"error":{"type":"invalid_request_error","code":"unsupported_parameter","message":"reasoning_content is not supported"}}`)
					return
				}
				writeSuccess(w, "answer")
			}))
			defer server.Close()
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			spec := openAIResponsesSpec(execution.OperationResponsesCreate, http.MethodPost, "/v1/responses")
			spec.ChannelID = string(channel.OpenAICompatible)
			spec.UpstreamModel = "gpt-6.1-sol"
			spec.TargetConfig, _ = json.Marshal(map[string]string{"base_url": server.URL + "/relay/v1"})
			spec.Body = []byte(`{"model":"worker","stream":false,"store":false,"input":[{"role":"user","content":"lookup"},` + test.reasoning + `,{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"tool result"},{"role":"user","content":"continue"}]}`)
			spec = freezeTestAttempt(spec)
			result := runtime.Execute(t.Context(), spec)
			if len(captured) != 1 {
				t.Fatalf("expected one upstream request, got %d; result=%+v", len(captured), result)
			}
			body := <-captured
			if !test.wantText && (strings.Contains(string(body), "brief summary") || strings.Contains(string(body), "opaque")) {
				t.Fatalf("summary/encrypted content leaked into Chat wire: %s", body)
			}
			var wire struct {
				Messages []map[string]json.RawMessage `json:"messages"`
			}
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatal(err)
			}
			var assistant, tool map[string]json.RawMessage
			for _, message := range wire.Messages {
				for _, key := range []string{"reasoning", "reasoning_details", "summary", "encrypted_content"} {
					if _, exists := message[key]; exists {
						t.Fatalf("unsupported details leaked into Chat wire: %s", body)
					}
				}
				if !test.wantText {
					if _, exists := message["reasoning_content"]; exists {
						t.Fatalf("summary/encrypted was recast as raw reasoning: %s", body)
					}
				}
				if string(message["role"]) == `"assistant"` {
					assistant = message
				}
				if string(message["role"]) == `"tool"` {
					tool = message
				}
			}
			if assistant == nil || tool == nil || string(tool["tool_call_id"]) != `"call_1"` || string(tool["content"]) != `"tool result"` {
				t.Fatalf("tool result pairing/content lost: %s", body)
			}
			var calls []struct {
				ID       string
				Function struct{ Name, Arguments string }
			}
			if err := json.Unmarshal(assistant["tool_calls"], &calls); err != nil || len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Function.Name != "lookup" || calls[0].Function.Arguments != "{}" {
				t.Fatalf("tool call pairing/arguments lost: %s", body)
			}
			if test.wantText {
				if string(assistant["reasoning_content"]) != `"original plan"` {
					t.Fatalf("raw reasoning was dropped or changed: %s", body)
				}
			}
			if result.UpstreamProtocol != protocol.OpenAICompletions {
				t.Fatalf("unexpected upstream protocol: %s", result.UpstreamProtocol)
			}
			if test.reject {
				if result.StatusCode != http.StatusBadRequest || result.Error == nil || result.Error.Kind != execution.ErrorKindHTTP || !strings.Contains(result.Error.Summary, "reasoning_content is not supported") {
					t.Fatalf("strict upstream rejection was hidden: %+v", result)
				}
			} else if result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("Chat replay failed: %+v", result)
			}
		})
	}
}
