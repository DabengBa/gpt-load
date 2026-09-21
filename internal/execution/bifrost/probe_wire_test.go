package bifrost

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/providers/bedrock"
	"github.com/maximhq/bifrost/core/schemas"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

// TestProbeWirePromptAndTokenValues proves every generative probe protocol
// sends the code-owned prompt, the correct token field, and a value > 1 where
// the scope requires it.  It replaces the old "ping" + max_tokens=1 fixture.
func TestProbeWirePromptAndTokenValues(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		channelID    channel.ID
		protocol     protocol.Protocol
		runtime      func(*testing.T, string) *testRuntime
		setTarget    bool // set the target base URL from the test server instead of runtime defaults
		response     string
		wantPath     string
		assertPrompt func(*testing.T, string) // body excerpt that must contain probe prompt
	}{
		{
			name: "openai_chat", channelID: channel.OpenAICompatible, protocol: protocol.OpenAICompletions,
			runtime: func(t *testing.T, _ string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			},
			setTarget: true,
			response:  `{"id":"chat_1","object":"chat.completion","created":1,"model":"probe-upstream","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`,
			wantPath:  "/chat/completions",
			assertPrompt: func(t *testing.T, body string) {
				if !contains(body, "What is 2 + 2?") {
					t.Errorf("probe prompt missing from body: %s", body)
				}
				if !contains(body, fmt.Sprintf(`"max_tokens":%d`, testProbeOutputTokens)) {
					t.Errorf("probe max_tokens=%d missing from body: %s", testProbeOutputTokens, body)
				}
				if contains(body, `"max_completion_tokens"`) {
					t.Errorf("max_completion_tokens must not appear on compatible Chat: %s", body)
				}
			},
		},
		{
			name: "multi_protocol_gateway_chat", channelID: channel.NewAPI, protocol: protocol.OpenAICompletions,
			runtime: func(t *testing.T, _ string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			},
			setTarget: true,
			response:  `{"id":"chat_1","object":"chat.completion","created":1,"model":"probe-upstream","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}]}`,
			wantPath:  "/v1/chat/completions",
			assertPrompt: func(t *testing.T, body string) {
				if !contains(body, "What is 2 + 2?") {
					t.Errorf("probe prompt missing from body: %s", body)
				}
				if !contains(body, fmt.Sprintf(`"max_tokens":%d`, testProbeOutputTokens)) {
					t.Errorf("probe max_tokens=%d missing from multi-protocol gateway Chat: %s", testProbeOutputTokens, body)
				}
				for _, field := range []string{`"max_completion_tokens"`, `"input"`, `"max_output_tokens"`} {
					if contains(body, field) {
						t.Errorf("multi-protocol gateway Chat must not contain %s: %s", field, body)
					}
				}
			},
		},
		{
			name: "anthropic_native", channelID: channel.Anthropic, protocol: protocol.Anthropic,
			runtime: func(t *testing.T, base string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, anthropicBaseURL: base})
			},
			response: `{"id":"msg_1","type":"message","role":"assistant","model":"probe-upstream","content":[{"type":"text","text":"4"}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":1}}`,
			wantPath: "/v1/messages",
			assertPrompt: func(t *testing.T, body string) {
				if !contains(body, "What is 2 + 2?") {
					t.Errorf("probe prompt missing from body: %s", body)
				}
			},
		},
		{
			name: "gemini_native", channelID: channel.Gemini, protocol: protocol.Gemini,
			runtime: func(t *testing.T, base string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, geminiBaseURL: base + "/v1beta"})
			},
			response: `{"candidates":[{"content":{"role":"model","parts":[{"text":"4"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":1,"totalTokenCount":21},"modelVersion":"probe-upstream"}`,
			wantPath: "/v1beta/models/probe-upstream:generateContent",
			assertPrompt: func(t *testing.T, body string) {
				if !contains(body, "What is 2 + 2?") {
					t.Errorf("probe prompt missing from body: %s", body)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int64
			var capturedBody string
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				calls.Add(1)
				if request.URL.Path != test.wantPath {
					t.Errorf("probe path = %s, want %s", request.URL.Path, test.wantPath)
				}
				raw, _ := io.ReadAll(request.Body)
				capturedBody = string(raw)
				test.assertPrompt(t, capturedBody)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, test.response)
			}))
			defer server.Close()

			runtime := test.runtime(t, server.URL)
			spec := utilitySpec(test.channelID, test.protocol, execution.OperationProbe, "", "", nil)
			spec.ClientModel = "probe-client"
			spec.UpstreamModel = "probe-upstream"
			if test.setTarget {
				target, _ := json.Marshal(map[string]string{"base_url": server.URL})
				spec.TargetConfig = target
				spec = freezeTestAttempt(spec)
			}
			result := runtime.Execute(context.Background(), spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
			}
		})
	}
}

func contains(s, substr string) bool {
	return len(substr) == 0 || s != "" && containsSubstring(s, substr)
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// TestProbeResponseTextEvidence proves that a valid probe response generates
// the operation-level evidence R4 requires: an attempt result with non-empty
// text that a caller can judge.  Each protocol's successful text extraction is
// shown with one happy-path wire fixture.
func TestProbeResponseTextEvidence(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		protocol protocol.Protocol
		body     string
		hasText  bool
	}{
		{name: "chat_has_text", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],"usage":{"total_tokens":21}}`, hasText: true},
		{name: "chat_text_content_part", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"4"}]}}]}`, hasText: true},
		{name: "chat_missing_object_marker", protocol: protocol.OpenAICompletions, body: `{"id":"r","choices":[{"message":{"role":"assistant","content":"4"}}]}`, hasText: false},
		{name: "chat_non_assistant_role", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[{"message":{"role":"user","content":"4"}}]}`, hasText: false},
		{name: "chat_tool_content_part", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[{"message":{"role":"assistant","content":[{"type":"tool_use","text":"4"}]}}]}`, hasText: false},
		{name: "chat_image_content_part", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[{"message":{"role":"assistant","content":[{"type":"image_url","text":"4"}]}}]}`, hasText: false},
		{name: "chat_empty_text_content_part", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"   "}]}}]}`, hasText: false},
		{name: "chat_empty_content", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}]}`, hasText: false},
		{name: "chat_null_content", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"stop"}]}`, hasText: false},
		{name: "chat_empty_choices", protocol: protocol.OpenAICompletions, body: `{"id":"r","object":"chat.completion","choices":[]}`, hasText: false},
		{name: "responses_has_text", protocol: protocol.OpenAIResponses, body: `{"id":"r","object":"response","output":[{"id":"m","type":"message","role":"assistant","content":[{"type":"output_text","text":"4","annotations":[]}]}],"status":"completed"}`, hasText: true},
		{name: "responses_empty_output", protocol: protocol.OpenAIResponses, body: `{"id":"r","object":"response","output":[],"status":"completed"}`, hasText: false},
		{name: "responses_null_text", protocol: protocol.OpenAIResponses, body: `{"id":"r","object":"response","output":[{"id":"m","type":"message","role":"assistant","content":[{"type":"output_text","text":null,"annotations":[]}]}],"status":"completed"}`, hasText: false},
		{name: "responses_wrong_content_type", protocol: protocol.OpenAIResponses, body: `{"id":"r","object":"response","output":[{"id":"m","type":"message","role":"assistant","content":[{"type":"input_text","text":"fake"}]}],"status":"completed"}`, hasText: false},
		{name: "anthropic_has_text", protocol: protocol.Anthropic, body: `{"id":"m","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"4"}],"stop_reason":"end_turn"}`, hasText: true},
		{name: "anthropic_empty_content", protocol: protocol.Anthropic, body: `{"id":"m","type":"message","role":"assistant","model":"claude","content":[]}`, hasText: false},
		{name: "anthropic_tool_use_text", protocol: protocol.Anthropic, body: `{"id":"m","type":"message","role":"assistant","content":[{"type":"tool_use","text":"fake"}],"stop_reason":"tool_use"}`, hasText: false},
		{name: "gemini_has_text", protocol: protocol.Gemini, body: `{"candidates":[{"content":{"parts":[{"text":"4"}],"role":"model"},"finishReason":"STOP"}],"modelVersion":"gemini"}`, hasText: true},
		{name: "gemini_empty_candidates", protocol: protocol.Gemini, body: `{"candidates":[]}`, hasText: false},
		{name: "malformed", protocol: protocol.OpenAICompletions, body: `not-json`, hasText: false},
		{name: "empty", protocol: protocol.OpenAICompletions, body: ``, hasText: false},
		{name: "valid_json_no_text_carrier", protocol: protocol.OpenAICompletions, body: `{"ok":true}`, hasText: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := probeResponseHasGeneratedText(test.protocol, []byte(test.body))
			if got != test.hasText {
				t.Errorf("probeResponseHasGeneratedText(%q) = %t, want %t", test.name, got, test.hasText)
			}
		})
	}
}

func TestProbeFullFlowRejectsMalformedChatEvidence(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		// A generated string without the Chat response marker is not evidence.
		_, _ = io.WriteString(writer, `{"unexpected":"payload"}`)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.OpenAICompatible, protocol.OpenAICompletions, execution.OperationProbe, "", "", nil)
	spec.ClientModel, spec.UpstreamModel = "probe-client", "probe-upstream"
	target, err := json.Marshal(map[string]string{"base_url": server.URL})
	if err != nil {
		t.Fatalf("marshal target: %v", err)
	}
	spec.TargetConfig = target
	spec = freezeTestAttempt(spec)

	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if calls.Load() != 1 || result.Error == nil || result.Error.Kind != execution.ErrorKindProvider ||
		result.StatusCode != http.StatusOK || result.ProbeAnswerPresent {
		t.Fatalf("calls/result = %d/%+v; want invalid probe evidence", calls.Load(), result)
	}
}

func TestProbeResponseTextEvidenceRejectsMismatchedProtocolCarriers(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		protocol protocol.Protocol
		body     string
	}{
		{name: "chat_rejects_responses_output", protocol: protocol.OpenAICompletions, body: `{"output":[{"content":[{"text":"wrong"}]}]}`},
		{name: "chat_rejects_completion_text", protocol: protocol.OpenAICompletions, body: `{"choices":[{"text":"wrong"}]}`},
		{name: "responses_rejects_chat_choices", protocol: protocol.OpenAIResponses, body: `{"choices":[{"message":{"content":"wrong"}}]}`},
		{name: "anthropic_rejects_gemini_candidates", protocol: protocol.Anthropic, body: `{"candidates":[{"content":{"parts":[{"text":"wrong"}]}}]}`},
		{name: "gemini_rejects_anthropic_content", protocol: protocol.Gemini, body: `{"content":[{"type":"text","text":"wrong"}]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if probeResponseHasGeneratedText(test.protocol, []byte(test.body)) {
				t.Fatalf("protocol %q accepted mismatched carrier: %s", test.protocol, test.body)
			}
		})
	}
}

func TestProbeFullFlowRejectsMismatchedProtocolCarriers(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		channelID channel.ID
		protocol  protocol.Protocol
		runtime   func(*testing.T, string) *testRuntime
		wrongBody string
		setTarget bool
	}{
		{
			name: "chat_receives_responses", channelID: channel.OpenAICompatible,
			protocol: protocol.OpenAICompletions,
			runtime: func(t *testing.T, _ string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			},
			wrongBody: `{"object":"response","output":[{"content":[{"text":"wrong"}]}]}`,
			setTarget: true,
		},
		{
			name: "responses_receives_chat", channelID: channel.OpenAI,
			protocol: protocol.OpenAIResponses,
			runtime: func(t *testing.T, base string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: base})
			},
			wrongBody: `{"object":"chat.completion","choices":[{"message":{"content":"wrong"}}]}`,
		},
		{
			name: "anthropic_receives_gemini", channelID: channel.Anthropic,
			protocol: protocol.Anthropic,
			runtime: func(t *testing.T, base string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, anthropicBaseURL: base})
			},
			wrongBody: `{"candidates":[{"content":{"parts":[{"text":"wrong"}]}}]}`,
		},
		{
			name: "gemini_receives_anthropic", channelID: channel.Gemini,
			protocol: protocol.Gemini,
			runtime: func(t *testing.T, base string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, geminiBaseURL: base + "/v1beta"})
			},
			wrongBody: `{"type":"message","content":[{"type":"text","text":"wrong"}]}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, test.wrongBody)
			}))
			defer server.Close()

			runtime := test.runtime(t, server.URL)
			spec := utilitySpec(test.channelID, test.protocol, execution.OperationProbe, "", "", nil)
			spec.ClientModel, spec.UpstreamModel = "probe-client", "probe-upstream"
			if test.setTarget {
				target, err := json.Marshal(map[string]string{"base_url": server.URL})
				if err != nil {
					t.Fatal(err)
				}
				spec.TargetConfig = target
				spec = freezeTestAttempt(spec)
			}
			result := runtime.Execute(context.Background(), spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if calls.Load() != 1 || result.StatusCode != http.StatusOK || result.ProbeAnswerPresent {
				t.Fatalf("calls/result = %d/%+v; want mismatched carrier rejected as probe evidence", calls.Load(), result)
			}
		})
	}
}

// TestProbeExecutionRejectsMismatchedProtocolBeforeDispatch proves direct
// execution cannot bypass the channel's explicit probe route.
func TestProbeExecutionRejectsMismatchedProtocolBeforeDispatch(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
	spec := utilitySpec(channel.OpenAI, protocol.OpenAICompletions, execution.OperationProbe, "", "", nil)
	spec.ClientModel, spec.UpstreamModel = "probe-client", "probe-upstream"
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if calls.Load() != 0 || result.DispatchState != execution.DispatchNotSent || result.Error == nil {
		t.Fatalf("calls/result = %d/%+v; want contract rejection before dispatch", calls.Load(), result)
	}
}

func TestNonGenerativeProbeIsUnsupportedBeforeDispatch(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.OpenAICompatible, protocol.OpenAIEmbeddings, execution.OperationProbe, "", "", nil)
	target, err := json.Marshal(map[string]string{"base_url": server.URL + "/v1"})
	if err != nil {
		t.Fatalf("marshal target: %v", err)
	}
	spec.TargetConfig = target
	spec = freezeTestAttempt(spec)
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if calls.Load() != 0 || result.DispatchState != execution.DispatchNotSent || result.Error == nil {
		t.Fatalf("calls/result = %d/%+v; want unsupported probe before dispatch", calls.Load(), result)
	}
}

// TestProbeTextEvidenceFromLiveAttempt proves that a full probe execution flow
// correctly extracts or rejects text evidence for each protocol.
func TestProbeTextEvidenceFromLiveAttempt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		channelID  channel.ID
		protocol   protocol.Protocol
		runtime    func(*testing.T, string) *testRuntime
		setTarget  bool
		responseOK string // a body the upstream may return
		responseNo string // an empty-text body
	}{
		{
			name: "openai_chat", channelID: channel.OpenAICompatible, protocol: protocol.OpenAICompletions,
			runtime: func(t *testing.T, _ string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			},
			setTarget:  true,
			responseOK: `{"id":"chat_1","object":"chat.completion","created":1,"model":"probe","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],"usage":{"total_tokens":21}}`,
			responseNo: `{"id":"chat_1","object":"chat.completion","created":1,"model":"probe","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}]}`,
		},
		{
			name: "anthropic", channelID: channel.Anthropic, protocol: protocol.Anthropic,
			runtime: func(t *testing.T, base string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, anthropicBaseURL: base})
			},
			responseOK: `{"id":"msg_1","type":"message","role":"assistant","model":"probe","content":[{"type":"text","text":"4"}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":1}}`,
			responseNo: `{"id":"msg_1","type":"message","role":"assistant","model":"probe","content":[],"stop_reason":"end_turn"}`,
		},
		{
			name: "gemini", channelID: channel.Gemini, protocol: protocol.Gemini,
			runtime: func(t *testing.T, base string) *testRuntime {
				return newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, geminiBaseURL: base + "/v1beta"})
			},
			responseOK: `{"candidates":[{"content":{"role":"model","parts":[{"text":"4"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":1,"totalTokenCount":21},"modelVersion":"probe"}`,
			responseNo: `{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"modelVersion":"probe"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name+"/text_accepted", func(t *testing.T) {
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, test.responseOK)
			}))
			defer server.Close()

			runtime := test.runtime(t, server.URL)
			spec := utilitySpec(test.channelID, test.protocol, execution.OperationProbe, "", "", nil)
			spec.ClientModel = "probe-client"
			spec.UpstreamModel = "probe-upstream"
			if test.setTarget {
				target, _ := json.Marshal(map[string]string{"base_url": server.URL})
				spec.TargetConfig = target
				spec = freezeTestAttempt(spec)
			}
			result := runtime.Execute(context.Background(), spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
			}
		})

		t.Run(test.name+"/empty_text_rejected", func(t *testing.T) {
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, test.responseNo)
			}))
			defer server.Close()

			runtime := test.runtime(t, server.URL)
			spec := utilitySpec(test.channelID, test.protocol, execution.OperationProbe, "", "", nil)
			spec.ClientModel = "probe-client"
			spec.UpstreamModel = "probe-upstream"
			if test.setTarget {
				target, _ := json.Marshal(map[string]string{"base_url": server.URL})
				spec.TargetConfig = target
				spec = freezeTestAttempt(spec)
			}
			result := runtime.Execute(context.Background(), spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK ||
				result.ProbeAnswerPresent || result.ProbeResponseInvalid {
				t.Fatalf("calls/result = %d/%+v (expected valid no-answer)", calls.Load(), result)
			}
		})
	}
}

// TestProbeSingleCallPerTarget proves that method, model, and protocol
// failures are terminal: one probe target executes at most one upstream
// request, never a second protocol probe or a cross-protocol retry.
func TestProbeSingleCallPerTarget(t *testing.T) {
	t.Parallel()

	t.Run("protocol_failure_never_dispatches", func(t *testing.T) {
		var calls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			calls.Add(1)
			t.Errorf("unexpected probe request: %s %s", request.Method, request.URL.Path)
			writer.WriteHeader(http.StatusMethodNotAllowed)
		}))
		defer server.Close()

		// Multi-protocol gateways only accept the Chat probe protocol; a
		// Responses probe target is unsupported and must not reach upstream.
		runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
		spec := utilitySpec(channel.NewAPI, protocol.OpenAIResponses, execution.OperationProbe, "", "", nil)
		spec.ClientModel = "probe-client"
		spec.UpstreamModel = "probe-upstream"
		result := runtime.Execute(context.Background(), spec)
		if err := result.Validate(); err != nil {
			t.Fatalf("result validation: %v; result=%+v", err, result)
		}
		if calls.Load() != 0 || result.Error == nil ||
			result.DispatchState != execution.DispatchNotSent {
			t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
		}
	})

	t.Run("upstream_failure_stops_after_one_request", func(t *testing.T) {
		var calls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			calls.Add(1)
			writer.WriteHeader(http.StatusInternalServerError)
		}))
		defer server.Close()

		runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, anthropicBaseURL: server.URL})
		spec := utilitySpec(channel.Anthropic, protocol.Anthropic, execution.OperationProbe, "", "", nil)
		spec.ClientModel = "probe-client"
		spec.UpstreamModel = "probe-upstream"
		result := runtime.Execute(context.Background(), spec)
		if err := result.Validate(); err != nil {
			t.Fatalf("result validation: %v; result=%+v", err, result)
		}
		if calls.Load() != 1 || result.Error == nil || result.DispatchState != execution.DispatchMaybeSent {
			t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
		}
	})
}

// TestNativeOpenAIResponsesProbeSendsResponsesWire proves the native OpenAI
// Responses probe keeps Responses semantics: /responses endpoint, input, and
// max_output_tokens=128 without any Chat max_tokens field.
func TestNativeOpenAIResponsesProbeSendsResponsesWire(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	const response = `{"id":"resp_1","object":"response","created_at":123,"status":"completed","model":"probe-upstream","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"4","annotations":[]}]}],"usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21}}`
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		if request.Method != http.MethodPost || request.URL.Path != "/v1/responses" {
			t.Errorf("probe target = %s %s", request.Method, request.URL.Path)
		}
		if request.Header.Get("Authorization") != "Bearer "+testAPIKey {
			t.Errorf("Authorization = %q", request.Header.Get("Authorization"))
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read probe body: %v", err)
			return
		}
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode probe body: %v", err)
			return
		}
		if _, hasInput := payload["input"]; !hasInput {
			t.Errorf("Responses probe body has no input: %s", body)
		}
		if _, hasMessages := payload["messages"]; hasMessages {
			t.Errorf("Responses probe body must not carry Chat messages: %s", body)
		}
		if string(payload["max_output_tokens"]) != fmt.Sprint(testProbeOutputTokens) {
			t.Errorf("max_output_tokens = %s, want %d", payload["max_output_tokens"], testProbeOutputTokens)
		}
		if _, hasMaxTokens := payload["max_tokens"]; hasMaxTokens {
			t.Errorf("Responses probe must not carry max_tokens: %s", body)
		}
		if _, hasMaxCompletionTokens := payload["max_completion_tokens"]; hasMaxCompletionTokens {
			t.Errorf("Responses probe must not carry max_completion_tokens: %s", body)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, response)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
	spec := utilitySpec(channel.OpenAI, protocol.OpenAIResponses, execution.OperationProbe, "", "", nil)
	spec.ClientModel = "probe-client"
	spec.UpstreamModel = "probe-upstream"
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK ||
		result.UpstreamProtocol != protocol.OpenAIResponses {
		t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
	}
}

// TestConvertedBedrockProbeUsesConverseWire proves the converted Bedrock probe
// is built for the Converse adapter, including its native token field.
func TestConvertedBedrockProbeUsesConverseWire(t *testing.T) {
	t.Parallel()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.AWSBedrock, protocol.OpenAICompletions, execution.OperationProbe, "", "", nil)
	spec.TargetConfig = json.RawMessage(`{"region":"us-east-1"}`)
	spec.Credential = execution.NewCredentialSnapshot(11, 1, 1, []byte(`{"api_key":"`+testAPIKey+`"}`))
	spec.ClientModel, spec.UpstreamModel = "probe-client", "anthropic.claude-3-haiku"
	spec = freezeTestAttempt(spec)
	prepared, failure := runtime.prepare(spec, false)
	if failure != nil {
		t.Fatalf("prepare() failure = %+v", failure)
	}
	if prepared.mode != channel.RouteConverted || prepared.request == nil {
		t.Fatalf("Bedrock probe preparation = %#v; want converted Chat request", prepared)
	}

	conversionContext := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer conversionContext.Cancel()
	wire, err := bedrock.ToBedrockChatCompletionRequest(conversionContext, prepared.request)
	if err != nil {
		t.Fatalf("convert Bedrock probe wire: %v", err)
	}
	if wire.ModelID != spec.UpstreamModel {
		t.Fatalf("Bedrock model ID = %q, want %q", wire.ModelID, spec.UpstreamModel)
	}
	raw, err := json.Marshal(wire)
	if err != nil {
		t.Fatalf("marshal Bedrock probe wire: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("decode Bedrock probe wire: %v", err)
	}
	if payload["messages"] == nil || payload["inferenceConfig"] == nil {
		t.Fatalf("Bedrock probe wire = %s", raw)
	}
	inferenceConfig, ok := payload["inferenceConfig"].(map[string]any)
	if !ok || inferenceConfig["maxTokens"] != float64(testProbeOutputTokens) {
		t.Fatalf("Bedrock inferenceConfig = %#v; want maxTokens=%d", payload["inferenceConfig"], testProbeOutputTokens)
	}
	if !bytes.Contains(raw, []byte("What is 2 + 2? Please answer briefly.")) {
		t.Fatalf("Bedrock probe prompt missing from wire: %s", raw)
	}
}

func TestNativeVertexProbeUsesGeminiWire(t *testing.T) {
	t.Parallel()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.GoogleVertex, protocol.Gemini, execution.OperationProbe, "", "", nil)
	spec.TargetConfig = json.RawMessage(`{"location":"us-central1"}`)
	spec.Credential = execution.NewCredentialSnapshot(12, 1, 1, []byte(`{"service_account_json":"{\"type\":\"service_account\",\"project_id\":\"probe-project\",\"client_email\":\"svc@example.iam.gserviceaccount.com\",\"private_key\":\"secret\"}"}`))
	spec.ClientModel, spec.UpstreamModel = "probe-client", "gemini-2.5-flash"
	spec = freezeTestAttempt(spec)
	prepared, failure := runtime.prepare(spec, false)
	if failure != nil {
		t.Fatalf("prepare() failure = %+v", failure)
	}
	if prepared.mode != channel.RouteNative || prepared.passthrough == nil {
		t.Fatalf("Vertex probe preparation = %#v; want native passthrough", prepared)
	}
	if prepared.passthrough.Path != "/publishers/google/models/gemini-2.5-flash:generateContent" {
		t.Fatalf("Vertex probe path = %q", prepared.passthrough.Path)
	}
	var payload map[string]any
	if err := json.Unmarshal(prepared.passthrough.Body, &payload); err != nil {
		t.Fatalf("decode Vertex probe wire: %v", err)
	}
	generationConfig, ok := payload["generationConfig"].(map[string]any)
	if !ok || generationConfig["maxOutputTokens"] != float64(testProbeOutputTokens) {
		t.Fatalf("Vertex generationConfig = %#v; want maxOutputTokens=%d", payload["generationConfig"], testProbeOutputTokens)
	}
	if !bytes.Contains(prepared.passthrough.Body, []byte("What is 2 + 2? Please answer briefly.")) {
		t.Fatalf("Vertex probe prompt missing from wire: %s", prepared.passthrough.Body)
	}
}

// TestConvertedProbeUsesFinalChatResponseShape proves a converted Azure probe
// uses the provider wire while judging the final OpenAI Chat response shape.
func TestConvertedProbeUsesFinalChatResponseShape(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		if request.URL.Path != "/openai/v1/chat/completions" || request.Header.Get("Api-Key") != testAPIKey {
			t.Errorf("probe target = %s %#v", request.URL.Path, request.Header)
		}
		body, _ := io.ReadAll(request.Body)
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode probe body: %v", err)
		}
		if payload["max_completion_tokens"] != float64(testProbeOutputTokens) || payload["messages"] == nil {
			t.Errorf("Azure Chat probe wire = %#v", payload)
		}
		if _, hasMaxTokens := payload["max_tokens"]; hasMaxTokens {
			t.Errorf("Azure Chat probe must not carry max_tokens: %s", body)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chat_1","object":"chat.completion","model":"probe-upstream","choices":[{"message":{"role":"assistant","content":"4"}}]}`)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.AzureOpenAI, protocol.OpenAICompletions, execution.OperationProbe, "", "", nil)
	spec.TargetConfig = json.RawMessage(`{"endpoint":"` + server.URL + `"}`)
	spec.Credential = execution.NewCredentialSnapshot(10, 1, 1, []byte(`{"api_key":"`+testAPIKey+`"}`))
	spec.ClientModel = "probe-client"
	spec.UpstreamModel = "probe-upstream"
	spec = freezeTestAttempt(spec)
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK {
		t.Fatalf("calls/result = %d/%+v error=%+v", calls.Load(), result, result.Error)
	}
	if !probeResponseHasGeneratedText(spec.ClientProtocol, result.Body) {
		t.Fatalf("final client response body carries no generated text: %s", result.Body)
	}
}
