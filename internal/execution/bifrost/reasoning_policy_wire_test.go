package bifrost

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
)

func TestReasoningPolicyStreamFinalProviderWire(t *testing.T) {
	cases := []struct {
		channel                                        channel.ID
		client                                         protocol.Protocol
		operation                                      execution.Operation
		path, body, model, field, want, streamResponse string
		mode                                           execution.RouteMode
	}{
		{channel.OpenAI, protocol.OpenAICompletions, execution.OperationChatCompletion, "/v1/chat/completions", `{"model":"client-model","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high","stream":true}`, "gpt-5.4", "reasoning_effort", "high", "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", execution.RouteNative},
		{channel.OpenAI, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", `{"model":"client-model","input":"hi","reasoning":{"effort":"high"}}`, "gpt-5.4", "reasoning.effort", "high", openAIResponsesStreamFixture, execution.RouteNative},
		{channel.OpenAI, protocol.Anthropic, execution.OperationChatCompletion, "/v1/messages", `{"model":"client-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"output_config":{"effort":"high"}}`, "gpt-5.4", "reasoning.effort", "high", openAIResponsesStreamFixture, execution.RouteConverted},
		{channel.Anthropic, protocol.Anthropic, execution.OperationChatCompletion, "/v1/messages", `{"model":"client-model","max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"output_config":{"effort":"high"}}`, "claude-sonnet-4-6", "output_config.effort", "high", anthropicResponsesStreamFixture, execution.RouteNative},
		{channel.Anthropic, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", `{"model":"client-model","input":"hi","max_output_tokens":4096,"reasoning":{"effort":"high"}}`, "claude-sonnet-4-6", "output_config.effort", "high", anthropicResponsesStreamFixture, execution.RouteConverted},
		{channel.Gemini, protocol.Gemini, execution.OperationChatCompletion, "/v1beta/models/client-model:streamGenerateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH"}}}`, "gemini-3-flash", "generationConfig.thinkingConfig.thinkingLevel", "HIGH", geminiResponsesStreamFixture, execution.RouteNative},
		{channel.Gemini, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", `{"model":"client-model","input":"hi","reasoning":{"effort":"high"}}`, "gemini-3-flash", "generationConfig.thinkingConfig.thinkingLevel", "high", geminiResponsesStreamFixture, execution.RouteConverted},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s/%s/%s", tc.channel, tc.client, tc.mode), func(t *testing.T) {
			wire := make(chan map[string]any, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var payload map[string]any
				if err := json.Unmarshal(body, &payload); err != nil {
					t.Errorf("decode wire: %v", err)
				}
				wire <- payload
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, tc.streamResponse)
			}))
			defer server.Close()
			options := testRuntimeOptions{allowPrivateNetwork: true}
			switch tc.channel {
			case channel.OpenAI:
				options.openAIBaseURL = server.URL
			case channel.Anthropic:
				options.anthropicBaseURL = server.URL
			case channel.Gemini:
				options.geminiBaseURL = server.URL + "/v1beta"
			}
			runtime := newProtocolTestRuntime(t, options)
			spec := convertedSpec(tc.channel, tc.client, tc.operation, tc.path, []byte(tc.body)).Clone()
			spec.RouteMode, spec.UpstreamModel = tc.mode, tc.model
			spec = freezeTestAttempt(spec)
			result := runtime.ExecuteStream(t.Context(), spec, func(execution.StreamEvent) error { return nil })
			if result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("stream execute: status=%d error=%v", result.StatusCode, result.Error)
			}
			select {
			case payload := <-wire:
				value := any(payload)
				for _, part := range strings.Split(tc.field, ".") {
					parent, ok := value.(map[string]any)
					if !ok {
						t.Fatalf("wire %s missing", tc.field)
					}
					value = parent[part]
				}
				if value != tc.want {
					t.Fatalf("wire %s = %v, want %s", tc.field, value, tc.want)
				}
				if tc.channel != channel.Gemini && payload["stream"] != true {
					t.Fatalf("wire stream = %v, want true", payload["stream"])
				}
			default:
				t.Fatal("no provider HTTP request")
			}
		})
	}
}

func TestReasoningPolicyFinalProviderWire(t *testing.T) {
	cases := []struct {
		name                                                 string
		channel                                              channel.ID
		client                                               protocol.Protocol
		operation                                            execution.Operation
		path, body, model, wirePath, field, effort, response string
		mode                                                 execution.RouteMode
	}{
		{"chat native", channel.OpenAI, protocol.OpenAICompletions, execution.OperationChatCompletion, "/v1/chat/completions", `{"model":"client-model","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`, "gpt-5.4", "/v1/chat/completions", "reasoning_effort", "high", `{"id":"chat_1","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, execution.RouteNative},
		{"responses native", channel.OpenAI, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", `{"model":"client-model","input":"hi","reasoning":{"effort":"high"}}`, "gpt-5.4", "/v1/responses", "reasoning.effort", "high", openAIResponsesConvertedFixture, execution.RouteNative},
		{"responses converted", channel.OpenAI, protocol.Anthropic, execution.OperationChatCompletion, "/v1/messages", `{"model":"client-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"output_config":{"effort":"high"}}`, "gpt-5.4", "/v1/responses", "reasoning.effort", "high", openAIResponsesConvertedFixture, execution.RouteConverted},
		{"anthropic native", channel.Anthropic, protocol.Anthropic, execution.OperationChatCompletion, "/v1/messages", `{"model":"client-model","max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"output_config":{"effort":"high"}}`, "claude-sonnet-4-6", "/v1/messages", "output_config.effort", "high", anthropicResponsesConvertedFixture, execution.RouteNative},
		{"anthropic converted", channel.Anthropic, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", `{"model":"client-model","input":"hi","max_output_tokens":4096,"reasoning":{"effort":"high"}}`, "claude-sonnet-4-6", "/v1/messages", "output_config.effort", "high", anthropicResponsesConvertedFixture, execution.RouteConverted},
		{"gemini native", channel.Gemini, protocol.Gemini, execution.OperationChatCompletion, "/v1beta/models/client-model:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH"}}}`, "gemini-3-flash", "/v1beta/models/gemini-3-flash:generateContent", "generationConfig.thinkingConfig.thinkingLevel", "HIGH", geminiResponsesConvertedFixture, execution.RouteNative},
		{"gemini converted", channel.Gemini, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", `{"model":"client-model","input":"hi","reasoning":{"effort":"high"}}`, "gemini-3-flash", "/v1beta/models/gemini-3-flash:generateContent", "generationConfig.thinkingConfig.thinkingLevel", "high", geminiResponsesConvertedFixture, execution.RouteConverted},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			provider := map[channel.ID]string{channel.OpenAI: "openai", channel.Anthropic: "anthropic", channel.Gemini: "gemini"}[tc.channel]
			if err := reasoning.ValidateEffort(provider, tc.model, "high"); err != nil {
				t.Fatalf("policy validation: %v", err)
			}
			wire := make(chan map[string]any, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tc.wirePath {
					t.Errorf("wire path = %q, want %q", r.URL.Path, tc.wirePath)
				}
				body, _ := io.ReadAll(r.Body)
				var payload map[string]any
				if err := json.Unmarshal(body, &payload); err != nil {
					t.Errorf("decode wire: %v", err)
				}
				wire <- payload
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.response)
			}))
			defer server.Close()
			options := testRuntimeOptions{allowPrivateNetwork: true}
			switch tc.channel {
			case channel.OpenAI:
				options.openAIBaseURL = server.URL
			case channel.Anthropic:
				options.anthropicBaseURL = server.URL
			case channel.Gemini:
				options.geminiBaseURL = server.URL + "/v1beta"
			}
			runtime := newProtocolTestRuntime(t, options)
			spec := convertedSpec(tc.channel, tc.client, tc.operation, tc.path, []byte(tc.body)).Clone()
			spec.RouteMode, spec.UpstreamModel = tc.mode, tc.model
			spec = freezeTestAttempt(spec)
			result := runtime.Execute(t.Context(), spec)
			if result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("execute: status=%d error=%v", result.StatusCode, result.Error)
			}
			select {
			case payload := <-wire:
				value := any(payload)
				for _, part := range strings.Split(tc.field, ".") {
					parent, ok := value.(map[string]any)
					if !ok {
						t.Fatalf("wire field %s missing", tc.field)
					}
					value = parent[part]
				}
				if value != tc.effort {
					t.Fatalf("wire %s = %v, want %s", tc.field, value, tc.effort)
				}
			default:
				t.Fatal("no provider HTTP request")
			}
		})
	}
}
