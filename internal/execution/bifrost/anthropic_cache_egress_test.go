package bifrost

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestConvertedResponsesAnthropicAutoCachesLatestToolResult(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "unary", true: "stream"}[stream], func(t *testing.T) {
			wireBodies := make(chan []byte, 2)
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.Method != http.MethodPost || request.URL.Path != "/v1/messages" {
					t.Errorf("Anthropic target = %s %s", request.Method, request.URL.Path)
				}
				body, err := io.ReadAll(request.Body)
				if err != nil {
					t.Errorf("read Anthropic request: %v", err)
					writer.WriteHeader(http.StatusBadRequest)
					return
				}
				wireBodies <- body
				if stream {
					writer.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(writer, anthropicResponsesStreamFixture)
					return
				}
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, anthropicResponsesConvertedFixture)
			}))
			defer server.Close()

			runtime := newProtocolTestRuntime(t, testRuntimeOptions{
				allowPrivateNetwork: true,
				anthropicBaseURL:    server.URL,
			})
			var previous map[string]any
			for turn := 1; turn <= 2; turn++ {
				originalBody := responsesToolHistoryBody(t, turn)
				spec := convertedSpec(
					channel.Anthropic,
					protocol.OpenAIResponses,
					execution.OperationResponsesCreate,
					"/v1/responses",
					bytes.Clone(originalBody),
				)
				spec.UpstreamModel = "claude-sonnet-4.6"
				if stream {
					result := runtime.ExecuteStream(t.Context(), spec, func(execution.StreamEvent) error { return nil })
					if err := result.Validate(); err != nil || result.Error != nil {
						t.Fatalf("turn %d ExecuteStream() = %+v validation=%v", turn, result, err)
					}
				} else {
					result := runtime.Execute(t.Context(), spec)
					if err := result.Validate(); err != nil || result.Error != nil {
						t.Fatalf("turn %d Execute() = %+v validation=%v", turn, result, err)
					}
				}
				if !bytes.Equal(spec.Body, originalBody) {
					t.Fatalf("turn %d mutated the caller's request body", turn)
				}
				wireBody := <-wireBodies
				assertAnthropicLatestToolResultMarker(t, wireBody, turn)
				current := decodeCacheTestBody(t, wireBody)
				removeCacheTestMarkers(current)
				if previous != nil {
					oldMessages := previous["messages"].([]any)
					newMessages := current["messages"].([]any)
					if len(newMessages) < len(oldMessages) || !reflect.DeepEqual(oldMessages, newMessages[:len(oldMessages)]) ||
						!reflect.DeepEqual(previous["system"], current["system"]) || !reflect.DeepEqual(previous["tools"], current["tools"]) {
						t.Fatal("extending the tool history changed the previously cached prefix")
					}
				}
				previous = current
			}
		})
	}
}

func TestConvertedResponsesAnthropicDoesNotOverrideCallerCacheIntent(t *testing.T) {
	tests := []struct {
		name       string
		prepare    func(map[string]any)
		wantCount  int
		wantTarget string
	}{
		{
			name: "input block marker",
			prepare: func(request map[string]any) {
				input := request["input"].([]any)
				input[0].(map[string]any)["content"] = []any{map[string]any{
					"type": "input_text", "text": "stable instructions",
					"cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"},
				}}
			},
			wantCount:  1,
			wantTarget: "input",
		},
		{
			name: "top-level tool-result marker",
			prepare: func(request map[string]any) {
				input := request["input"].([]any)
				input[len(input)-1].(map[string]any)["cache_control"] = map[string]any{
					"type": "ephemeral", "ttl": "1h",
				}
			},
			wantCount:  1,
			wantTarget: "tool_result",
		},
		{
			name: "four marked tools",
			prepare: func(request map[string]any) {
				tools := make([]any, 4)
				for index := range tools {
					tools[index] = map[string]any{
						"type": "function", "name": "tool_" + string(rune('a'+index)),
						"parameters":    map[string]any{"type": "object", "properties": map[string]any{}},
						"cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"},
					}
				}
				request["tools"] = tools
			},
			wantCount:  4,
			wantTarget: "tools",
		},
		{
			name: "OpenAI cache options do not disable Anthropic",
			prepare: func(request map[string]any) {
				request["prompt_cache_options"] = map[string]any{"mode": "explicit"}
			},
			wantCount: 1,
		},
	}

	for _, stream := range []bool{false, true} {
		for _, test := range tests {
			t.Run(map[bool]string{false: "unary/", true: "stream/"}[stream]+test.name, func(t *testing.T) {
				wireBody := make(chan []byte, 1)
				server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
					body, err := io.ReadAll(request.Body)
					if err != nil {
						t.Errorf("read Anthropic request: %v", err)
						writer.WriteHeader(http.StatusBadRequest)
						return
					}
					wireBody <- body
					if stream {
						writer.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(writer, anthropicResponsesStreamFixture)
						return
					}
					writer.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(writer, anthropicResponsesConvertedFixture)
				}))
				defer server.Close()

				request := responsesToolHistory(1)
				test.prepare(request)
				body := mustMarshalCacheTestBody(t, request)
				runtime := newProtocolTestRuntime(t, testRuntimeOptions{
					allowPrivateNetwork: true,
					anthropicBaseURL:    server.URL,
				})
				spec := convertedSpec(channel.Anthropic, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", body)
				spec.UpstreamModel = "claude-sonnet-4.6"
				if stream {
					result := runtime.ExecuteStream(t.Context(), spec, func(execution.StreamEvent) error { return nil })
					if err := result.Validate(); err != nil || result.Error != nil {
						t.Fatalf("ExecuteStream() = %+v validation=%v", result, err)
					}
				} else {
					result := runtime.Execute(t.Context(), spec)
					if err := result.Validate(); err != nil || result.Error != nil {
						t.Fatalf("Execute() = %+v validation=%v", result, err)
					}
				}

				wire := decodeCacheTestBody(t, <-wireBody)
				markers := collectAnthropicCacheMarkers(wire)
				if len(markers) != test.wantCount {
					t.Fatalf("wire cache marker count = %d, want %d; body=%s", len(markers), test.wantCount, mustMarshalCacheTestBody(t, wire))
				}
				for _, marker := range markers {
					if marker["type"] != "ephemeral" || marker["ttl"] != "1h" {
						if test.name != "OpenAI cache options do not disable Anthropic" {
							t.Errorf("caller cache marker changed: %#v", marker)
						}
					}
				}
				if test.wantTarget == "tool_result" {
					assertAnthropicToolResultHasExplicitTTL(t, wire, "call_1", "1h")
				}
				if test.wantTarget == "tools" {
					assertFourAnthropicToolMarkers(t, wire)
				}
			})
		}
	}
}

func TestConvertedResponsesOpenAIAttemptDoesNotInheritAnthropicCacheMarker(t *testing.T) {
	var wireBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			t.Errorf("OpenAI target = %s", request.URL.Path)
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read OpenAI request: %v", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		wireBody = bytes.Clone(body)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, openAIResponsesConvertedFixture)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{
		allowPrivateNetwork: true,
		openAIBaseURL:       server.URL,
	})
	originalBody := responsesToolHistoryBody(t, 2)
	spec := convertedSpec(channel.OpenAI, protocol.OpenAIResponses, execution.OperationResponsesCreate, "/v1/responses", bytes.Clone(originalBody))
	spec.UpstreamModel = "gpt-4o"
	result := runtime.Execute(t.Context(), spec)
	if err := result.Validate(); err != nil || result.Error != nil {
		t.Fatalf("Execute() = %+v validation=%v", result, err)
	}
	if !bytes.Equal(spec.Body, originalBody) {
		t.Fatal("OpenAI attempt mutated the caller's request body")
	}
	if count := len(collectAnthropicCacheMarkers(decodeCacheTestBody(t, wireBody))); count != 0 {
		t.Fatalf("OpenAI wire request has %d cache markers; body=%s", count, wireBody)
	}
}

func assertAnthropicLatestToolResultMarker(t *testing.T, wireBody []byte, turn int) {
	t.Helper()
	wire := decodeCacheTestBody(t, wireBody)
	input := responsesToolHistory(turn)["input"].([]any)
	latest := input[len(input)-1].(map[string]any)
	callID := latest["call_id"].(string)
	assertAnthropicToolResultHasExplicitTTL(t, wire, callID, "")
	markers := collectAnthropicCacheMarkers(wire)
	if len(markers) != 1 {
		t.Fatalf("turn %d wire cache marker count = %d, want one marker on the latest tool_result; body=%s", turn, len(markers), wireBody)
	}
	for _, messageValue := range wire["messages"].([]any) {
		message := messageValue.(map[string]any)
		blocks, ok := message["content"].([]any)
		if !ok {
			continue
		}
		for _, blockValue := range blocks {
			block := blockValue.(map[string]any)
			if block["type"] == "tool_result" && block["tool_use_id"] != callID && block["cache_control"] != nil {
				t.Errorf("earlier tool_result %q unexpectedly moved to cache boundary", block["tool_use_id"])
			}
		}
	}
}

func assertAnthropicToolResultHasExplicitTTL(t *testing.T, wire map[string]any, callID, ttl string) {
	t.Helper()
	for _, messageValue := range wire["messages"].([]any) {
		message := messageValue.(map[string]any)
		blocks, ok := message["content"].([]any)
		if !ok {
			continue
		}
		for _, blockValue := range blocks {
			block := blockValue.(map[string]any)
			if block["type"] != "tool_result" || block["tool_use_id"] != callID {
				continue
			}
			marker, ok := block["cache_control"].(map[string]any)
			if !ok || marker["type"] != "ephemeral" || (ttl != "" && marker["ttl"] != ttl) {
				t.Fatalf("tool_result %q cache_control = %#v, want ephemeral ttl %q; block=%#v", callID, block["cache_control"], ttl, block)
			}
			if block["content"] != "tool result "+callID {
				t.Fatalf("tool_result %q content = %#v, want complete result", callID, block["content"])
			}
			return
		}
	}
	t.Fatalf("wire has no tool_result for %q; messages=%#v", callID, wire["messages"])
}

func assertFourAnthropicToolMarkers(t *testing.T, wire map[string]any) {
	t.Helper()
	tools, ok := wire["tools"].([]any)
	if !ok || len(tools) != 4 {
		t.Fatalf("Anthropic tools = %#v, want four tools", wire["tools"])
	}
	for index, value := range tools {
		tool := value.(map[string]any)
		wantName := "tool_" + string(rune('a'+index))
		marker, ok := tool["cache_control"].(map[string]any)
		if tool["name"] != wantName || !ok || marker["type"] != "ephemeral" || marker["ttl"] != "1h" {
			t.Errorf("tool %d marker = %#v; want %s with its original ephemeral 1h marker", index, tool, wantName)
		}
	}
}

func responsesToolHistoryBody(t *testing.T, turn int) []byte {
	t.Helper()
	return mustMarshalCacheTestBody(t, responsesToolHistory(turn))
}

func responsesToolHistory(turn int) map[string]any {
	input := []any{
		map[string]any{"role": "developer", "content": "stable instructions"},
		map[string]any{"role": "user", "content": "first question"},
		map[string]any{"type": "function_call", "status": "completed", "call_id": "call_1", "name": "lookup", "arguments": `{"query":"first"}`},
		map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "tool result call_1"},
	}
	if turn > 1 {
		input = append(input,
			map[string]any{"role": "assistant", "content": "the first result is ready"},
			map[string]any{"role": "user", "content": "second question"},
			map[string]any{"type": "function_call", "status": "completed", "call_id": "call_2", "name": "lookup", "arguments": `{"query":"second"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_2", "output": "tool result call_2"},
		)
	}
	return map[string]any{
		"model":                  "client-model",
		"prompt_cache_key":       "shared-agent-session",
		"prompt_cache_retention": "24h",
		"store":                  false,
		"max_output_tokens":      32,
		"input":                  input,
		"tools": []any{map[string]any{
			"type": "function", "name": "lookup", "description": "Look up a value",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{}},
		}},
	}
}

func mustMarshalCacheTestBody(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal cache test body: %v", err)
	}
	return body
}

func decodeCacheTestBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(body, &value); err != nil {
		t.Fatalf("decode cache test body: %v; body=%s", err, body)
	}
	return value
}

func collectAnthropicCacheMarkers(value any) []map[string]any {
	var markers []map[string]any
	var walk func(any)
	walk = func(value any) {
		switch value := value.(type) {
		case map[string]any:
			for key, child := range value {
				if key == "cache_control" {
					if marker, ok := child.(map[string]any); ok {
						markers = append(markers, marker)
					}
				}
				walk(child)
			}
		case []any:
			for _, child := range value {
				walk(child)
			}
		}
	}
	walk(value)
	return markers
}

func removeCacheTestMarkers(value any) {
	switch v := value.(type) {
	case map[string]any:
		delete(v, "cache_control")
		for _, nested := range v {
			removeCacheTestMarkers(nested)
		}
	case []any:
		for _, nested := range v {
			removeCacheTestMarkers(nested)
		}
	}
}
