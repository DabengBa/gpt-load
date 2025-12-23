package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"testing"

	"gpt-load/internal/transformer/model"

	"pgregory.net/rapid"
)

// Property 2: 出站转换请求格式正确性 (Anthropic)
// For any InternalLLMRequest and target API format (Anthropic), the generated HTTP request should:
// - Contain correct x-api-key header
// - Contain correct anthropic-version header
// - Contain correct Content-Type
// - Request body conforms to Anthropic API format specification
// **Validates: Requirements 3.5**

func TestMessagesOutbound_TransformRequest_HeaderFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request
		req := generateValidInternalRequest(t)

		// Generate random base URL and API key
		baseUrl := generateValidBaseURL(t)
		apiKey := rapid.StringMatching(`^sk-ant-[a-zA-Z0-9]{32,64}$`).Draw(t, "apiKey")

		// Transform request using MessagesOutbound
		outbound := NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, baseUrl, apiKey)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify x-api-key header
		xApiKey := httpReq.Header.Get("x-api-key")
		if xApiKey != apiKey {
			t.Fatalf("x-api-key header mismatch: expected %q, got %q", apiKey, xApiKey)
		}

		// Verify anthropic-version header
		anthropicVersion := httpReq.Header.Get("anthropic-version")
		if anthropicVersion != AnthropicVersion {
			t.Fatalf("anthropic-version header mismatch: expected %q, got %q", AnthropicVersion, anthropicVersion)
		}

		// Verify Content-Type header
		contentType := httpReq.Header.Get("Content-Type")
		if contentType != "application/json" {
			t.Fatalf("Content-Type header mismatch: expected %q, got %q", "application/json", contentType)
		}

		// Verify HTTP method
		if httpReq.Method != "POST" {
			t.Fatalf("HTTP method mismatch: expected POST, got %s", httpReq.Method)
		}
	})
}


func TestMessagesOutbound_TransformRequest_BodyFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request
		req := generateValidInternalRequest(t)

		// Generate random base URL and API key
		baseUrl := generateValidBaseURL(t)
		apiKey := rapid.StringMatching(`^sk-ant-[a-zA-Z0-9]{32,64}$`).Draw(t, "apiKey")

		// Transform request using MessagesOutbound
		outbound := NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, baseUrl, apiKey)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read and parse the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Verify body is valid JSON
		var parsedBody map[string]interface{}
		if err := json.Unmarshal(body, &parsedBody); err != nil {
			t.Fatalf("request body is not valid JSON: %v", err)
		}

		// Verify required fields exist
		if _, ok := parsedBody["model"]; !ok {
			t.Fatalf("request body missing required field: model")
		}
		if _, ok := parsedBody["messages"]; !ok {
			t.Fatalf("request body missing required field: messages")
		}
		if _, ok := parsedBody["max_tokens"]; !ok {
			t.Fatalf("request body missing required field: max_tokens")
		}

		// Verify model matches
		if parsedBody["model"] != req.Model {
			t.Fatalf("model mismatch: expected %s, got %v", req.Model, parsedBody["model"])
		}

		// Verify messages is an array
		messages, ok := parsedBody["messages"].([]interface{})
		if !ok {
			t.Fatalf("messages should be an array")
		}

		// Count non-system messages in original request
		nonSystemMsgCount := 0
		for _, msg := range req.Messages {
			if msg.Role != "system" {
				nonSystemMsgCount++
			}
		}

		// Anthropic messages should not include system messages (they go to system field)
		if len(messages) != nonSystemMsgCount {
			t.Fatalf("messages count mismatch: expected %d non-system messages, got %d", nonSystemMsgCount, len(messages))
		}

		// Verify each message has role and content
		for i, msg := range messages {
			msgMap, ok := msg.(map[string]interface{})
			if !ok {
				t.Fatalf("message[%d] should be an object", i)
			}
			if _, ok := msgMap["role"]; !ok {
				t.Fatalf("message[%d] missing required field: role", i)
			}
			if _, ok := msgMap["content"]; !ok {
				t.Fatalf("message[%d] missing required field: content", i)
			}

			// Verify role is valid for Anthropic (user or assistant)
			role, _ := msgMap["role"].(string)
			if role != "user" && role != "assistant" {
				t.Fatalf("message[%d] has invalid role for Anthropic: %s", i, role)
			}
		}

		// Verify system message is extracted to system field if present
		hasSystemMessage := false
		for _, msg := range req.Messages {
			if msg.Role == "system" {
				hasSystemMessage = true
				break
			}
		}
		if hasSystemMessage {
			if _, ok := parsedBody["system"]; !ok {
				t.Fatalf("system field should be present when request has system message")
			}
		}

		// Verify max_tokens is a positive number
		maxTokens, ok := parsedBody["max_tokens"].(float64)
		if !ok {
			t.Fatalf("max_tokens should be a number")
		}
		if maxTokens <= 0 {
			t.Fatalf("max_tokens should be positive, got %v", maxTokens)
		}
	})
}


func TestMessagesOutbound_TransformRequest_URLConstruction(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request
		req := generateValidInternalRequest(t)
		apiKey := "sk-ant-test1234567890"

		// Test various base URL formats
		baseURLFormats := []struct {
			input    string
			expected string
		}{
			{"https://api.anthropic.com", "https://api.anthropic.com/v1/messages"},
			{"https://api.anthropic.com/", "https://api.anthropic.com/v1/messages"},
			{"https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"},
			{"https://api.anthropic.com/v1/", "https://api.anthropic.com/v1/messages"},
			{"https://api.anthropic.com/v1/messages", "https://api.anthropic.com/v1/messages"},
		}

		outbound := NewMessagesOutbound()
		ctx := context.Background()

		for _, tc := range baseURLFormats {
			httpReq, err := outbound.TransformRequest(ctx, req, tc.input, apiKey)
			if err != nil {
				t.Fatalf("TransformRequest failed for base URL %q: %v", tc.input, err)
			}

			if httpReq.URL.String() != tc.expected {
				t.Fatalf("URL mismatch for base URL %q: expected %q, got %q", tc.input, tc.expected, httpReq.URL.String())
			}
		}
	})
}

func TestMessagesOutbound_TransformRequest_ToolsFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a request with tools
		req := generateRequestWithTools(t)

		baseUrl := "https://api.anthropic.com"
		apiKey := "sk-ant-test1234567890"

		outbound := NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, baseUrl, apiKey)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read and parse the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		var parsedBody map[string]interface{}
		if err := json.Unmarshal(body, &parsedBody); err != nil {
			t.Fatalf("request body is not valid JSON: %v", err)
		}

		// Verify tools are present and correctly formatted
		tools, ok := parsedBody["tools"].([]interface{})
		if !ok {
			t.Fatalf("tools should be an array")
		}

		if len(tools) != len(req.Tools) {
			t.Fatalf("tools count mismatch: expected %d, got %d", len(req.Tools), len(tools))
		}

		// Verify each tool has required Anthropic fields
		for i, tool := range tools {
			toolMap, ok := tool.(map[string]interface{})
			if !ok {
				t.Fatalf("tool[%d] should be an object", i)
			}

			// Anthropic uses name, description, input_schema (not function wrapper)
			if _, ok := toolMap["name"]; !ok {
				t.Fatalf("tool[%d] missing required field: name", i)
			}
			if _, ok := toolMap["input_schema"]; !ok {
				t.Fatalf("tool[%d] missing required field: input_schema", i)
			}

			// Verify name matches
			if toolMap["name"] != req.Tools[i].Function.Name {
				t.Fatalf("tool[%d] name mismatch: expected %s, got %v", i, req.Tools[i].Function.Name, toolMap["name"])
			}
		}
	})
}


// generateValidInternalRequest generates a random valid InternalLLMRequest
func generateValidInternalRequest(t *rapid.T) *model.InternalLLMRequest {
	// Generate model name
	modelName := rapid.StringMatching(`^claude-[a-z0-9-]+$`).Draw(t, "model")

	// Generate messages (at least 1 user message)
	numMessages := rapid.IntRange(1, 5).Draw(t, "numMessages")
	messages := make([]model.Message, numMessages)

	// First message should be user (Anthropic requires alternating user/assistant)
	roles := []string{"user", "assistant"}
	for i := 0; i < numMessages; i++ {
		var role string
		if i == 0 {
			role = "user"
		} else {
			// Alternate roles
			role = roles[i%2]
		}
		content := rapid.StringN(1, 100, 500).Draw(t, "content")

		messages[i] = model.Message{
			Role: role,
			Content: model.MessageContent{
				Content: &content,
			},
		}
	}

	// Optionally add system message at the beginning
	if rapid.Bool().Draw(t, "hasSystemMessage") {
		systemContent := rapid.StringN(1, 50, 200).Draw(t, "systemContent")
		systemMsg := model.Message{
			Role: "system",
			Content: model.MessageContent{
				Content: &systemContent,
			},
		}
		messages = append([]model.Message{systemMsg}, messages...)
	}

	req := &model.InternalLLMRequest{
		Model:    modelName,
		Messages: messages,
	}

	// Add max_tokens (required for Anthropic)
	maxTokens := int64(rapid.IntRange(1, 4096).Draw(t, "maxTokens"))
	req.MaxTokens = &maxTokens

	// Optionally add temperature
	if rapid.Bool().Draw(t, "hasTemperature") {
		temp := rapid.Float64Range(0.0, 1.0).Draw(t, "temperature")
		req.Temperature = &temp
	}

	// Optionally add stream
	if rapid.Bool().Draw(t, "hasStream") {
		stream := rapid.Bool().Draw(t, "stream")
		req.Stream = &stream
	}

	// Optionally add top_p
	if rapid.Bool().Draw(t, "hasTopP") {
		topP := rapid.Float64Range(0.0, 1.0).Draw(t, "topP")
		req.TopP = &topP
	}

	return req
}

// generateRequestWithTools generates a request with tools
func generateRequestWithTools(t *rapid.T) *model.InternalLLMRequest {
	req := generateValidInternalRequest(t)

	// Add tools
	numTools := rapid.IntRange(1, 3).Draw(t, "numTools")
	tools := make([]model.Tool, numTools)
	for i := 0; i < numTools; i++ {
		toolName := rapid.StringMatching(`^[a-z_][a-z0-9_]*$`).Draw(t, "toolName")
		tools[i] = model.Tool{
			Type: "function",
			Function: model.Function{
				Name:        toolName,
				Description: rapid.StringN(0, 50, 200).Draw(t, "toolDescription"),
				Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
			},
		}
	}
	req.Tools = tools

	return req
}

// generateValidBaseURL generates a random valid base URL
func generateValidBaseURL(t *rapid.T) string {
	domains := []string{
		"https://api.anthropic.com",
		"https://api.anthropic.com/v1",
		"https://custom-api.example.com",
		"https://proxy.example.org/anthropic",
	}
	return domains[rapid.IntRange(0, len(domains)-1).Draw(t, "domainIndex")]
}
