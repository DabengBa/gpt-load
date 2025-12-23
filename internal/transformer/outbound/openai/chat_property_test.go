package openai

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"gpt-load/internal/transformer/model"

	"pgregory.net/rapid"
)

// Property 2: 出站转换请求格式正确性 (OpenAI)
// For any InternalLLMRequest and target API format (OpenAI), the generated HTTP request should:
// - Contain correct Authorization Bearer header
// - Contain correct Content-Type
// - Request body conforms to OpenAI API format specification
// **Validates: Requirements 3.4**

func TestChatOutbound_TransformRequest_HeaderFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request
		req := generateValidInternalRequest(t)

		// Generate random base URL and API key
		baseUrl := generateValidBaseURL(t)
		apiKey := rapid.StringMatching(`^sk-[a-zA-Z0-9]{32,64}$`).Draw(t, "apiKey")

		// Transform request using ChatOutbound
		outbound := NewChatOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, baseUrl, apiKey)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify Authorization header format
		authHeader := httpReq.Header.Get("Authorization")
		expectedAuth := "Bearer " + apiKey
		if authHeader != expectedAuth {
			t.Fatalf("Authorization header mismatch: expected %q, got %q", expectedAuth, authHeader)
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

		// Verify URL contains /chat/completions
		if !strings.Contains(httpReq.URL.String(), "/chat/completions") {
			t.Fatalf("URL should contain /chat/completions, got %s", httpReq.URL.String())
		}
	})
}

func TestChatOutbound_TransformRequest_BodyFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request
		req := generateValidInternalRequest(t)

		// Generate random base URL and API key
		baseUrl := generateValidBaseURL(t)
		apiKey := rapid.StringMatching(`^sk-[a-zA-Z0-9]{32,64}$`).Draw(t, "apiKey")

		// Transform request using ChatOutbound
		outbound := NewChatOutbound()
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

		// Verify model matches
		if parsedBody["model"] != req.Model {
			t.Fatalf("model mismatch: expected %s, got %v", req.Model, parsedBody["model"])
		}

		// Verify messages is an array
		messages, ok := parsedBody["messages"].([]interface{})
		if !ok {
			t.Fatalf("messages should be an array")
		}
		if len(messages) != len(req.Messages) {
			t.Fatalf("messages count mismatch: expected %d, got %d", len(req.Messages), len(messages))
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
		}

		// Verify optional fields if present
		if req.Temperature != nil {
			if _, ok := parsedBody["temperature"]; !ok {
				t.Fatalf("request body missing temperature field when it was set")
			}
		}
		if req.MaxTokens != nil {
			if _, ok := parsedBody["max_tokens"]; !ok {
				t.Fatalf("request body missing max_tokens field when it was set")
			}
		}
		if req.Stream != nil {
			if _, ok := parsedBody["stream"]; !ok {
				t.Fatalf("request body missing stream field when it was set")
			}
		}
	})
}

func TestChatOutbound_TransformRequest_URLConstruction(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request
		req := generateValidInternalRequest(t)
		apiKey := "sk-test1234567890"

		// Test various base URL formats
		baseURLFormats := []struct {
			input    string
			expected string
		}{
			{"https://api.openai.com", "https://api.openai.com/v1/chat/completions"},
			{"https://api.openai.com/", "https://api.openai.com/v1/chat/completions"},
			{"https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"},
			{"https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"},
			{"https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1/chat/completions"},
		}

		outbound := NewChatOutbound()
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

// generateValidInternalRequest generates a random valid InternalLLMRequest
func generateValidInternalRequest(t *rapid.T) *model.InternalLLMRequest {
	// Generate model name
	modelName := rapid.StringMatching(`^(gpt-4|gpt-3\.5-turbo|gpt-4o)[a-z0-9-]*$`).Draw(t, "model")

	// Generate messages (at least 1)
	numMessages := rapid.IntRange(1, 5).Draw(t, "numMessages")
	messages := make([]model.Message, numMessages)

	roles := []string{"system", "user", "assistant"}
	for i := 0; i < numMessages; i++ {
		role := roles[rapid.IntRange(0, len(roles)-1).Draw(t, "roleIndex")]
		content := rapid.StringN(1, 100, 500).Draw(t, "content")

		messages[i] = model.Message{
			Role: role,
			Content: model.MessageContent{
				Content: &content,
			},
		}
	}

	req := &model.InternalLLMRequest{
		Model:    modelName,
		Messages: messages,
	}

	// Optionally add temperature
	if rapid.Bool().Draw(t, "hasTemperature") {
		temp := rapid.Float64Range(0.0, 2.0).Draw(t, "temperature")
		req.Temperature = &temp
	}

	// Optionally add max_tokens
	if rapid.Bool().Draw(t, "hasMaxTokens") {
		maxTokens := int64(rapid.IntRange(1, 4096).Draw(t, "maxTokens"))
		req.MaxTokens = &maxTokens
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

	// Optionally add tools
	if rapid.Bool().Draw(t, "hasTools") {
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
	}

	return req
}

// generateValidBaseURL generates a random valid base URL
func generateValidBaseURL(t *rapid.T) string {
	domains := []string{
		"https://api.openai.com",
		"https://api.openai.com/v1",
		"https://custom-api.example.com",
		"https://proxy.example.org/openai",
	}
	return domains[rapid.IntRange(0, len(domains)-1).Draw(t, "domainIndex")]
}
