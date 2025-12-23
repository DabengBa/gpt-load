package gemini

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"gpt-load/internal/transformer/model"

	"pgregory.net/rapid"
)

// Property 2: 出站转换请求格式正确性 (Gemini)
// For any InternalLLMRequest and target API format (Gemini), the generated HTTP request should:
// - Contain API key in query parameter
// - Contain correct Content-Type
// - Request body conforms to Gemini API format specification
// **Validates: Requirements 3.6**

func TestMessagesOutbound_TransformRequest_APIKeyInQueryParam(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request
		req := generateValidInternalRequest(t)

		// Generate random base URL and API key
		baseUrl := generateValidBaseURL(t)
		apiKey := rapid.StringMatching(`^AIza[a-zA-Z0-9_-]{35}$`).Draw(t, "apiKey")

		// Transform request using MessagesOutbound
		outbound := NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, baseUrl, apiKey)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify API key is in query parameter
		queryKey := httpReq.URL.Query().Get("key")
		if queryKey != apiKey {
			t.Fatalf("API key in query parameter mismatch: expected %q, got %q", apiKey, queryKey)
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

		// Verify URL contains model name and action
		urlStr := httpReq.URL.String()
		if !strings.Contains(urlStr, "/models/") {
			t.Fatalf("URL should contain /models/, got %s", urlStr)
		}
		if !strings.Contains(urlStr, ":generateContent") && !strings.Contains(urlStr, ":streamGenerateContent") {
			t.Fatalf("URL should contain :generateContent or :streamGenerateContent, got %s", urlStr)
		}
	})
}


func TestMessagesOutbound_TransformRequest_BodyFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request
		req := generateValidInternalRequest(t)

		// Generate random base URL and API key
		baseUrl := generateValidBaseURL(t)
		apiKey := rapid.StringMatching(`^AIza[a-zA-Z0-9_-]{35}$`).Draw(t, "apiKey")

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

		// Verify required field: contents
		if _, ok := parsedBody["contents"]; !ok {
			t.Fatalf("request body missing required field: contents")
		}

		// Verify contents is an array
		contents, ok := parsedBody["contents"].([]interface{})
		if !ok {
			t.Fatalf("contents should be an array")
		}

		// Count non-system messages in original request
		nonSystemMsgCount := 0
		for _, msg := range req.Messages {
			if msg.Role != "system" {
				nonSystemMsgCount++
			}
		}

		// Gemini contents should not include system messages (they go to systemInstruction)
		if len(contents) != nonSystemMsgCount {
			t.Fatalf("contents count mismatch: expected %d non-system messages, got %d", nonSystemMsgCount, len(contents))
		}

		// Verify each content has role and parts
		for i, content := range contents {
			contentMap, ok := content.(map[string]interface{})
			if !ok {
				t.Fatalf("content[%d] should be an object", i)
			}
			if _, ok := contentMap["role"]; !ok {
				t.Fatalf("content[%d] missing required field: role", i)
			}
			if _, ok := contentMap["parts"]; !ok {
				t.Fatalf("content[%d] missing required field: parts", i)
			}

			// Verify role is valid for Gemini (user or model)
			role, _ := contentMap["role"].(string)
			if role != "user" && role != "model" {
				t.Fatalf("content[%d] has invalid role for Gemini: %s", i, role)
			}

			// Verify parts is an array
			parts, ok := contentMap["parts"].([]interface{})
			if !ok {
				t.Fatalf("content[%d].parts should be an array", i)
			}
			if len(parts) == 0 {
				t.Fatalf("content[%d].parts should not be empty", i)
			}
		}

		// Verify system message is extracted to systemInstruction field if present
		hasSystemMessage := false
		for _, msg := range req.Messages {
			if msg.Role == "system" {
				hasSystemMessage = true
				break
			}
		}
		if hasSystemMessage {
			if _, ok := parsedBody["systemInstruction"]; !ok {
				t.Fatalf("systemInstruction field should be present when request has system message")
			}
		}
	})
}


func TestMessagesOutbound_TransformRequest_URLConstruction(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal request (non-streaming for this test)
		req := generateValidInternalRequest(t)
		// Force non-streaming to test generateContent URL
		stream := false
		req.Stream = &stream
		apiKey := "AIzaSyTest1234567890123456789012345"

		// Test various base URL formats
		baseURLFormats := []struct {
			input    string
			contains []string
		}{
			{"https://generativelanguage.googleapis.com", []string{"/v1beta/models/", ":generateContent", "key="}},
			{"https://generativelanguage.googleapis.com/", []string{"/v1beta/models/", ":generateContent", "key="}},
			{"https://generativelanguage.googleapis.com/v1beta", []string{"/v1beta/models/", ":generateContent", "key="}},
			{"https://generativelanguage.googleapis.com/v1", []string{"/v1/models/", ":generateContent", "key="}},
		}

		outbound := NewMessagesOutbound()
		ctx := context.Background()

		for _, tc := range baseURLFormats {
			httpReq, err := outbound.TransformRequest(ctx, req, tc.input, apiKey)
			if err != nil {
				t.Fatalf("TransformRequest failed for base URL %q: %v", tc.input, err)
			}

			urlStr := httpReq.URL.String()
			for _, expected := range tc.contains {
				if !strings.Contains(urlStr, expected) {
					t.Fatalf("URL for base URL %q should contain %q, got %s", tc.input, expected, urlStr)
				}
			}
		}
	})
}

func TestMessagesOutbound_TransformRequest_StreamingURL(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a request with streaming enabled
		req := generateValidInternalRequest(t)
		stream := true
		req.Stream = &stream

		baseUrl := "https://generativelanguage.googleapis.com"
		apiKey := "AIzaSyTest1234567890123456789012345"

		outbound := NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, baseUrl, apiKey)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify URL contains streamGenerateContent for streaming requests
		urlStr := httpReq.URL.String()
		if !strings.Contains(urlStr, ":streamGenerateContent") {
			t.Fatalf("URL should contain :streamGenerateContent for streaming requests, got %s", urlStr)
		}
	})
}


func TestMessagesOutbound_TransformRequest_ToolsFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a request with tools
		req := generateRequestWithTools(t)

		baseUrl := "https://generativelanguage.googleapis.com"
		apiKey := "AIzaSyTest1234567890123456789012345"

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

		if len(tools) != 1 {
			t.Fatalf("tools should have exactly 1 element (containing functionDeclarations), got %d", len(tools))
		}

		// Verify the tool contains functionDeclarations
		toolMap, ok := tools[0].(map[string]interface{})
		if !ok {
			t.Fatalf("tool[0] should be an object")
		}

		declarations, ok := toolMap["functionDeclarations"].([]interface{})
		if !ok {
			t.Fatalf("tool[0] should have functionDeclarations array")
		}

		if len(declarations) != len(req.Tools) {
			t.Fatalf("functionDeclarations count mismatch: expected %d, got %d", len(req.Tools), len(declarations))
		}

		// Verify each function declaration has required Gemini fields
		for i, decl := range declarations {
			declMap, ok := decl.(map[string]interface{})
			if !ok {
				t.Fatalf("functionDeclaration[%d] should be an object", i)
			}

			// Gemini uses name, description, parameters
			if _, ok := declMap["name"]; !ok {
				t.Fatalf("functionDeclaration[%d] missing required field: name", i)
			}

			// Verify name matches
			if declMap["name"] != req.Tools[i].Function.Name {
				t.Fatalf("functionDeclaration[%d] name mismatch: expected %s, got %v", i, req.Tools[i].Function.Name, declMap["name"])
			}
		}
	})
}

func TestMessagesOutbound_TransformRequest_RoleConversion(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a request with assistant messages
		req := generateValidInternalRequest(t)

		baseUrl := "https://generativelanguage.googleapis.com"
		apiKey := "AIzaSyTest1234567890123456789012345"

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

		contents, _ := parsedBody["contents"].([]interface{})

		// Build expected roles map (excluding system messages)
		expectedRoles := make([]string, 0)
		for _, msg := range req.Messages {
			if msg.Role == "system" {
				continue
			}
			expectedRole := msg.Role
			if expectedRole == "assistant" {
				expectedRole = "model"
			}
			expectedRoles = append(expectedRoles, expectedRole)
		}

		// Verify role conversion
		for i, content := range contents {
			contentMap, _ := content.(map[string]interface{})
			role, _ := contentMap["role"].(string)

			if role != expectedRoles[i] {
				t.Fatalf("content[%d] role mismatch: expected %s, got %s", i, expectedRoles[i], role)
			}
		}
	})
}


// generateValidInternalRequest generates a random valid InternalLLMRequest
func generateValidInternalRequest(t *rapid.T) *model.InternalLLMRequest {
	// Generate model name
	modelName := rapid.StringMatching(`^gemini-(1\.5|2\.0)-(pro|flash)[a-z0-9-]*$`).Draw(t, "model")

	// Generate messages (at least 1 user message)
	numMessages := rapid.IntRange(1, 5).Draw(t, "numMessages")
	messages := make([]model.Message, numMessages)

	// First message should be user (Gemini requires alternating user/model)
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

	// Optionally add max_tokens
	if rapid.Bool().Draw(t, "hasMaxTokens") {
		maxTokens := int64(rapid.IntRange(1, 8192).Draw(t, "maxTokens"))
		req.MaxTokens = &maxTokens
	}

	// Optionally add temperature
	if rapid.Bool().Draw(t, "hasTemperature") {
		temp := rapid.Float64Range(0.0, 2.0).Draw(t, "temperature")
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
		"https://generativelanguage.googleapis.com",
		"https://generativelanguage.googleapis.com/v1beta",
		"https://generativelanguage.googleapis.com/v1",
		"https://custom-api.example.com",
	}
	return domains[rapid.IntRange(0, len(domains)-1).Draw(t, "domainIndex")]
}
