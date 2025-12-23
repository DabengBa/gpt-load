package transformer

import (
	"encoding/json"
	"testing"

	"gpt-load/internal/transformer/inbound"

	"pgregory.net/rapid"
)

// Property 5: 格式检测正确性
// For any request path and body combination, the format detector should
// correctly identify the client's API format (OpenAI Chat, OpenAI Response, Anthropic).
// **Validates: Requirements 6.1**

// TestDetectFormat_PathBased_OpenAIChat tests that OpenAI Chat paths are correctly detected
func TestDetectFormat_PathBased_OpenAIChat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate various OpenAI Chat path patterns
		pathPatterns := []string{
			"/v1/chat/completions",
			"/chat/completions",
			"/api/v1/chat/completions",
			"/proxy/v1/chat/completions",
		}

		idx := rapid.IntRange(0, len(pathPatterns)-1).Draw(t, "pathIndex")
		path := pathPatterns[idx]

		// Add optional prefix/suffix
		prefix := rapid.StringMatching(`^[a-z0-9/]*`).Draw(t, "prefix")
		suffix := rapid.StringMatching(`^[a-z0-9?=&]*`).Draw(t, "suffix")
		fullPath := prefix + path + suffix

		result, err := detector.DetectFormat(fullPath, nil)
		if err != nil {
			t.Fatalf("DetectFormat(%q, nil) returned error: %v", fullPath, err)
		}

		if result != inbound.InboundTypeOpenAIChat {
			t.Fatalf("DetectFormat(%q, nil) = %v, expected InboundTypeOpenAIChat", fullPath, result)
		}
	})
}

// TestDetectFormat_PathBased_Anthropic tests that Anthropic paths are correctly detected
func TestDetectFormat_PathBased_Anthropic(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate various Anthropic path patterns
		pathPatterns := []string{
			"/v1/messages",
			"/messages",
			"/api/v1/messages",
			"/proxy/v1/messages",
		}

		idx := rapid.IntRange(0, len(pathPatterns)-1).Draw(t, "pathIndex")
		path := pathPatterns[idx]

		result, err := detector.DetectFormat(path, nil)
		if err != nil {
			t.Fatalf("DetectFormat(%q, nil) returned error: %v", path, err)
		}

		if result != inbound.InboundTypeAnthropic {
			t.Fatalf("DetectFormat(%q, nil) = %v, expected InboundTypeAnthropic", path, result)
		}
	})
}

// TestDetectFormat_PathBased_OpenAIResponse tests that OpenAI Response paths are correctly detected
func TestDetectFormat_PathBased_OpenAIResponse(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate various OpenAI Response path patterns
		pathPatterns := []string{
			"/v1/responses",
			"/responses",
			"/api/v1/responses",
		}

		idx := rapid.IntRange(0, len(pathPatterns)-1).Draw(t, "pathIndex")
		path := pathPatterns[idx]

		result, err := detector.DetectFormat(path, nil)
		if err != nil {
			t.Fatalf("DetectFormat(%q, nil) returned error: %v", path, err)
		}

		if result != inbound.InboundTypeOpenAIResponse {
			t.Fatalf("DetectFormat(%q, nil) = %v, expected InboundTypeOpenAIResponse", path, result)
		}
	})
}

// TestDetectFormat_BodyBased_Anthropic tests that Anthropic body format is correctly detected
func TestDetectFormat_BodyBased_Anthropic(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate Anthropic-style request body with content blocks
		textContent := rapid.StringMatching(`^[a-zA-Z0-9 .,!?]+`).Draw(t, "textContent")
		model := rapid.StringMatching(`^claude-[0-9]+-[a-z]+`).Draw(t, "model")

		anthropicBody := map[string]any{
			"model":      model,
			"max_tokens": rapid.IntRange(100, 4096).Draw(t, "maxTokens"),
			"messages": []map[string]any{
				{
					"role": "user",
					"content": []map[string]any{
						{
							"type": "text",
							"text": textContent,
						},
					},
				},
			},
		}

		bodyBytes, err := json.Marshal(anthropicBody)
		if err != nil {
			t.Fatalf("Failed to marshal Anthropic body: %v", err)
		}

		// Use a generic path that doesn't indicate format
		result, err := detector.DetectFormat("/api/generate", bodyBytes)
		if err != nil {
			t.Fatalf("DetectFormat returned error: %v", err)
		}

		if result != inbound.InboundTypeAnthropic {
			t.Fatalf("DetectFormat with Anthropic body = %v, expected InboundTypeAnthropic", result)
		}
	})
}

// TestDetectFormat_BodyBased_AnthropicWithSystem tests Anthropic detection via system field
func TestDetectFormat_BodyBased_AnthropicWithSystem(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate Anthropic-style request body with top-level system field
		systemPrompt := rapid.StringMatching(`^[a-zA-Z0-9 .,!?]+`).Draw(t, "systemPrompt")
		userMessage := rapid.StringMatching(`^[a-zA-Z0-9 .,!?]+`).Draw(t, "userMessage")
		model := rapid.StringMatching(`^claude-[0-9]+-[a-z]+`).Draw(t, "model")

		anthropicBody := map[string]any{
			"model":      model,
			"max_tokens": rapid.IntRange(100, 4096).Draw(t, "maxTokens"),
			"system":     systemPrompt,
			"messages": []map[string]any{
				{
					"role":    "user",
					"content": userMessage,
				},
			},
		}

		bodyBytes, err := json.Marshal(anthropicBody)
		if err != nil {
			t.Fatalf("Failed to marshal Anthropic body: %v", err)
		}

		// Use a generic path that doesn't indicate format
		result, err := detector.DetectFormat("/api/generate", bodyBytes)
		if err != nil {
			t.Fatalf("DetectFormat returned error: %v", err)
		}

		if result != inbound.InboundTypeAnthropic {
			t.Fatalf("DetectFormat with Anthropic system field = %v, expected InboundTypeAnthropic", result)
		}
	})
}

// TestDetectFormat_BodyBased_OpenAIChat tests that OpenAI Chat body format is correctly detected
func TestDetectFormat_BodyBased_OpenAIChat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate OpenAI-style request body with string content
		userMessage := rapid.StringMatching(`^[a-zA-Z0-9 .,!?]+`).Draw(t, "userMessage")
		model := rapid.StringMatching(`^gpt-[0-9]+-[a-z]+`).Draw(t, "model")

		openaiBody := map[string]any{
			"model": model,
			"messages": []map[string]any{
				{
					"role":    "user",
					"content": userMessage, // String content, not array
				},
			},
		}

		bodyBytes, err := json.Marshal(openaiBody)
		if err != nil {
			t.Fatalf("Failed to marshal OpenAI body: %v", err)
		}

		// Use a generic path that doesn't indicate format
		result, err := detector.DetectFormat("/api/generate", bodyBytes)
		if err != nil {
			t.Fatalf("DetectFormat returned error: %v", err)
		}

		// Should default to OpenAI Chat since it doesn't have Anthropic-specific features
		if result != inbound.InboundTypeOpenAIChat {
			t.Fatalf("DetectFormat with OpenAI body = %v, expected InboundTypeOpenAIChat", result)
		}
	})
}

// TestDetectFormat_BodyBased_OpenAIResponse tests OpenAI Response format detection
func TestDetectFormat_BodyBased_OpenAIResponse(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate OpenAI Response-style request body with "input" field
		inputText := rapid.StringMatching(`^[a-zA-Z0-9 .,!?]+`).Draw(t, "inputText")
		model := rapid.StringMatching(`^gpt-[0-9]+-[a-z]+`).Draw(t, "model")

		responseBody := map[string]any{
			"model": model,
			"input": inputText,
		}

		bodyBytes, err := json.Marshal(responseBody)
		if err != nil {
			t.Fatalf("Failed to marshal OpenAI Response body: %v", err)
		}

		// Use a generic path that doesn't indicate format
		result, err := detector.DetectFormat("/api/generate", bodyBytes)
		if err != nil {
			t.Fatalf("DetectFormat returned error: %v", err)
		}

		if result != inbound.InboundTypeOpenAIResponse {
			t.Fatalf("DetectFormat with OpenAI Response body = %v, expected InboundTypeOpenAIResponse", result)
		}
	})
}

// TestDetectFormat_EmptyBody_DefaultsToOpenAI tests that empty body defaults to OpenAI Chat
func TestDetectFormat_EmptyBody_DefaultsToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate a generic path that doesn't indicate format
		genericPaths := []string{
			"/api/generate",
			"/proxy",
			"/llm",
			"/completion",
		}

		idx := rapid.IntRange(0, len(genericPaths)-1).Draw(t, "pathIndex")
		path := genericPaths[idx]

		result, err := detector.DetectFormat(path, nil)
		if err != nil {
			t.Fatalf("DetectFormat(%q, nil) returned error: %v", path, err)
		}

		if result != inbound.InboundTypeOpenAIChat {
			t.Fatalf("DetectFormat(%q, nil) = %v, expected InboundTypeOpenAIChat (default)", path, result)
		}
	})
}

// TestDetectFormat_InvalidJSON_DefaultsToOpenAI tests that invalid JSON defaults to OpenAI Chat
func TestDetectFormat_InvalidJSON_DefaultsToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate invalid JSON
		invalidJSON := rapid.StringMatching(`^[{}\[\]"':,a-z0-9]+`).Draw(t, "invalidJSON")

		result, err := detector.DetectFormat("/api/generate", []byte(invalidJSON))
		if err != nil {
			t.Fatalf("DetectFormat returned error: %v", err)
		}

		if result != inbound.InboundTypeOpenAIChat {
			t.Fatalf("DetectFormat with invalid JSON = %v, expected InboundTypeOpenAIChat (default)", result)
		}
	})
}

// TestDetectFormat_PathTakesPrecedence tests that path detection takes precedence over body
func TestDetectFormat_PathTakesPrecedence(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Create an Anthropic-style body
		anthropicBody := map[string]any{
			"model":      "claude-3-opus",
			"max_tokens": 1024,
			"system":     "You are a helpful assistant",
			"messages": []map[string]any{
				{
					"role":    "user",
					"content": "Hello",
				},
			},
		}

		bodyBytes, err := json.Marshal(anthropicBody)
		if err != nil {
			t.Fatalf("Failed to marshal body: %v", err)
		}

		// Use OpenAI Chat path - should detect as OpenAI despite Anthropic body
		result, err := detector.DetectFormat("/v1/chat/completions", bodyBytes)
		if err != nil {
			t.Fatalf("DetectFormat returned error: %v", err)
		}

		// Path should take precedence
		if result != inbound.InboundTypeOpenAIChat {
			t.Fatalf("DetectFormat with OpenAI path but Anthropic body = %v, expected InboundTypeOpenAIChat", result)
		}
	})
}

// TestDetectFormat_CaseInsensitivePath tests that path detection is case-insensitive
func TestDetectFormat_CaseInsensitivePath(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate mixed case paths
		pathVariants := []string{
			"/V1/Chat/Completions",
			"/V1/CHAT/COMPLETIONS",
			"/v1/Chat/completions",
			"/V1/Messages",
			"/v1/MESSAGES",
		}

		idx := rapid.IntRange(0, len(pathVariants)-1).Draw(t, "pathIndex")
		path := pathVariants[idx]

		result, err := detector.DetectFormat(path, nil)
		if err != nil {
			t.Fatalf("DetectFormat(%q, nil) returned error: %v", path, err)
		}

		// Should detect correctly regardless of case
		if idx < 3 {
			// Chat completions variants
			if result != inbound.InboundTypeOpenAIChat {
				t.Fatalf("DetectFormat(%q, nil) = %v, expected InboundTypeOpenAIChat", path, result)
			}
		} else {
			// Messages variants
			if result != inbound.InboundTypeAnthropic {
				t.Fatalf("DetectFormat(%q, nil) = %v, expected InboundTypeAnthropic", path, result)
			}
		}
	})
}

// TestDetectFormat_ToolUseContentType tests Anthropic detection via tool_use content type
func TestDetectFormat_ToolUseContentType(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		detector := NewFormatDetector()

		// Generate Anthropic-style request body with tool_use content
		toolName := rapid.StringMatching(`^[a-z_]+`).Draw(t, "toolName")
		toolInput := rapid.StringMatching(`^[a-zA-Z0-9]+`).Draw(t, "toolInput")

		anthropicBody := map[string]any{
			"model":      "claude-3-opus",
			"max_tokens": 1024,
			"messages": []map[string]any{
				{
					"role": "assistant",
					"content": []map[string]any{
						{
							"type":  "tool_use",
							"id":    "tool_123",
							"name":  toolName,
							"input": map[string]any{"query": toolInput},
						},
					},
				},
			},
		}

		bodyBytes, err := json.Marshal(anthropicBody)
		if err != nil {
			t.Fatalf("Failed to marshal body: %v", err)
		}

		result, err := detector.DetectFormat("/api/generate", bodyBytes)
		if err != nil {
			t.Fatalf("DetectFormat returned error: %v", err)
		}

		if result != inbound.InboundTypeAnthropic {
			t.Fatalf("DetectFormat with tool_use content = %v, expected InboundTypeAnthropic", result)
		}
	})
}

// TestDefaultDetector tests the package-level convenience function
func TestDefaultDetector(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that DefaultDetector is initialized and works
		if DefaultDetector == nil {
			t.Fatal("DefaultDetector is nil")
		}

		// Test the convenience function
		result, err := DetectFormat("/v1/chat/completions", nil)
		if err != nil {
			t.Fatalf("DetectFormat returned error: %v", err)
		}

		if result != inbound.InboundTypeOpenAIChat {
			t.Fatalf("DetectFormat = %v, expected InboundTypeOpenAIChat", result)
		}
	})
}
