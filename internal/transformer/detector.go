package transformer

import (
	"encoding/json"
	"strings"

	"gpt-load/internal/transformer/inbound"
)

// FormatDetector detects the API format of client requests
type FormatDetector struct{}

// NewFormatDetector creates a new FormatDetector instance
func NewFormatDetector() *FormatDetector {
	return &FormatDetector{}
}

// DetectFormat detects the API format based on request path and body content.
// It first tries to detect the format from the request path, then falls back
// to analyzing the request body structure.
// Returns InboundTypeOpenAIChat as the default if no specific format is detected.
func (d *FormatDetector) DetectFormat(path string, body []byte) (inbound.InboundType, error) {
	// 1. Detect by path
	if formatType, detected := d.detectByPath(path); detected {
		return formatType, nil
	}

	// 2. Detect by request body features
	if formatType, detected := d.detectByBody(body); detected {
		return formatType, nil
	}

	// Default to OpenAI Chat format
	return inbound.InboundTypeOpenAIChat, nil
}

// detectByPath attempts to detect the API format from the request path
func (d *FormatDetector) detectByPath(path string) (inbound.InboundType, bool) {
	// Normalize path for comparison
	normalizedPath := strings.ToLower(path)

	// OpenAI Chat Completions endpoint
	if strings.Contains(normalizedPath, "/v1/chat/completions") ||
		strings.Contains(normalizedPath, "/chat/completions") {
		return inbound.InboundTypeOpenAIChat, true
	}

	// OpenAI Response endpoint (for future support)
	if strings.Contains(normalizedPath, "/v1/responses") ||
		strings.Contains(normalizedPath, "/responses") {
		return inbound.InboundTypeOpenAIResponse, true
	}

	// Anthropic Messages endpoint
	if strings.Contains(normalizedPath, "/v1/messages") ||
		strings.Contains(normalizedPath, "/messages") {
		return inbound.InboundTypeAnthropic, true
	}

	return inbound.InboundTypeOpenAIChat, false
}

// detectByBody attempts to detect the API format from the request body structure
func (d *FormatDetector) detectByBody(body []byte) (inbound.InboundType, bool) {
	if len(body) == 0 {
		return inbound.InboundTypeOpenAIChat, false
	}

	var probe map[string]any
	if err := json.Unmarshal(body, &probe); err != nil {
		return inbound.InboundTypeOpenAIChat, false
	}

	// Check for Anthropic-specific features
	if d.isAnthropicFormat(probe) {
		return inbound.InboundTypeAnthropic, true
	}

	// Check for OpenAI Response format features (for future support)
	if d.isOpenAIResponseFormat(probe) {
		return inbound.InboundTypeOpenAIResponse, true
	}

	return inbound.InboundTypeOpenAIChat, false
}

// isAnthropicFormat checks if the request body has Anthropic-specific features
// Anthropic format characteristics:
// - Has "max_tokens" field (required in Anthropic, optional in OpenAI)
// - Messages have content as array of content blocks (not just string)
// - May have separate "system" field at top level
func (d *FormatDetector) isAnthropicFormat(probe map[string]any) bool {
	// Check for Anthropic-specific "system" field at top level
	if _, hasSystem := probe["system"]; hasSystem {
		return true
	}

	// Check for Anthropic message content structure
	// Anthropic messages have content as array of content blocks
	messages, ok := probe["messages"].([]any)
	if !ok || len(messages) == 0 {
		return false
	}

	// Check first message for Anthropic content structure
	firstMsg, ok := messages[0].(map[string]any)
	if !ok {
		return false
	}

	// Anthropic content is typically an array of content blocks
	content, ok := firstMsg["content"].([]any)
	if !ok || len(content) == 0 {
		return false
	}

	// Check if content blocks have Anthropic-style structure (type + text/source)
	firstContent, ok := content[0].(map[string]any)
	if !ok {
		return false
	}

	// Anthropic content blocks have "type" field with values like "text", "image", "tool_use", etc.
	if contentType, hasType := firstContent["type"].(string); hasType {
		// Anthropic-specific content types
		switch contentType {
		case "text", "image", "tool_use", "tool_result":
			return true
		}
	}

	return false
}

// isOpenAIResponseFormat checks if the request body has OpenAI Response format features
// This is for future support of the OpenAI Response API
func (d *FormatDetector) isOpenAIResponseFormat(probe map[string]any) bool {
	// OpenAI Response format has specific fields like "input" instead of "messages"
	if _, hasInput := probe["input"]; hasInput {
		return true
	}

	// Check for "modalities" field which is specific to Response API
	if _, hasModalities := probe["modalities"]; hasModalities {
		return true
	}

	return false
}

// DetectFormatFunc is a function type for format detection
// This allows for easy mocking in tests
type DetectFormatFunc func(path string, body []byte) (inbound.InboundType, error)

// DefaultDetector is the default format detector instance
var DefaultDetector = NewFormatDetector()

// DetectFormat is a convenience function that uses the default detector
func DetectFormat(path string, body []byte) (inbound.InboundType, error) {
	return DefaultDetector.DetectFormat(path, body)
}
