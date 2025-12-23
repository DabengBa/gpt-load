package transformer

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"gpt-load/internal/transformer/inbound/anthropic"
	inboundOpenai "gpt-load/internal/transformer/inbound/openai"
	"gpt-load/internal/transformer/model"
	outboundAnthropic "gpt-load/internal/transformer/outbound/anthropic"
	outboundGemini "gpt-load/internal/transformer/outbound/gemini"

	"pgregory.net/rapid"
)

// Property 10: 消息内容往返一致性
// For any message content (text, images, multi-part content), cross-format conversion should:
// - Preserve text content exactly
// - Correctly convert image format (base64 data URL <-> source object)
// - Correctly handle system message placement differences
// - Correctly handle multi-part content order and types
// **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

// TestTextContent_OpenAIToAnthropic_RoundTrip tests that text content is preserved
// when converting from OpenAI to Anthropic format
func TestTextContent_OpenAIToAnthropic_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random text content
		textContent := rapid.StringN(1, 50, 500).Draw(t, "textContent")

		// Create an OpenAI request with text content
		req := &model.InternalLLMRequest{
			Model: "gpt-4",
			Messages: []model.Message{
				{
					Role: "user",
					Content: model.MessageContent{
						Content: &textContent,
					},
				},
			},
		}

		// Convert to Anthropic format using outbound transformer
		outbound := outboundAnthropic.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://api.anthropic.com", "sk-ant-test")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Parse the Anthropic request
		var anthropicReq map[string]interface{}
		if err := json.Unmarshal(body, &anthropicReq); err != nil {
			t.Fatalf("failed to unmarshal Anthropic request: %v", err)
		}

		// Verify messages are present
		messages, ok := anthropicReq["messages"].([]interface{})
		if !ok || len(messages) == 0 {
			t.Fatalf("messages should be present")
		}

		// Verify text content is preserved
		msg, ok := messages[0].(map[string]interface{})
		if !ok {
			t.Fatalf("message should be an object")
		}

		// Content can be string or array
		content := msg["content"]
		switch c := content.(type) {
		case string:
			if c != textContent {
				t.Fatalf("text content mismatch: expected %q, got %q", textContent, c)
			}
		case []interface{}:
			// Find text block
			found := false
			for _, block := range c {
				blockMap, ok := block.(map[string]interface{})
				if !ok {
					continue
				}
				if blockMap["type"] == "text" {
					text, _ := blockMap["text"].(string)
					if text == textContent {
						found = true
						break
					}
				}
			}
			if !found {
				t.Fatalf("text content not found in content blocks")
			}
		default:
			t.Fatalf("unexpected content type: %T", content)
		}
	})
}


// TestTextContent_OpenAIToGemini_RoundTrip tests that text content is preserved
// when converting from OpenAI to Gemini format
func TestTextContent_OpenAIToGemini_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random text content
		textContent := rapid.StringN(1, 50, 500).Draw(t, "textContent")

		// Create an OpenAI request with text content
		req := &model.InternalLLMRequest{
			Model: "gemini-pro",
			Messages: []model.Message{
				{
					Role: "user",
					Content: model.MessageContent{
						Content: &textContent,
					},
				},
			},
		}

		// Convert to Gemini format using outbound transformer
		outbound := outboundGemini.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://generativelanguage.googleapis.com", "AIzaSyTest")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Parse the Gemini request
		var geminiReq map[string]interface{}
		if err := json.Unmarshal(body, &geminiReq); err != nil {
			t.Fatalf("failed to unmarshal Gemini request: %v", err)
		}

		// Verify contents are present
		contents, ok := geminiReq["contents"].([]interface{})
		if !ok || len(contents) == 0 {
			t.Fatalf("contents should be present")
		}

		// Verify text content is preserved
		content, ok := contents[0].(map[string]interface{})
		if !ok {
			t.Fatalf("content should be an object")
		}

		parts, ok := content["parts"].([]interface{})
		if !ok || len(parts) == 0 {
			t.Fatalf("parts should be present")
		}

		// Find text part
		found := false
		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			text, ok := partMap["text"].(string)
			if ok && text == textContent {
				found = true
				break
			}
		}

		if !found {
			t.Fatalf("text content not found in parts")
		}
	})
}

// TestTextContent_AnthropicToOpenAI_RoundTrip tests that text content is preserved
// when converting from Anthropic to OpenAI format
func TestTextContent_AnthropicToOpenAI_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random text content
		textContent := rapid.StringN(1, 50, 500).Draw(t, "textContent")

		// Create an Anthropic request
		anthropicReq := map[string]interface{}{
			"model":      "claude-3-opus-20240229",
			"max_tokens": 1024,
			"messages": []map[string]interface{}{
				{
					"role":    "user",
					"content": textContent,
				},
			},
		}

		// Marshal to JSON
		body, err := json.Marshal(anthropicReq)
		if err != nil {
			t.Fatalf("failed to marshal Anthropic request: %v", err)
		}

		// Convert to internal format using inbound transformer
		inbound := anthropic.NewMessagesInbound()
		ctx := context.Background()

		internalReq, err := inbound.TransformRequest(ctx, body)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify text content is preserved
		if len(internalReq.Messages) == 0 {
			t.Fatalf("messages should be present")
		}

		// Find user message
		found := false
		for _, msg := range internalReq.Messages {
			if msg.Role == "user" {
				actualText := msg.Content.GetText()
				if actualText == textContent {
					found = true
					break
				}
			}
		}

		if !found {
			t.Fatalf("text content not preserved in internal format")
		}
	})
}


// TestImageContent_OpenAIToAnthropic_RoundTrip tests that image content is correctly
// converted from OpenAI image_url format to Anthropic source format
func TestImageContent_OpenAIToAnthropic_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random image data (small for testing)
		imageData := rapid.SliceOfN(rapid.Byte(), 10, 100).Draw(t, "imageData")
		base64Data := base64.StdEncoding.EncodeToString(imageData)

		// Generate media type
		mediaType := rapid.SampledFrom([]string{
			"image/png",
			"image/jpeg",
			"image/gif",
			"image/webp",
		}).Draw(t, "mediaType")

		// Create data URL
		dataURL := "data:" + mediaType + ";base64," + base64Data

		// Create an OpenAI request with image content
		req := &model.InternalLLMRequest{
			Model: "gpt-4-vision-preview",
			Messages: []model.Message{
				{
					Role: "user",
					Content: model.MessageContent{
						MultipleContent: []model.MessageContentPart{
							{
								Type: "text",
								Text: stringPtr("What's in this image?"),
							},
							{
								Type: "image_url",
								ImageURL: &model.ImageURL{
									URL: dataURL,
								},
							},
						},
					},
				},
			},
		}

		// Convert to Anthropic format using outbound transformer
		outbound := outboundAnthropic.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://api.anthropic.com", "sk-ant-test")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Parse the Anthropic request
		var anthropicReq map[string]interface{}
		if err := json.Unmarshal(body, &anthropicReq); err != nil {
			t.Fatalf("failed to unmarshal Anthropic request: %v", err)
		}

		// Verify messages are present
		messages, ok := anthropicReq["messages"].([]interface{})
		if !ok || len(messages) == 0 {
			t.Fatalf("messages should be present")
		}

		// Verify content blocks
		msg, ok := messages[0].(map[string]interface{})
		if !ok {
			t.Fatalf("message should be an object")
		}

		content, ok := msg["content"].([]interface{})
		if !ok {
			t.Fatalf("content should be an array for multi-part content")
		}

		// Find image block
		foundImage := false
		for _, block := range content {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			if blockMap["type"] == "image" {
				source, ok := blockMap["source"].(map[string]interface{})
				if !ok {
					t.Fatalf("image block should have source")
				}

				// Verify source properties
				if source["type"] != "base64" {
					t.Fatalf("source type should be 'base64', got %v", source["type"])
				}
				if source["media_type"] != mediaType {
					t.Fatalf("media_type mismatch: expected %s, got %v", mediaType, source["media_type"])
				}
				if source["data"] != base64Data {
					t.Fatalf("base64 data mismatch")
				}

				foundImage = true
				break
			}
		}

		if !foundImage {
			t.Fatalf("image block not found in Anthropic request")
		}
	})
}

// TestImageContent_OpenAIToGemini_RoundTrip tests that image content is correctly
// converted from OpenAI image_url format to Gemini inlineData format
func TestImageContent_OpenAIToGemini_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random image data (small for testing)
		imageData := rapid.SliceOfN(rapid.Byte(), 10, 100).Draw(t, "imageData")
		base64Data := base64.StdEncoding.EncodeToString(imageData)

		// Generate media type
		mediaType := rapid.SampledFrom([]string{
			"image/png",
			"image/jpeg",
			"image/gif",
			"image/webp",
		}).Draw(t, "mediaType")

		// Create data URL
		dataURL := "data:" + mediaType + ";base64," + base64Data

		// Create an OpenAI request with image content
		req := &model.InternalLLMRequest{
			Model: "gemini-pro-vision",
			Messages: []model.Message{
				{
					Role: "user",
					Content: model.MessageContent{
						MultipleContent: []model.MessageContentPart{
							{
								Type: "text",
								Text: stringPtr("What's in this image?"),
							},
							{
								Type: "image_url",
								ImageURL: &model.ImageURL{
									URL: dataURL,
								},
							},
						},
					},
				},
			},
		}

		// Convert to Gemini format using outbound transformer
		outbound := outboundGemini.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://generativelanguage.googleapis.com", "AIzaSyTest")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Parse the Gemini request
		var geminiReq map[string]interface{}
		if err := json.Unmarshal(body, &geminiReq); err != nil {
			t.Fatalf("failed to unmarshal Gemini request: %v", err)
		}

		// Verify contents are present
		contents, ok := geminiReq["contents"].([]interface{})
		if !ok || len(contents) == 0 {
			t.Fatalf("contents should be present")
		}

		// Verify parts
		content, ok := contents[0].(map[string]interface{})
		if !ok {
			t.Fatalf("content should be an object")
		}

		parts, ok := content["parts"].([]interface{})
		if !ok {
			t.Fatalf("parts should be present")
		}

		// Find inlineData part
		foundImage := false
		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			inlineData, ok := partMap["inlineData"].(map[string]interface{})
			if ok {
				// Verify inlineData properties
				if inlineData["mimeType"] != mediaType {
					t.Fatalf("mimeType mismatch: expected %s, got %v", mediaType, inlineData["mimeType"])
				}
				if inlineData["data"] != base64Data {
					t.Fatalf("base64 data mismatch")
				}

				foundImage = true
				break
			}
		}

		if !foundImage {
			t.Fatalf("inlineData part not found in Gemini request")
		}
	})
}


// TestImageContent_AnthropicToOpenAI_RoundTrip tests that Anthropic image source
// is correctly converted to OpenAI image_url format
func TestImageContent_AnthropicToOpenAI_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random image data (small for testing)
		imageData := rapid.SliceOfN(rapid.Byte(), 10, 100).Draw(t, "imageData")
		base64Data := base64.StdEncoding.EncodeToString(imageData)

		// Generate media type
		mediaType := rapid.SampledFrom([]string{
			"image/png",
			"image/jpeg",
			"image/gif",
			"image/webp",
		}).Draw(t, "mediaType")

		// Create an Anthropic request with image content
		anthropicReq := map[string]interface{}{
			"model":      "claude-3-opus-20240229",
			"max_tokens": 1024,
			"messages": []map[string]interface{}{
				{
					"role": "user",
					"content": []map[string]interface{}{
						{
							"type": "text",
							"text": "What's in this image?",
						},
						{
							"type": "image",
							"source": map[string]interface{}{
								"type":       "base64",
								"media_type": mediaType,
								"data":       base64Data,
							},
						},
					},
				},
			},
		}

		// Marshal to JSON
		body, err := json.Marshal(anthropicReq)
		if err != nil {
			t.Fatalf("failed to marshal Anthropic request: %v", err)
		}

		// Convert to internal format using inbound transformer
		inbound := anthropic.NewMessagesInbound()
		ctx := context.Background()

		internalReq, err := inbound.TransformRequest(ctx, body)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify messages are present
		if len(internalReq.Messages) == 0 {
			t.Fatalf("messages should be present")
		}

		// Find user message with image
		found := false
		for _, msg := range internalReq.Messages {
			if msg.Role == "user" && len(msg.Content.MultipleContent) > 0 {
				for _, part := range msg.Content.MultipleContent {
					if part.Type == "image_url" && part.ImageURL != nil {
						// Verify data URL format
						expectedDataURL := "data:" + mediaType + ";base64," + base64Data
						if part.ImageURL.URL == expectedDataURL {
							found = true
							break
						}
					}
				}
			}
		}

		if !found {
			t.Fatalf("image content not correctly converted to OpenAI format")
		}
	})
}

// TestSystemMessage_OpenAIToAnthropic_RoundTrip tests that system messages are correctly
// converted from OpenAI messages array to Anthropic separate system field
func TestSystemMessage_OpenAIToAnthropic_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random system message
		systemContent := rapid.StringN(1, 50, 500).Draw(t, "systemContent")
		userContent := rapid.StringN(1, 50, 500).Draw(t, "userContent")

		// Create an OpenAI request with system message
		req := &model.InternalLLMRequest{
			Model: "gpt-4",
			Messages: []model.Message{
				{
					Role: "system",
					Content: model.MessageContent{
						Content: &systemContent,
					},
				},
				{
					Role: "user",
					Content: model.MessageContent{
						Content: &userContent,
					},
				},
			},
		}

		// Convert to Anthropic format using outbound transformer
		outbound := outboundAnthropic.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://api.anthropic.com", "sk-ant-test")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Parse the Anthropic request
		var anthropicReq map[string]interface{}
		if err := json.Unmarshal(body, &anthropicReq); err != nil {
			t.Fatalf("failed to unmarshal Anthropic request: %v", err)
		}

		// Verify system field is present and contains the system content
		system := anthropicReq["system"]
		if system == nil {
			t.Fatalf("system field should be present")
		}

		// System can be string or array
		switch s := system.(type) {
		case string:
			if s != systemContent {
				t.Fatalf("system content mismatch: expected %q, got %q", systemContent, s)
			}
		default:
			t.Fatalf("unexpected system type: %T", system)
		}

		// Verify messages don't contain system message
		messages, ok := anthropicReq["messages"].([]interface{})
		if !ok {
			t.Fatalf("messages should be present")
		}

		for _, msg := range messages {
			msgMap, ok := msg.(map[string]interface{})
			if !ok {
				continue
			}
			if msgMap["role"] == "system" {
				t.Fatalf("system message should not be in messages array")
			}
		}
	})
}

// TestSystemMessage_OpenAIToGemini_RoundTrip tests that system messages are correctly
// converted from OpenAI messages array to Gemini systemInstruction field
func TestSystemMessage_OpenAIToGemini_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random system message
		systemContent := rapid.StringN(1, 50, 500).Draw(t, "systemContent")
		userContent := rapid.StringN(1, 50, 500).Draw(t, "userContent")

		// Create an OpenAI request with system message
		req := &model.InternalLLMRequest{
			Model: "gemini-pro",
			Messages: []model.Message{
				{
					Role: "system",
					Content: model.MessageContent{
						Content: &systemContent,
					},
				},
				{
					Role: "user",
					Content: model.MessageContent{
						Content: &userContent,
					},
				},
			},
		}

		// Convert to Gemini format using outbound transformer
		outbound := outboundGemini.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://generativelanguage.googleapis.com", "AIzaSyTest")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Parse the Gemini request
		var geminiReq map[string]interface{}
		if err := json.Unmarshal(body, &geminiReq); err != nil {
			t.Fatalf("failed to unmarshal Gemini request: %v", err)
		}

		// Verify systemInstruction field is present
		systemInstruction, ok := geminiReq["systemInstruction"].(map[string]interface{})
		if !ok {
			t.Fatalf("systemInstruction field should be present")
		}

		// Verify parts contain the system content
		parts, ok := systemInstruction["parts"].([]interface{})
		if !ok || len(parts) == 0 {
			t.Fatalf("systemInstruction should have parts")
		}

		found := false
		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			text, ok := partMap["text"].(string)
			if ok && text == systemContent {
				found = true
				break
			}
		}

		if !found {
			t.Fatalf("system content not found in systemInstruction")
		}

		// Verify contents don't contain system message
		contents, ok := geminiReq["contents"].([]interface{})
		if !ok {
			t.Fatalf("contents should be present")
		}

		for _, content := range contents {
			contentMap, ok := content.(map[string]interface{})
			if !ok {
				continue
			}
			if contentMap["role"] == "system" {
				t.Fatalf("system message should not be in contents array")
			}
		}
	})
}


// TestSystemMessage_AnthropicToOpenAI_RoundTrip tests that Anthropic system field
// is correctly converted to OpenAI messages array
func TestSystemMessage_AnthropicToOpenAI_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random system message
		systemContent := rapid.StringN(1, 50, 500).Draw(t, "systemContent")
		userContent := rapid.StringN(1, 50, 500).Draw(t, "userContent")

		// Create an Anthropic request with system field
		anthropicReq := map[string]interface{}{
			"model":      "claude-3-opus-20240229",
			"max_tokens": 1024,
			"system":     systemContent,
			"messages": []map[string]interface{}{
				{
					"role":    "user",
					"content": userContent,
				},
			},
		}

		// Marshal to JSON
		body, err := json.Marshal(anthropicReq)
		if err != nil {
			t.Fatalf("failed to marshal Anthropic request: %v", err)
		}

		// Convert to internal format using inbound transformer
		inbound := anthropic.NewMessagesInbound()
		ctx := context.Background()

		internalReq, err := inbound.TransformRequest(ctx, body)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify system message is present in messages array
		foundSystem := false
		for _, msg := range internalReq.Messages {
			if msg.Role == "system" {
				actualText := msg.Content.GetText()
				if actualText == systemContent {
					foundSystem = true
					break
				}
			}
		}

		if !foundSystem {
			t.Fatalf("system message not found in internal format messages")
		}
	})
}

// TestMultiPartContent_OpenAIToAnthropic_RoundTrip tests that multi-part content
// (text + images) is correctly converted and order is preserved
func TestMultiPartContent_OpenAIToAnthropic_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random content parts
		numTextParts := rapid.IntRange(1, 3).Draw(t, "numTextParts")
		textParts := make([]string, numTextParts)
		for i := 0; i < numTextParts; i++ {
			textParts[i] = rapid.StringN(1, 50, 200).Draw(t, "textPart")
		}

		// Build multi-part content
		multipleContent := make([]model.MessageContentPart, 0)
		for _, text := range textParts {
			textCopy := text
			multipleContent = append(multipleContent, model.MessageContentPart{
				Type: "text",
				Text: &textCopy,
			})
		}

		// Create an OpenAI request with multi-part content
		req := &model.InternalLLMRequest{
			Model: "gpt-4",
			Messages: []model.Message{
				{
					Role: "user",
					Content: model.MessageContent{
						MultipleContent: multipleContent,
					},
				},
			},
		}

		// Convert to Anthropic format using outbound transformer
		outbound := outboundAnthropic.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://api.anthropic.com", "sk-ant-test")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the request body
		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Parse the Anthropic request
		var anthropicReq map[string]interface{}
		if err := json.Unmarshal(body, &anthropicReq); err != nil {
			t.Fatalf("failed to unmarshal Anthropic request: %v", err)
		}

		// Verify messages are present
		messages, ok := anthropicReq["messages"].([]interface{})
		if !ok || len(messages) == 0 {
			t.Fatalf("messages should be present")
		}

		// Verify content blocks
		msg, ok := messages[0].(map[string]interface{})
		if !ok {
			t.Fatalf("message should be an object")
		}

		content, ok := msg["content"].([]interface{})
		if !ok {
			t.Fatalf("content should be an array for multi-part content")
		}

		// Verify order and content of text blocks
		textIdx := 0
		for _, block := range content {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			if blockMap["type"] == "text" {
				text, _ := blockMap["text"].(string)
				if textIdx < len(textParts) {
					if text != textParts[textIdx] {
						t.Fatalf("text content mismatch at index %d: expected %q, got %q", textIdx, textParts[textIdx], text)
					}
					textIdx++
				}
			}
		}

		if textIdx != len(textParts) {
			t.Fatalf("not all text parts were converted: expected %d, got %d", len(textParts), textIdx)
		}
	})
}

// TestMultiPartContent_AnthropicToOpenAI_RoundTrip tests that Anthropic multi-part content
// is correctly converted to OpenAI format with order preserved
func TestMultiPartContent_AnthropicToOpenAI_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random content parts
		numTextParts := rapid.IntRange(1, 3).Draw(t, "numTextParts")
		textParts := make([]string, numTextParts)
		for i := 0; i < numTextParts; i++ {
			textParts[i] = rapid.StringN(1, 50, 200).Draw(t, "textPart")
		}

		// Build Anthropic content blocks
		contentBlocks := make([]map[string]interface{}, 0)
		for _, text := range textParts {
			contentBlocks = append(contentBlocks, map[string]interface{}{
				"type": "text",
				"text": text,
			})
		}

		// Create an Anthropic request with multi-part content
		anthropicReq := map[string]interface{}{
			"model":      "claude-3-opus-20240229",
			"max_tokens": 1024,
			"messages": []map[string]interface{}{
				{
					"role":    "user",
					"content": contentBlocks,
				},
			},
		}

		// Marshal to JSON
		body, err := json.Marshal(anthropicReq)
		if err != nil {
			t.Fatalf("failed to marshal Anthropic request: %v", err)
		}

		// Convert to internal format using inbound transformer
		inbound := anthropic.NewMessagesInbound()
		ctx := context.Background()

		internalReq, err := inbound.TransformRequest(ctx, body)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify messages are present
		if len(internalReq.Messages) == 0 {
			t.Fatalf("messages should be present")
		}

		// Find user message
		var userMsg *model.Message
		for i := range internalReq.Messages {
			if internalReq.Messages[i].Role == "user" {
				userMsg = &internalReq.Messages[i]
				break
			}
		}

		if userMsg == nil {
			t.Fatalf("user message not found")
		}

		// Verify content parts
		if len(userMsg.Content.MultipleContent) > 0 {
			// Multi-part content
			textIdx := 0
			for _, part := range userMsg.Content.MultipleContent {
				if part.Type == "text" && part.Text != nil {
					if textIdx < len(textParts) {
						if *part.Text != textParts[textIdx] {
							t.Fatalf("text content mismatch at index %d: expected %q, got %q", textIdx, textParts[textIdx], *part.Text)
						}
						textIdx++
					}
				}
			}
			if textIdx != len(textParts) {
				t.Fatalf("not all text parts were converted: expected %d, got %d", len(textParts), textIdx)
			}
		} else if userMsg.Content.Content != nil {
			// Single text content (when there's only one text part)
			if len(textParts) == 1 {
				if *userMsg.Content.Content != textParts[0] {
					t.Fatalf("text content mismatch: expected %q, got %q", textParts[0], *userMsg.Content.Content)
				}
			} else {
				// Multiple parts should result in concatenated text
				expectedText := strings.Join(textParts, "")
				if *userMsg.Content.Content != expectedText {
					t.Fatalf("concatenated text content mismatch: expected %q, got %q", expectedText, *userMsg.Content.Content)
				}
			}
		} else {
			t.Fatalf("message content is empty")
		}
	})
}


// TestFullRoundTrip_OpenAI_Anthropic_OpenAI tests the full round-trip:
// OpenAI request -> Anthropic request -> Anthropic response -> OpenAI response
func TestFullRoundTrip_OpenAI_Anthropic_OpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random text content
		textContent := rapid.StringN(1, 50, 500).Draw(t, "textContent")
		responseContent := rapid.StringN(1, 50, 500).Draw(t, "responseContent")

		// Create an OpenAI request
		req := &model.InternalLLMRequest{
			Model: "gpt-4",
			Messages: []model.Message{
				{
					Role: "user",
					Content: model.MessageContent{
						Content: &textContent,
					},
				},
			},
		}

		// Step 1: Convert OpenAI request to Anthropic request
		outbound := outboundAnthropic.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://api.anthropic.com", "sk-ant-test")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the Anthropic request body
		anthropicReqBody, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Verify the request was converted
		var anthropicReq map[string]interface{}
		if err := json.Unmarshal(anthropicReqBody, &anthropicReq); err != nil {
			t.Fatalf("failed to unmarshal Anthropic request: %v", err)
		}

		// Step 2: Simulate Anthropic response
		anthropicResp := map[string]interface{}{
			"id":    "msg_test",
			"type":  "message",
			"role":  "assistant",
			"model": "claude-3-opus-20240229",
			"content": []map[string]interface{}{
				{
					"type": "text",
					"text": responseContent,
				},
			},
			"stop_reason": "end_turn",
			"usage": map[string]interface{}{
				"input_tokens":  100,
				"output_tokens": 50,
			},
		}
		anthropicRespBody, _ := json.Marshal(anthropicResp)

		// Step 3: Convert Anthropic response to internal format
		mockResp := createMockHTTPResponseForContent(200, anthropicRespBody)
		defer mockResp.Body.Close()

		internalResp, err := outbound.TransformResponse(ctx, mockResp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Step 4: Convert internal response to OpenAI format
		openaiInbound := inboundOpenai.NewChatInbound()
		openaiRespBody, err := openaiInbound.TransformResponse(ctx, internalResp)
		if err != nil {
			t.Fatalf("TransformResponse to OpenAI failed: %v", err)
		}

		// Verify the OpenAI response
		var openaiResp map[string]interface{}
		if err := json.Unmarshal(openaiRespBody, &openaiResp); err != nil {
			t.Fatalf("failed to unmarshal OpenAI response: %v", err)
		}

		// Verify content is preserved
		choices, ok := openaiResp["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			t.Fatalf("OpenAI response should have choices")
		}

		choice, ok := choices[0].(map[string]interface{})
		if !ok {
			t.Fatalf("choice should be an object")
		}

		message, ok := choice["message"].(map[string]interface{})
		if !ok {
			t.Fatalf("message should be an object")
		}

		content, ok := message["content"].(string)
		if !ok {
			t.Fatalf("content should be a string")
		}

		if content != responseContent {
			t.Fatalf("response content mismatch: expected %q, got %q", responseContent, content)
		}
	})
}

// TestFullRoundTrip_OpenAI_Gemini_OpenAI tests the full round-trip:
// OpenAI request -> Gemini request -> Gemini response -> OpenAI response
func TestFullRoundTrip_OpenAI_Gemini_OpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random text content
		textContent := rapid.StringN(1, 50, 500).Draw(t, "textContent")
		responseContent := rapid.StringN(1, 50, 500).Draw(t, "responseContent")

		// Create an OpenAI request
		req := &model.InternalLLMRequest{
			Model: "gemini-pro",
			Messages: []model.Message{
				{
					Role: "user",
					Content: model.MessageContent{
						Content: &textContent,
					},
				},
			},
		}

		// Step 1: Convert OpenAI request to Gemini request
		outbound := outboundGemini.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, req, "https://generativelanguage.googleapis.com", "AIzaSyTest")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the Gemini request body
		geminiReqBody, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Verify the request was converted
		var geminiReq map[string]interface{}
		if err := json.Unmarshal(geminiReqBody, &geminiReq); err != nil {
			t.Fatalf("failed to unmarshal Gemini request: %v", err)
		}

		// Step 2: Simulate Gemini response
		geminiResp := map[string]interface{}{
			"candidates": []map[string]interface{}{
				{
					"content": map[string]interface{}{
						"role": "model",
						"parts": []map[string]interface{}{
							{
								"text": responseContent,
							},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]interface{}{
				"promptTokenCount":     100,
				"candidatesTokenCount": 50,
				"totalTokenCount":      150,
			},
		}
		geminiRespBody, _ := json.Marshal(geminiResp)

		// Step 3: Convert Gemini response to internal format
		mockResp := createMockHTTPResponseForContent(200, geminiRespBody)
		defer mockResp.Body.Close()

		internalResp, err := outbound.TransformResponse(ctx, mockResp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Step 4: Convert internal response to OpenAI format
		openaiInbound := inboundOpenai.NewChatInbound()
		openaiRespBody, err := openaiInbound.TransformResponse(ctx, internalResp)
		if err != nil {
			t.Fatalf("TransformResponse to OpenAI failed: %v", err)
		}

		// Verify the OpenAI response
		var openaiResp map[string]interface{}
		if err := json.Unmarshal(openaiRespBody, &openaiResp); err != nil {
			t.Fatalf("failed to unmarshal OpenAI response: %v", err)
		}

		// Verify content is preserved
		choices, ok := openaiResp["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			t.Fatalf("OpenAI response should have choices")
		}

		choice, ok := choices[0].(map[string]interface{})
		if !ok {
			t.Fatalf("choice should be an object")
		}

		message, ok := choice["message"].(map[string]interface{})
		if !ok {
			t.Fatalf("message should be an object")
		}

		content, ok := message["content"].(string)
		if !ok {
			t.Fatalf("content should be a string")
		}

		if content != responseContent {
			t.Fatalf("response content mismatch: expected %q, got %q", responseContent, content)
		}
	})
}

// Helper functions

func stringPtr(s string) *string {
	return &s
}

// createMockHTTPResponseForContent creates a mock HTTP response for testing
func createMockHTTPResponseForContent(statusCode int, body []byte) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     make(http.Header),
	}
}
