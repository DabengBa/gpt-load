package proxy

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"gpt-load/internal/transformer/inbound"
	"gpt-load/internal/transformer/inbound/openai"
	"gpt-load/internal/transformer/model"
	"gpt-load/internal/transformer/outbound/anthropic"
	"gpt-load/internal/transformer/outbound/gemini"

	"pgregory.net/rapid"
)

// Property 6: 跨格式流式转换
// For any upstream streaming response (Anthropic SSE or Gemini streaming),
// when the client expects OpenAI format, each streaming chunk should be
// correctly converted to OpenAI SSE format, including:
// - Correct "data: " prefix
// - Correct JSON structure
// - Correct [DONE] marker handling
// **Validates: Requirements 7.1, 7.2, 7.4**

func TestCrossFormatStreamTransform_AnthropicToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random Anthropic stream chunk
		chunk := generateAnthropicStreamChunk(t)

		// Create outbound transformer (Anthropic)
		outboundTransformer := anthropic.NewMessagesOutbound()

		// Create inbound transformer (OpenAI)
		inboundTransformer := openai.NewChatInbound()

		ctx := context.Background()

		// Transform Anthropic stream data to internal format
		internalChunk, err := outboundTransformer.TransformStream(ctx, chunk)
		if err != nil {
			// Some chunks may not produce output (e.g., ping events)
			return
		}
		if internalChunk == nil {
			return
		}

		// Transform internal chunk to OpenAI format
		clientChunk, err := inboundTransformer.TransformStream(ctx, internalChunk)
		if err != nil {
			t.Fatalf("TransformStream failed: %v", err)
		}
		if clientChunk == nil {
			return
		}

		// Verify the output has correct SSE format
		verifyOpenAISSEFormat(t, clientChunk)
	})
}

func TestCrossFormatStreamTransform_GeminiToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random Gemini stream chunk
		chunk := generateGeminiStreamChunk(t)

		// Create outbound transformer (Gemini)
		outboundTransformer := gemini.NewMessagesOutbound()

		// Create inbound transformer (OpenAI)
		inboundTransformer := openai.NewChatInbound()

		ctx := context.Background()

		// Transform Gemini stream data to internal format
		internalChunk, err := outboundTransformer.TransformStream(ctx, chunk)
		if err != nil {
			return
		}
		if internalChunk == nil {
			return
		}

		// Transform internal chunk to OpenAI format
		clientChunk, err := inboundTransformer.TransformStream(ctx, internalChunk)
		if err != nil {
			t.Fatalf("TransformStream failed: %v", err)
		}
		if clientChunk == nil {
			return
		}

		// Verify the output has correct SSE format
		verifyOpenAISSEFormat(t, clientChunk)
	})
}


// generateAnthropicStreamChunk generates a random Anthropic SSE stream chunk
func generateAnthropicStreamChunk(t *rapid.T) []byte {
	eventTypes := []string{"message_start", "content_block_start", "content_block_delta", "message_delta"}
	eventType := eventTypes[rapid.IntRange(0, len(eventTypes)-1).Draw(t, "eventType")]

	var data map[string]any

	switch eventType {
	case "message_start":
		data = map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":    "msg_" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "msgId"),
				"type":  "message",
				"role":  "assistant",
				"model": "claude-3-sonnet-20240229",
			},
		}
	case "content_block_start":
		data = map[string]any{
			"type":  "content_block_start",
			"index": rapid.IntRange(0, 3).Draw(t, "index"),
			"content_block": map[string]any{
				"type": "text",
				"text": "",
			},
		}
	case "content_block_delta":
		text := rapid.StringN(1, 10, 50).Draw(t, "deltaText")
		data = map[string]any{
			"type":  "content_block_delta",
			"index": rapid.IntRange(0, 3).Draw(t, "index"),
			"delta": map[string]any{
				"type": "text_delta",
				"text": text,
			},
		}
	case "message_delta":
		data = map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason": "end_turn",
			},
			"usage": map[string]any{
				"output_tokens": rapid.IntRange(1, 1000).Draw(t, "outputTokens"),
			},
		}
	}

	jsonData, _ := json.Marshal(data)
	return []byte("event: " + eventType + "\ndata: " + string(jsonData))
}

// generateGeminiStreamChunk generates a random Gemini stream chunk
func generateGeminiStreamChunk(t *rapid.T) []byte {
	text := rapid.StringN(1, 10, 100).Draw(t, "text")

	data := map[string]any{
		"candidates": []map[string]any{
			{
				"content": map[string]any{
					"parts": []map[string]any{
						{"text": text},
					},
					"role": "model",
				},
			},
		},
	}

	// Optionally add finish reason
	if rapid.Bool().Draw(t, "hasFinishReason") {
		finishReasons := []string{"STOP", "MAX_TOKENS", "SAFETY"}
		data["candidates"].([]map[string]any)[0]["finishReason"] = finishReasons[rapid.IntRange(0, len(finishReasons)-1).Draw(t, "finishReason")]
	}

	// Optionally add usage metadata
	if rapid.Bool().Draw(t, "hasUsage") {
		data["usageMetadata"] = map[string]any{
			"promptTokenCount":     rapid.IntRange(1, 1000).Draw(t, "promptTokens"),
			"candidatesTokenCount": rapid.IntRange(1, 1000).Draw(t, "candidatesTokens"),
			"totalTokenCount":      rapid.IntRange(2, 2000).Draw(t, "totalTokens"),
		}
	}

	jsonData, _ := json.Marshal(data)
	return jsonData
}

// verifyOpenAISSEFormat verifies that the output has correct OpenAI SSE format
func verifyOpenAISSEFormat(t *rapid.T, chunk []byte) {
	chunkStr := string(chunk)

	// Verify "data: " prefix
	if !strings.HasPrefix(chunkStr, "data: ") {
		t.Fatalf("chunk should start with 'data: ', got: %s", chunkStr[:min(50, len(chunkStr))])
	}

	// Verify ends with double newline
	if !strings.HasSuffix(chunkStr, "\n\n") {
		t.Fatalf("chunk should end with '\\n\\n', got: %s", chunkStr[max(0, len(chunkStr)-10):])
	}

	// Extract JSON part
	jsonPart := strings.TrimPrefix(chunkStr, "data: ")
	jsonPart = strings.TrimSuffix(jsonPart, "\n\n")

	// Verify it's valid JSON
	var parsed map[string]any
	if err := json.Unmarshal([]byte(jsonPart), &parsed); err != nil {
		t.Fatalf("chunk JSON is invalid: %v, content: %s", err, jsonPart)
	}

	// Verify it has expected OpenAI structure
	if _, ok := parsed["choices"]; !ok {
		t.Fatalf("chunk should have 'choices' field")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}


// Property 7: 流式响应聚合
// For any streaming response sequence, calling GetInternalResponse should return
// a complete InternalLLMResponse where:
// - All text content is correctly concatenated
// - All tool calls are correctly aggregated
// - Usage statistics are correctly accumulated
// **Validates: Requirements 7.3, 7.5**

func TestStreamResponseAggregation_OpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a sequence of streaming chunks
		chunks := generateOpenAIStreamChunks(t)

		// Create inbound transformer
		inboundTransformer := openai.NewChatInbound()
		ctx := context.Background()

		// Process each chunk
		var expectedContent string
		var expectedFinishReason *string

		for _, chunk := range chunks {
			// Transform the chunk
			_, err := inboundTransformer.TransformStream(ctx, chunk)
			if err != nil {
				t.Fatalf("TransformStream failed: %v", err)
			}

			// Track expected content
			for _, choice := range chunk.Choices {
				if choice.Delta != nil {
					expectedContent += choice.Delta.Content.GetText()
				}
				if choice.FinishReason != nil {
					expectedFinishReason = choice.FinishReason
				}
			}
		}

		// Get aggregated response
		aggregated, err := inboundTransformer.GetInternalResponse(ctx)
		if err != nil {
			t.Fatalf("GetInternalResponse failed: %v", err)
		}

		// Verify aggregated content
		if len(aggregated.Choices) > 0 && aggregated.Choices[0].Message != nil {
			actualContent := aggregated.Choices[0].Message.Content.GetText()
			if actualContent != expectedContent {
				t.Fatalf("aggregated content mismatch: expected %q, got %q", expectedContent, actualContent)
			}
		}

		// Verify finish reason is preserved
		if expectedFinishReason != nil && len(aggregated.Choices) > 0 {
			if aggregated.Choices[0].FinishReason == nil {
				t.Fatalf("expected finish reason %s, got nil", *expectedFinishReason)
			}
			if *aggregated.Choices[0].FinishReason != *expectedFinishReason {
				t.Fatalf("finish reason mismatch: expected %s, got %s", *expectedFinishReason, *aggregated.Choices[0].FinishReason)
			}
		}
	})
}

// generateOpenAIStreamChunks generates a sequence of OpenAI streaming chunks
func generateOpenAIStreamChunks(t *rapid.T) []*model.InternalLLMResponse {
	numChunks := rapid.IntRange(1, 10).Draw(t, "numChunks")
	chunks := make([]*model.InternalLLMResponse, numChunks)

	id := "chatcmpl-" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "id")
	modelName := "gpt-4"
	created := int64(rapid.IntRange(1600000000, 1800000000).Draw(t, "created"))

	for i := 0; i < numChunks; i++ {
		text := rapid.StringN(1, 5, 20).Draw(t, "chunkText")

		chunk := &model.InternalLLMResponse{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   modelName,
			Choices: []model.Choice{
				{
					Index: 0,
					Delta: &model.Message{
						Content: model.MessageContent{
							Content: &text,
						},
					},
				},
			},
		}

		// Last chunk has finish reason
		if i == numChunks-1 {
			finishReason := "stop"
			chunk.Choices[0].FinishReason = &finishReason

			// Optionally add usage
			if rapid.Bool().Draw(t, "hasUsage") {
				promptTokens := int64(rapid.IntRange(1, 1000).Draw(t, "promptTokens"))
				completionTokens := int64(rapid.IntRange(1, 1000).Draw(t, "completionTokens"))
				chunk.Usage = &model.Usage{
					PromptTokens:     promptTokens,
					CompletionTokens: completionTokens,
					TotalTokens:      promptTokens + completionTokens,
				}
			}
		}

		chunks[i] = chunk
	}

	return chunks
}


// Property 8: 错误响应转换
// For any upstream error response, the system should:
// - Correctly parse error information into InternalLLMResponse.Error
// - Convert to the client's expected error format
// - Preserve HTTP status codes, error codes, and error messages
// **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

func TestErrorResponseTransform_AnthropicToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random Anthropic error response
		errorResp := generateAnthropicErrorResponse(t)

		// Create inbound transformer (OpenAI)
		inboundTransformer := openai.NewChatInbound()
		ctx := context.Background()

		// Transform error response to client format
		clientBody, err := inboundTransformer.TransformResponse(ctx, errorResp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Parse the client response
		var clientResp map[string]any
		if err := json.Unmarshal(clientBody, &clientResp); err != nil {
			t.Fatalf("failed to unmarshal client response: %v", err)
		}

		// Verify error structure is preserved
		verifyErrorResponseStructure(t, errorResp, clientResp)
	})
}

func TestErrorResponseTransform_GeminiToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random Gemini error response
		errorResp := generateGeminiErrorResponse(t)

		// Create inbound transformer (OpenAI)
		inboundTransformer := openai.NewChatInbound()
		ctx := context.Background()

		// Transform error response to client format
		clientBody, err := inboundTransformer.TransformResponse(ctx, errorResp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Parse the client response
		var clientResp map[string]any
		if err := json.Unmarshal(clientBody, &clientResp); err != nil {
			t.Fatalf("failed to unmarshal client response: %v", err)
		}

		// Verify error structure is preserved
		verifyErrorResponseStructure(t, errorResp, clientResp)
	})
}

// generateAnthropicErrorResponse generates a random Anthropic-style error response
func generateAnthropicErrorResponse(t *rapid.T) *model.InternalLLMResponse {
	errorTypes := []string{"invalid_api_key", "rate_limit_error", "overloaded_error", "invalid_request_error"}
	errorType := errorTypes[rapid.IntRange(0, len(errorTypes)-1).Draw(t, "errorType")]

	statusCodes := []int{400, 401, 403, 429, 500, 503}
	statusCode := statusCodes[rapid.IntRange(0, len(statusCodes)-1).Draw(t, "statusCode")]

	message := rapid.StringN(10, 50, 200).Draw(t, "errorMessage")

	return &model.InternalLLMResponse{
		Object: "error",
		Error: &model.ResponseError{
			StatusCode: statusCode,
			Detail: model.ErrorDetail{
				Type:    errorType,
				Message: message,
			},
		},
	}
}

// generateGeminiErrorResponse generates a random Gemini-style error response
func generateGeminiErrorResponse(t *rapid.T) *model.InternalLLMResponse {
	errorStatuses := []string{"INVALID_ARGUMENT", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "INTERNAL"}
	errorStatus := errorStatuses[rapid.IntRange(0, len(errorStatuses)-1).Draw(t, "errorStatus")]

	statusCodes := []int{400, 401, 403, 429, 500}
	statusCode := statusCodes[rapid.IntRange(0, len(statusCodes)-1).Draw(t, "statusCode")]

	message := rapid.StringN(10, 50, 200).Draw(t, "errorMessage")

	return &model.InternalLLMResponse{
		Object: "error",
		Error: &model.ResponseError{
			StatusCode: statusCode,
			Detail: model.ErrorDetail{
				Code:    errorStatus,
				Type:    "api_error",
				Message: message,
			},
		},
	}
}

// verifyErrorResponseStructure verifies that the error response has correct structure
func verifyErrorResponseStructure(t *rapid.T, original *model.InternalLLMResponse, clientResp map[string]any) {
	// Verify error field exists
	errorField, ok := clientResp["error"]
	if !ok {
		t.Fatalf("client response should have 'error' field")
	}

	errorMap, ok := errorField.(map[string]any)
	if !ok {
		t.Fatalf("error field should be an object")
	}

	// Verify message is preserved (check both 'message' and nested 'error.message')
	if original.Error != nil && original.Error.Detail.Message != "" {
		// The error structure might be nested differently
		var message string
		if msg, ok := errorMap["message"].(string); ok {
			message = msg
		} else if errDetail, ok := errorMap["error"].(map[string]any); ok {
			if msg, ok := errDetail["message"].(string); ok {
				message = msg
			}
		}

		if message == "" {
			// Check if the message is in the Detail field
			if detail, ok := errorMap["Detail"].(map[string]any); ok {
				if msg, ok := detail["message"].(string); ok {
					message = msg
				}
			}
		}

		if message != original.Error.Detail.Message {
			t.Fatalf("error message mismatch: expected %q, got %q", original.Error.Detail.Message, message)
		}
	}

	// Verify type is present (check various locations)
	hasType := false
	if _, ok := errorMap["type"]; ok {
		hasType = true
	} else if errDetail, ok := errorMap["error"].(map[string]any); ok {
		if _, ok := errDetail["type"]; ok {
			hasType = true
		}
	} else if detail, ok := errorMap["Detail"].(map[string]any); ok {
		if _, ok := detail["type"]; ok {
			hasType = true
		}
	}

	if !hasType {
		t.Fatalf("error should have 'type' field somewhere in the structure")
	}
}

// TestInboundTypeFromAPIFormat tests the mapping from API format to inbound type
func TestInboundTypeFromAPIFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		formats := []model.APIFormat{
			model.APIFormatOpenAIChat,
			model.APIFormatOpenAIResponse,
			model.APIFormatAnthropic,
			model.APIFormatGemini,
		}

		format := formats[rapid.IntRange(0, len(formats)-1).Draw(t, "format")]
		inboundType := inbound.InboundTypeFromAPIFormat(format)

		// Verify the mapping is correct
		switch format {
		case model.APIFormatOpenAIChat:
			if inboundType != inbound.InboundTypeOpenAIChat {
				t.Fatalf("expected InboundTypeOpenAIChat for %s, got %v", format, inboundType)
			}
		case model.APIFormatOpenAIResponse:
			if inboundType != inbound.InboundTypeOpenAIResponse {
				t.Fatalf("expected InboundTypeOpenAIResponse for %s, got %v", format, inboundType)
			}
		case model.APIFormatAnthropic:
			if inboundType != inbound.InboundTypeAnthropic {
				t.Fatalf("expected InboundTypeAnthropic for %s, got %v", format, inboundType)
			}
		default:
			// Unknown formats should default to OpenAI Chat
			if inboundType != inbound.InboundTypeOpenAIChat {
				t.Fatalf("expected InboundTypeOpenAIChat for unknown format %s, got %v", format, inboundType)
			}
		}
	})
}
