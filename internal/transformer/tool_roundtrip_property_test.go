package transformer

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"gpt-load/internal/transformer/inbound/anthropic"
	inboundOpenai "gpt-load/internal/transformer/inbound/openai"
	"gpt-load/internal/transformer/model"
	outboundAnthropic "gpt-load/internal/transformer/outbound/anthropic"
	outboundGemini "gpt-load/internal/transformer/outbound/gemini"

	"pgregory.net/rapid"
)

// Suppress unused import warning
var _ = inboundOpenai.NewChatInbound

// Property 9: 工具调用往返一致性
// For any request/response containing tool definitions and tool calls,
// cross-format conversion should:
// - Preserve tool names and parameter definitions
// - Preserve tool call IDs
// - Correctly map OpenAI tools <-> Anthropic tools format
// - Correctly map OpenAI tool_calls <-> Anthropic tool_use format
// **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

// TestToolDefinition_OpenAIToAnthropic_RoundTrip tests that tool definitions
// are preserved when converting from OpenAI to Anthropic format
func TestToolDefinition_OpenAIToAnthropic_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a request with tools
		req := generateRequestWithTools(t)

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

		// Verify tools are present
		tools, ok := anthropicReq["tools"].([]interface{})
		if !ok {
			t.Fatalf("tools should be an array")
		}

		// Verify tool count matches
		if len(tools) != len(req.Tools) {
			t.Fatalf("tool count mismatch: expected %d, got %d", len(req.Tools), len(tools))
		}

		// Verify each tool's properties are preserved
		for i, tool := range tools {
			toolMap, ok := tool.(map[string]interface{})
			if !ok {
				t.Fatalf("tool[%d] should be an object", i)
			}

			// Verify name is preserved
			name, _ := toolMap["name"].(string)
			if name != req.Tools[i].Function.Name {
				t.Fatalf("tool[%d] name mismatch: expected %s, got %s", i, req.Tools[i].Function.Name, name)
			}

			// Verify description is preserved
			desc, _ := toolMap["description"].(string)
			if desc != req.Tools[i].Function.Description {
				t.Fatalf("tool[%d] description mismatch: expected %s, got %s", i, req.Tools[i].Function.Description, desc)
			}

			// Verify input_schema is present (Anthropic uses input_schema instead of parameters)
			if _, ok := toolMap["input_schema"]; !ok {
				t.Fatalf("tool[%d] missing input_schema", i)
			}
		}
	})
}

// TestToolDefinition_OpenAIToGemini_RoundTrip tests that tool definitions
// are preserved when converting from OpenAI to Gemini format
func TestToolDefinition_OpenAIToGemini_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a request with tools
		req := generateRequestWithTools(t)

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

		// Verify tools are present
		tools, ok := geminiReq["tools"].([]interface{})
		if !ok {
			t.Fatalf("tools should be an array")
		}

		// Gemini wraps function declarations in a single tool object
		if len(tools) != 1 {
			t.Fatalf("Gemini tools should have exactly 1 element, got %d", len(tools))
		}

		toolObj, ok := tools[0].(map[string]interface{})
		if !ok {
			t.Fatalf("tool[0] should be an object")
		}

		declarations, ok := toolObj["functionDeclarations"].([]interface{})
		if !ok {
			t.Fatalf("functionDeclarations should be an array")
		}

		// Verify declaration count matches
		if len(declarations) != len(req.Tools) {
			t.Fatalf("functionDeclarations count mismatch: expected %d, got %d", len(req.Tools), len(declarations))
		}

		// Verify each declaration's properties are preserved
		for i, decl := range declarations {
			declMap, ok := decl.(map[string]interface{})
			if !ok {
				t.Fatalf("functionDeclaration[%d] should be an object", i)
			}

			// Verify name is preserved
			name, _ := declMap["name"].(string)
			if name != req.Tools[i].Function.Name {
				t.Fatalf("functionDeclaration[%d] name mismatch: expected %s, got %s", i, req.Tools[i].Function.Name, name)
			}

			// Verify description is preserved
			desc, _ := declMap["description"].(string)
			if desc != req.Tools[i].Function.Description {
				t.Fatalf("functionDeclaration[%d] description mismatch: expected %s, got %s", i, req.Tools[i].Function.Description, desc)
			}
		}
	})
}


// TestToolDefinition_AnthropicToOpenAI_RoundTrip tests that Anthropic tool definitions
// are correctly converted to OpenAI format
func TestToolDefinition_AnthropicToOpenAI_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate an Anthropic request with tools
		anthropicReq := generateAnthropicRequestWithTools(t)

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

		// Verify tool count matches
		if len(internalReq.Tools) != len(anthropicReq.Tools) {
			t.Fatalf("tool count mismatch: expected %d, got %d", len(anthropicReq.Tools), len(internalReq.Tools))
		}

		// Verify each tool's properties are preserved
		for i, tool := range internalReq.Tools {
			// Verify type is "function"
			if tool.Type != "function" {
				t.Fatalf("tool[%d] type should be 'function', got %s", i, tool.Type)
			}

			// Verify name is preserved
			if tool.Function.Name != anthropicReq.Tools[i].Name {
				t.Fatalf("tool[%d] name mismatch: expected %s, got %s", i, anthropicReq.Tools[i].Name, tool.Function.Name)
			}

			// Verify description is preserved
			if tool.Function.Description != anthropicReq.Tools[i].Description {
				t.Fatalf("tool[%d] description mismatch: expected %s, got %s", i, anthropicReq.Tools[i].Description, tool.Function.Description)
			}

			// Verify parameters (input_schema) is preserved
			if string(tool.Function.Parameters) != string(anthropicReq.Tools[i].InputSchema) {
				t.Fatalf("tool[%d] parameters mismatch: expected %s, got %s", i, string(anthropicReq.Tools[i].InputSchema), string(tool.Function.Parameters))
			}
		}
	})
}

// TestToolCallResponse_AnthropicToOpenAI tests that Anthropic tool_use responses
// are correctly converted to OpenAI tool_calls format
func TestToolCallResponse_AnthropicToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate an Anthropic response with tool_use
		anthropicResp := generateAnthropicResponseWithToolUse(t)

		// Marshal to JSON
		body, err := json.Marshal(anthropicResp)
		if err != nil {
			t.Fatalf("failed to marshal Anthropic response: %v", err)
		}

		// Simulate the outbound transformer parsing the response
		outbound := outboundAnthropic.NewMessagesOutbound()
		ctx := context.Background()

		// Create a mock HTTP response
		mockResp := createMockHTTPResponse(200, body)
		defer mockResp.Body.Close()

		internalResp, err := outbound.TransformResponse(ctx, mockResp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Count tool_use blocks in original response
		toolUseCount := 0
		for _, block := range anthropicResp.Content {
			if block.Type == "tool_use" {
				toolUseCount++
			}
		}

		// Verify tool calls are present in internal response
		if len(internalResp.Choices) == 0 {
			t.Fatalf("internal response should have at least one choice")
		}

		choice := internalResp.Choices[0]
		if choice.Message == nil {
			t.Fatalf("choice should have a message")
		}

		// Verify tool call count matches
		if len(choice.Message.ToolCalls) != toolUseCount {
			t.Fatalf("tool call count mismatch: expected %d, got %d", toolUseCount, len(choice.Message.ToolCalls))
		}

		// Verify each tool call's properties are preserved
		toolUseIdx := 0
		for _, block := range anthropicResp.Content {
			if block.Type != "tool_use" {
				continue
			}

			tc := choice.Message.ToolCalls[toolUseIdx]

			// Verify ID is preserved
			if block.ID != nil && tc.ID != *block.ID {
				t.Fatalf("tool call[%d] ID mismatch: expected %s, got %s", toolUseIdx, *block.ID, tc.ID)
			}

			// Verify type is "function"
			if tc.Type != "function" {
				t.Fatalf("tool call[%d] type should be 'function', got %s", toolUseIdx, tc.Type)
			}

			// Verify function name is preserved
			if block.Name != nil && tc.Function.Name != *block.Name {
				t.Fatalf("tool call[%d] function name mismatch: expected %s, got %s", toolUseIdx, *block.Name, tc.Function.Name)
			}

			// Verify arguments are preserved
			if block.Input != nil {
				expectedArgs := string(*block.Input)
				if tc.Function.Arguments != expectedArgs {
					t.Fatalf("tool call[%d] arguments mismatch: expected %s, got %s", toolUseIdx, expectedArgs, tc.Function.Arguments)
				}
			}

			toolUseIdx++
		}
	})
}

// TestToolCallResponse_GeminiToOpenAI tests that Gemini functionCall responses
// are correctly converted to OpenAI tool_calls format
func TestToolCallResponse_GeminiToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a Gemini response with functionCall
		geminiResp := generateGeminiResponseWithFunctionCall(t)

		// Marshal to JSON
		body, err := json.Marshal(geminiResp)
		if err != nil {
			t.Fatalf("failed to marshal Gemini response: %v", err)
		}

		// Simulate the outbound transformer parsing the response
		outbound := outboundGemini.NewMessagesOutbound()
		ctx := context.Background()

		// Create a mock HTTP response
		mockResp := createMockHTTPResponse(200, body)
		defer mockResp.Body.Close()

		internalResp, err := outbound.TransformResponse(ctx, mockResp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Count functionCall parts in original response
		functionCallCount := 0
		for _, candidate := range geminiResp.Candidates {
			if candidate.Content != nil {
				for _, part := range candidate.Content.Parts {
					if part.FunctionCall != nil {
						functionCallCount++
					}
				}
			}
		}

		// Verify tool calls are present in internal response
		if len(internalResp.Choices) == 0 {
			t.Fatalf("internal response should have at least one choice")
		}

		choice := internalResp.Choices[0]
		if choice.Message == nil {
			t.Fatalf("choice should have a message")
		}

		// Verify tool call count matches
		if len(choice.Message.ToolCalls) != functionCallCount {
			t.Fatalf("tool call count mismatch: expected %d, got %d", functionCallCount, len(choice.Message.ToolCalls))
		}

		// Verify each tool call's properties are preserved
		tcIdx := 0
		for _, candidate := range geminiResp.Candidates {
			if candidate.Content == nil {
				continue
			}
			for _, part := range candidate.Content.Parts {
				if part.FunctionCall == nil {
					continue
				}

				tc := choice.Message.ToolCalls[tcIdx]

				// Verify type is "function"
				if tc.Type != "function" {
					t.Fatalf("tool call[%d] type should be 'function', got %s", tcIdx, tc.Type)
				}

				// Verify function name is preserved
				if tc.Function.Name != part.FunctionCall.Name {
					t.Fatalf("tool call[%d] function name mismatch: expected %s, got %s", tcIdx, part.FunctionCall.Name, tc.Function.Name)
				}

				// Verify arguments are preserved
				if part.FunctionCall.Args != nil {
					expectedArgs := string(part.FunctionCall.Args)
					if tc.Function.Arguments != expectedArgs {
						t.Fatalf("tool call[%d] arguments mismatch: expected %s, got %s", tcIdx, expectedArgs, tc.Function.Arguments)
					}
				}

				// Verify ID is generated (Gemini doesn't provide IDs, so we generate them)
				if tc.ID == "" {
					t.Fatalf("tool call[%d] should have an ID", tcIdx)
				}

				tcIdx++
			}
		}
	})
}


// TestToolCallResponse_OpenAIToAnthropic tests that OpenAI tool_calls responses
// are correctly converted to Anthropic tool_use format
func TestToolCallResponse_OpenAIToAnthropic(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate an internal response with tool calls
		internalResp := generateInternalResponseWithToolCalls(t)

		// Convert to Anthropic format using inbound transformer
		inbound := anthropic.NewMessagesInbound()
		ctx := context.Background()

		clientBody, err := inbound.TransformResponse(ctx, internalResp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Parse the Anthropic response
		var anthropicResp anthropicMessageResponse
		if err := json.Unmarshal(clientBody, &anthropicResp); err != nil {
			t.Fatalf("failed to unmarshal Anthropic response: %v", err)
		}

		// Count tool calls in original response
		toolCallCount := 0
		if len(internalResp.Choices) > 0 && internalResp.Choices[0].Message != nil {
			toolCallCount = len(internalResp.Choices[0].Message.ToolCalls)
		}

		// Count tool_use blocks in Anthropic response
		toolUseCount := 0
		for _, block := range anthropicResp.Content {
			if block.Type == "tool_use" {
				toolUseCount++
			}
		}

		// Verify tool_use count matches
		if toolUseCount != toolCallCount {
			t.Fatalf("tool_use count mismatch: expected %d, got %d", toolCallCount, toolUseCount)
		}

		// Verify each tool_use block's properties are preserved
		if len(internalResp.Choices) > 0 && internalResp.Choices[0].Message != nil {
			toolUseIdx := 0
			for _, block := range anthropicResp.Content {
				if block.Type != "tool_use" {
					continue
				}

				tc := internalResp.Choices[0].Message.ToolCalls[toolUseIdx]

				// Verify ID is preserved
				if block.ID == nil || *block.ID != tc.ID {
					expectedID := tc.ID
					gotID := ""
					if block.ID != nil {
						gotID = *block.ID
					}
					t.Fatalf("tool_use[%d] ID mismatch: expected %s, got %s", toolUseIdx, expectedID, gotID)
				}

				// Verify name is preserved
				if block.Name == nil || *block.Name != tc.Function.Name {
					expectedName := tc.Function.Name
					gotName := ""
					if block.Name != nil {
						gotName = *block.Name
					}
					t.Fatalf("tool_use[%d] name mismatch: expected %s, got %s", toolUseIdx, expectedName, gotName)
				}

				// Verify input is preserved
				if block.Input != nil {
					expectedInput := tc.Function.Arguments
					gotInput := string(*block.Input)
					if gotInput != expectedInput {
						t.Fatalf("tool_use[%d] input mismatch: expected %s, got %s", toolUseIdx, expectedInput, gotInput)
					}
				}

				toolUseIdx++
			}
		}
	})
}

// TestToolCallIDPreservation tests that tool call IDs are preserved across format conversions
func TestToolCallIDPreservation(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a unique tool call ID
		toolCallID := "toolu_" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "toolCallID")
		toolName := rapid.StringMatching(`^[a-z_][a-z0-9_]*$`).Draw(t, "toolName")
		toolArgs := `{"param":"value"}`

		// Create an internal response with the tool call
		internalResp := &model.InternalLLMResponse{
			ID:      "msg_test",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "gpt-4",
			Choices: []model.Choice{
				{
					Index: 0,
					Message: &model.Message{
						Role: "assistant",
						ToolCalls: []model.ToolCall{
							{
								ID:   toolCallID,
								Type: "function",
								Function: model.FunctionCall{
									Name:      toolName,
									Arguments: toolArgs,
								},
								Index: 0,
							},
						},
					},
				},
			},
		}

		// Convert to Anthropic format
		inbound := anthropic.NewMessagesInbound()
		ctx := context.Background()

		clientBody, err := inbound.TransformResponse(ctx, internalResp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Parse the Anthropic response
		var anthropicResp anthropicMessageResponse
		if err := json.Unmarshal(clientBody, &anthropicResp); err != nil {
			t.Fatalf("failed to unmarshal Anthropic response: %v", err)
		}

		// Find the tool_use block and verify ID is preserved
		found := false
		for _, block := range anthropicResp.Content {
			if block.Type == "tool_use" {
				found = true
				if block.ID == nil || *block.ID != toolCallID {
					gotID := ""
					if block.ID != nil {
						gotID = *block.ID
					}
					t.Fatalf("tool call ID not preserved: expected %s, got %s", toolCallID, gotID)
				}
				break
			}
		}

		if !found {
			t.Fatalf("tool_use block not found in Anthropic response")
		}
	})
}

// TestToolCallFullRoundTrip tests the full round-trip of tool calls:
// OpenAI request -> Anthropic request -> Anthropic response -> OpenAI response
func TestToolCallFullRoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate an OpenAI request with tools
		openaiReq := generateRequestWithTools(t)

		// Step 1: Convert OpenAI request to Anthropic request
		outbound := outboundAnthropic.NewMessagesOutbound()
		ctx := context.Background()

		httpReq, err := outbound.TransformRequest(ctx, openaiReq, "https://api.anthropic.com", "sk-ant-test")
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Read the Anthropic request body
		anthropicReqBody, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}

		// Verify tools are present in Anthropic request
		var anthropicReq map[string]interface{}
		if err := json.Unmarshal(anthropicReqBody, &anthropicReq); err != nil {
			t.Fatalf("failed to unmarshal Anthropic request: %v", err)
		}

		tools, ok := anthropicReq["tools"].([]interface{})
		if !ok || len(tools) != len(openaiReq.Tools) {
			t.Fatalf("tools not properly converted to Anthropic format")
		}

		// Step 2: Simulate Anthropic response with tool_use
		anthropicResp := generateAnthropicResponseWithToolUse(t)
		anthropicRespBody, _ := json.Marshal(anthropicResp)

		// Step 3: Convert Anthropic response to internal format
		mockResp := createMockHTTPResponse(200, anthropicRespBody)
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

		// Verify the OpenAI response is valid JSON
		var openaiResp map[string]interface{}
		if err := json.Unmarshal(openaiRespBody, &openaiResp); err != nil {
			t.Fatalf("failed to unmarshal OpenAI response: %v", err)
		}

		// Verify tool_calls are present in OpenAI response
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

		// Verify tool_calls are present if the Anthropic response had tool_use
		hasToolUse := false
		for _, block := range anthropicResp.Content {
			if block.Type == "tool_use" {
				hasToolUse = true
				break
			}
		}

		if hasToolUse {
			toolCalls, ok := message["tool_calls"].([]interface{})
			if !ok || len(toolCalls) == 0 {
				t.Fatalf("OpenAI response should have tool_calls when Anthropic response has tool_use")
			}
		}
	})
}


// Helper types for parsing responses

type anthropicMessageResponse struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Role       string                 `json:"role"`
	Content    []anthropicContentBlock `json:"content"`
	Model      string                 `json:"model"`
	StopReason *string                `json:"stop_reason,omitempty"`
}

type anthropicContentBlock struct {
	Type  string           `json:"type"`
	Text  *string          `json:"text,omitempty"`
	ID    *string          `json:"id,omitempty"`
	Name  *string          `json:"name,omitempty"`
	Input *json.RawMessage `json:"input,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicRequest struct {
	Model     string          `json:"model"`
	Messages  []anthropicMsg  `json:"messages"`
	MaxTokens int64           `json:"max_tokens"`
	Tools     []anthropicTool `json:"tools,omitempty"`
}

type anthropicMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type geminiResponse struct {
	Candidates []geminiCandidate `json:"candidates"`
}

type geminiCandidate struct {
	Content      *geminiContent `json:"content,omitempty"`
	FinishReason string         `json:"finishReason,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text         *string             `json:"text,omitempty"`
	FunctionCall *geminiFunctionCall `json:"functionCall,omitempty"`
}

type geminiFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

// Helper functions

func generateRequestWithTools(t *rapid.T) *model.InternalLLMRequest {
	// Generate model name
	modelName := rapid.SampledFrom([]string{
		"gpt-4",
		"gpt-4-turbo",
		"gpt-3.5-turbo",
	}).Draw(t, "model")

	// Generate messages
	numMessages := rapid.IntRange(1, 3).Draw(t, "numMessages")
	messages := make([]model.Message, numMessages)

	roles := []string{"user", "assistant"}
	for i := 0; i < numMessages; i++ {
		role := roles[i%2]
		content := rapid.StringN(1, 50, 200).Draw(t, "content")

		messages[i] = model.Message{
			Role: role,
			Content: model.MessageContent{
				Content: &content,
			},
		}
	}

	// Generate tools
	numTools := rapid.IntRange(1, 3).Draw(t, "numTools")
	tools := make([]model.Tool, numTools)
	for i := 0; i < numTools; i++ {
		toolName := rapid.StringMatching(`^[a-z_][a-z0-9_]*$`).Draw(t, "toolName")
		toolDesc := rapid.StringN(0, 20, 100).Draw(t, "toolDescription")
		tools[i] = model.Tool{
			Type: "function",
			Function: model.Function{
				Name:        toolName,
				Description: toolDesc,
				Parameters:  json.RawMessage(`{"type":"object","properties":{"param":{"type":"string"}}}`),
			},
		}
	}

	req := &model.InternalLLMRequest{
		Model:    modelName,
		Messages: messages,
		Tools:    tools,
	}

	return req
}

func generateAnthropicRequestWithTools(t *rapid.T) *anthropicRequest {
	// Generate model name
	modelName := rapid.SampledFrom([]string{
		"claude-3-opus-20240229",
		"claude-3-sonnet-20240229",
		"claude-3-haiku-20240307",
	}).Draw(t, "model")

	// Generate messages
	numMessages := rapid.IntRange(1, 3).Draw(t, "numMessages")
	messages := make([]anthropicMsg, numMessages)

	roles := []string{"user", "assistant"}
	for i := 0; i < numMessages; i++ {
		role := roles[i%2]
		content := rapid.StringN(1, 50, 200).Draw(t, "content")

		messages[i] = anthropicMsg{
			Role:    role,
			Content: content,
		}
	}

	// Generate tools
	numTools := rapid.IntRange(1, 3).Draw(t, "numTools")
	tools := make([]anthropicTool, numTools)
	for i := 0; i < numTools; i++ {
		toolName := rapid.StringMatching(`^[a-z_][a-z0-9_]*$`).Draw(t, "toolName")
		toolDesc := rapid.StringN(0, 20, 100).Draw(t, "toolDescription")
		tools[i] = anthropicTool{
			Name:        toolName,
			Description: toolDesc,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"param":{"type":"string"}}}`),
		}
	}

	maxTokens := int64(rapid.IntRange(100, 4096).Draw(t, "maxTokens"))

	return &anthropicRequest{
		Model:     modelName,
		Messages:  messages,
		MaxTokens: maxTokens,
		Tools:     tools,
	}
}

func generateAnthropicResponseWithToolUse(t *rapid.T) *anthropicMessageResponse {
	id := "msg_" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "id")
	modelName := rapid.SampledFrom([]string{
		"claude-3-opus-20240229",
		"claude-3-sonnet-20240229",
	}).Draw(t, "model")

	// Generate content blocks with tool_use
	numToolUse := rapid.IntRange(1, 2).Draw(t, "numToolUse")
	content := make([]anthropicContentBlock, 0)

	// Optionally add text block first
	if rapid.Bool().Draw(t, "hasTextBlock") {
		text := rapid.StringN(1, 50, 200).Draw(t, "textContent")
		content = append(content, anthropicContentBlock{
			Type: "text",
			Text: &text,
		})
	}

	// Add tool_use blocks
	for i := 0; i < numToolUse; i++ {
		toolID := "toolu_" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "toolID")
		toolName := rapid.StringMatching(`^[a-z_][a-z0-9_]*$`).Draw(t, "toolName")
		toolInput := json.RawMessage(`{"param":"value"}`)

		content = append(content, anthropicContentBlock{
			Type:  "tool_use",
			ID:    &toolID,
			Name:  &toolName,
			Input: &toolInput,
		})
	}

	stopReason := "tool_use"

	return &anthropicMessageResponse{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Content:    content,
		Model:      modelName,
		StopReason: &stopReason,
	}
}

func generateGeminiResponseWithFunctionCall(t *rapid.T) *geminiResponse {
	// Generate content with function calls
	numFunctionCalls := rapid.IntRange(1, 2).Draw(t, "numFunctionCalls")
	parts := make([]geminiPart, 0)

	// Optionally add text part first
	if rapid.Bool().Draw(t, "hasTextPart") {
		text := rapid.StringN(1, 50, 200).Draw(t, "textContent")
		parts = append(parts, geminiPart{
			Text: &text,
		})
	}

	// Add function call parts
	for i := 0; i < numFunctionCalls; i++ {
		funcName := rapid.StringMatching(`^[a-z_][a-z0-9_]*$`).Draw(t, "funcName")
		funcArgs := json.RawMessage(`{"param":"value"}`)

		parts = append(parts, geminiPart{
			FunctionCall: &geminiFunctionCall{
				Name: funcName,
				Args: funcArgs,
			},
		})
	}

	return &geminiResponse{
		Candidates: []geminiCandidate{
			{
				Content: &geminiContent{
					Role:  "model",
					Parts: parts,
				},
				FinishReason: "STOP",
			},
		},
	}
}

func generateInternalResponseWithToolCalls(t *rapid.T) *model.InternalLLMResponse {
	id := "chatcmpl-" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "id")
	modelName := rapid.SampledFrom([]string{
		"gpt-4",
		"gpt-4-turbo",
		"gpt-3.5-turbo",
	}).Draw(t, "model")

	// Generate tool calls
	numToolCalls := rapid.IntRange(1, 2).Draw(t, "numToolCalls")
	toolCalls := make([]model.ToolCall, numToolCalls)

	for i := 0; i < numToolCalls; i++ {
		toolID := "call_" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "toolID")
		toolName := rapid.StringMatching(`^[a-z_][a-z0-9_]*$`).Draw(t, "toolName")
		toolArgs := `{"param":"value"}`

		toolCalls[i] = model.ToolCall{
			ID:   toolID,
			Type: "function",
			Function: model.FunctionCall{
				Name:      toolName,
				Arguments: toolArgs,
			},
			Index: i,
		}
	}

	// Optionally add text content
	var textContent string
	if rapid.Bool().Draw(t, "hasTextContent") {
		textContent = rapid.StringN(1, 50, 200).Draw(t, "textContent")
	}

	finishReason := "tool_calls"

	return &model.InternalLLMResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   modelName,
		Choices: []model.Choice{
			{
				Index: 0,
				Message: &model.Message{
					Role: "assistant",
					Content: model.MessageContent{
						Content: &textContent,
					},
					ToolCalls: toolCalls,
				},
				FinishReason: &finishReason,
			},
		},
	}
}


// createMockHTTPResponse creates a mock HTTP response for testing
func createMockHTTPResponse(statusCode int, body []byte) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     make(http.Header),
	}
}
