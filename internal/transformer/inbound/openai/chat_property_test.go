package openai

import (
	"context"
	"encoding/json"
	"testing"

	"gpt-load/internal/transformer/model"

	"pgregory.net/rapid"
)

// Property 1: 入站转换往返一致性
// For any valid OpenAI Chat format request, converting it to InternalLLMRequest
// and then back should preserve all semantically important fields.
// **Validates: Requirements 1.1, 1.5, 2.5**

func TestChatInbound_TransformRequest_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid OpenAI Chat request
		req := generateValidOpenAIChatRequest(t)

		// Marshal to JSON (simulating client request)
		body, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("failed to marshal request: %v", err)
		}

		// Transform request using ChatInbound
		inbound := NewChatInbound()
		ctx := context.Background()

		internalReq, err := inbound.TransformRequest(ctx, body)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify the internal request preserves semantic fields
		verifyRequestPreservation(t, req, internalReq)
	})
}

func TestChatInbound_TransformResponse_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal response
		resp := generateValidInternalResponse(t)

		// Transform response using ChatInbound
		inbound := NewChatInbound()
		ctx := context.Background()

		// Transform to client format
		clientBody, err := inbound.TransformResponse(ctx, resp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Parse the client response
		var clientResp model.InternalLLMResponse
		if err := json.Unmarshal(clientBody, &clientResp); err != nil {
			t.Fatalf("failed to unmarshal client response: %v", err)
		}

		// Verify the response preserves semantic fields
		verifyResponsePreservation(t, resp, &clientResp)
	})
}

// generateValidOpenAIChatRequest generates a random valid OpenAI Chat request
func generateValidOpenAIChatRequest(t *rapid.T) *model.InternalLLMRequest {
	// Generate model name
	modelName := rapid.StringMatching(`^(gpt-4|gpt-3\.5-turbo|claude-3)[a-z0-9-]*$`).Draw(t, "model")

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

// generateValidInternalResponse generates a random valid internal response
func generateValidInternalResponse(t *rapid.T) *model.InternalLLMResponse {
	id := "chatcmpl-" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "id")
	modelName := rapid.StringMatching(`^(gpt-4|gpt-3\.5-turbo)[a-z0-9-]*$`).Draw(t, "model")
	created := int64(rapid.IntRange(1600000000, 1800000000).Draw(t, "created"))

	// Generate choices
	numChoices := rapid.IntRange(1, 3).Draw(t, "numChoices")
	choices := make([]model.Choice, numChoices)

	finishReasons := []string{"stop", "length", "tool_calls"}
	for i := 0; i < numChoices; i++ {
		content := rapid.StringN(1, 100, 1000).Draw(t, "responseContent")
		finishReason := finishReasons[rapid.IntRange(0, len(finishReasons)-1).Draw(t, "finishReasonIndex")]

		choices[i] = model.Choice{
			Index: i,
			Message: &model.Message{
				Role: "assistant",
				Content: model.MessageContent{
					Content: &content,
				},
			},
			FinishReason: &finishReason,
		}
	}

	resp := &model.InternalLLMResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   modelName,
		Choices: choices,
	}

	// Optionally add usage
	if rapid.Bool().Draw(t, "hasUsage") {
		promptTokens := int64(rapid.IntRange(1, 1000).Draw(t, "promptTokens"))
		completionTokens := int64(rapid.IntRange(1, 1000).Draw(t, "completionTokens"))
		resp.Usage = &model.Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		}
	}

	return resp
}

// verifyRequestPreservation verifies that the internal request preserves semantic fields
func verifyRequestPreservation(t *rapid.T, original, internal *model.InternalLLMRequest) {
	// Verify model
	if original.Model != internal.Model {
		t.Fatalf("model mismatch: expected %s, got %s", original.Model, internal.Model)
	}

	// Verify messages count
	if len(original.Messages) != len(internal.Messages) {
		t.Fatalf("messages count mismatch: expected %d, got %d", len(original.Messages), len(internal.Messages))
	}

	// Verify each message
	for i, origMsg := range original.Messages {
		intMsg := internal.Messages[i]

		if origMsg.Role != intMsg.Role {
			t.Fatalf("message[%d] role mismatch: expected %s, got %s", i, origMsg.Role, intMsg.Role)
		}

		origContent := origMsg.Content.GetText()
		intContent := intMsg.Content.GetText()
		if origContent != intContent {
			t.Fatalf("message[%d] content mismatch: expected %s, got %s", i, origContent, intContent)
		}
	}

	// Verify temperature
	if (original.Temperature == nil) != (internal.Temperature == nil) {
		t.Fatalf("temperature presence mismatch")
	}
	if original.Temperature != nil && *original.Temperature != *internal.Temperature {
		t.Fatalf("temperature mismatch: expected %f, got %f", *original.Temperature, *internal.Temperature)
	}

	// Verify max_tokens
	if (original.MaxTokens == nil) != (internal.MaxTokens == nil) {
		t.Fatalf("max_tokens presence mismatch")
	}
	if original.MaxTokens != nil && *original.MaxTokens != *internal.MaxTokens {
		t.Fatalf("max_tokens mismatch: expected %d, got %d", *original.MaxTokens, *internal.MaxTokens)
	}

	// Verify stream
	if (original.Stream == nil) != (internal.Stream == nil) {
		t.Fatalf("stream presence mismatch")
	}
	if original.Stream != nil && *original.Stream != *internal.Stream {
		t.Fatalf("stream mismatch: expected %v, got %v", *original.Stream, *internal.Stream)
	}

	// Verify top_p
	if (original.TopP == nil) != (internal.TopP == nil) {
		t.Fatalf("top_p presence mismatch")
	}
	if original.TopP != nil && *original.TopP != *internal.TopP {
		t.Fatalf("top_p mismatch: expected %f, got %f", *original.TopP, *internal.TopP)
	}

	// Verify tools count
	if len(original.Tools) != len(internal.Tools) {
		t.Fatalf("tools count mismatch: expected %d, got %d", len(original.Tools), len(internal.Tools))
	}

	// Verify each tool
	for i, origTool := range original.Tools {
		intTool := internal.Tools[i]

		if origTool.Type != intTool.Type {
			t.Fatalf("tool[%d] type mismatch: expected %s, got %s", i, origTool.Type, intTool.Type)
		}
		if origTool.Function.Name != intTool.Function.Name {
			t.Fatalf("tool[%d] function name mismatch: expected %s, got %s", i, origTool.Function.Name, intTool.Function.Name)
		}
	}

	// Verify RawAPIFormat is set correctly
	if internal.RawAPIFormat != model.APIFormatOpenAIChat {
		t.Fatalf("RawAPIFormat mismatch: expected %s, got %s", model.APIFormatOpenAIChat, internal.RawAPIFormat)
	}
}

// verifyResponsePreservation verifies that the client response preserves semantic fields
func verifyResponsePreservation(t *rapid.T, original, client *model.InternalLLMResponse) {
	// Verify ID
	if original.ID != client.ID {
		t.Fatalf("id mismatch: expected %s, got %s", original.ID, client.ID)
	}

	// Verify Object
	if original.Object != client.Object {
		t.Fatalf("object mismatch: expected %s, got %s", original.Object, client.Object)
	}

	// Verify Created
	if original.Created != client.Created {
		t.Fatalf("created mismatch: expected %d, got %d", original.Created, client.Created)
	}

	// Verify Model
	if original.Model != client.Model {
		t.Fatalf("model mismatch: expected %s, got %s", original.Model, client.Model)
	}

	// Verify choices count
	if len(original.Choices) != len(client.Choices) {
		t.Fatalf("choices count mismatch: expected %d, got %d", len(original.Choices), len(client.Choices))
	}

	// Verify each choice
	for i, origChoice := range original.Choices {
		clientChoice := client.Choices[i]

		if origChoice.Index != clientChoice.Index {
			t.Fatalf("choice[%d] index mismatch: expected %d, got %d", i, origChoice.Index, clientChoice.Index)
		}

		// Verify message content
		if origChoice.Message != nil && clientChoice.Message != nil {
			origContent := origChoice.Message.Content.GetText()
			clientContent := clientChoice.Message.Content.GetText()
			if origContent != clientContent {
				t.Fatalf("choice[%d] message content mismatch: expected %s, got %s", i, origContent, clientContent)
			}
		}

		// Verify finish reason
		if (origChoice.FinishReason == nil) != (clientChoice.FinishReason == nil) {
			t.Fatalf("choice[%d] finish_reason presence mismatch", i)
		}
		if origChoice.FinishReason != nil && *origChoice.FinishReason != *clientChoice.FinishReason {
			t.Fatalf("choice[%d] finish_reason mismatch: expected %s, got %s", i, *origChoice.FinishReason, *clientChoice.FinishReason)
		}
	}

	// Verify usage
	if (original.Usage == nil) != (client.Usage == nil) {
		t.Fatalf("usage presence mismatch")
	}
	if original.Usage != nil {
		if original.Usage.PromptTokens != client.Usage.PromptTokens {
			t.Fatalf("usage prompt_tokens mismatch: expected %d, got %d", original.Usage.PromptTokens, client.Usage.PromptTokens)
		}
		if original.Usage.CompletionTokens != client.Usage.CompletionTokens {
			t.Fatalf("usage completion_tokens mismatch: expected %d, got %d", original.Usage.CompletionTokens, client.Usage.CompletionTokens)
		}
		if original.Usage.TotalTokens != client.Usage.TotalTokens {
			t.Fatalf("usage total_tokens mismatch: expected %d, got %d", original.Usage.TotalTokens, client.Usage.TotalTokens)
		}
	}
}
