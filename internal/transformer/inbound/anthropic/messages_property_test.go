package anthropic

import (
	"context"
	"encoding/json"
	"testing"

	"gpt-load/internal/transformer/model"

	"pgregory.net/rapid"
)

// Property 1: 入站转换往返一致性
// For any valid Anthropic Messages format request, converting it to InternalLLMRequest
// and then back should preserve all semantically important fields.
// **Validates: Requirements 1.1, 1.5, 2.6**

func TestMessagesInbound_TransformRequest_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid Anthropic Messages request
		req := generateValidAnthropicRequest(t)

		// Marshal to JSON (simulating client request)
		body, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("failed to marshal request: %v", err)
		}

		// Transform request using MessagesInbound
		inbound := NewMessagesInbound()
		ctx := context.Background()

		internalReq, err := inbound.TransformRequest(ctx, body)
		if err != nil {
			t.Fatalf("TransformRequest failed: %v", err)
		}

		// Verify the internal request preserves semantic fields
		verifyRequestPreservation(t, req, internalReq)
	})
}

func TestMessagesInbound_TransformResponse_RoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random valid internal response
		resp := generateValidInternalResponse(t)

		// Transform response using MessagesInbound
		inbound := NewMessagesInbound()
		ctx := context.Background()

		// Transform to client format
		clientBody, err := inbound.TransformResponse(ctx, resp)
		if err != nil {
			t.Fatalf("TransformResponse failed: %v", err)
		}

		// Parse the client response
		var clientResp MessageResponse
		if err := json.Unmarshal(clientBody, &clientResp); err != nil {
			t.Fatalf("failed to unmarshal client response: %v", err)
		}

		// Verify the response preserves semantic fields
		verifyResponsePreservation(t, resp, &clientResp)
	})
}


// generateValidAnthropicRequest generates a random valid Anthropic Messages request
func generateValidAnthropicRequest(t *rapid.T) *MessageRequest {
	// Generate model name
	modelName := rapid.SampledFrom([]string{
		"claude-3-opus-20240229",
		"claude-3-sonnet-20240229",
		"claude-3-haiku-20240307",
		"claude-3-5-sonnet-20241022",
	}).Draw(t, "model")

	// Generate max_tokens (required for Anthropic)
	maxTokens := int64(rapid.IntRange(1, 4096).Draw(t, "maxTokens"))

	// Generate messages (at least 1)
	numMessages := rapid.IntRange(1, 5).Draw(t, "numMessages")
	messages := make([]Message, numMessages)

	roles := []string{"user", "assistant"}
	for i := 0; i < numMessages; i++ {
		// Alternate roles, starting with user
		role := roles[i%2]
		content := rapid.StringN(1, 100, 500).Draw(t, "content")

		messages[i] = Message{
			Role: role,
			Content: MessageContent{
				Text: &content,
			},
		}
	}

	req := &MessageRequest{
		Model:     modelName,
		Messages:  messages,
		MaxTokens: maxTokens,
	}

	// Optionally add system message
	if rapid.Bool().Draw(t, "hasSystem") {
		systemText := rapid.StringN(1, 50, 200).Draw(t, "systemText")
		req.System = &SystemContent{
			Text: &systemText,
		}
	}

	// Optionally add temperature
	if rapid.Bool().Draw(t, "hasTemperature") {
		temp := rapid.Float64Range(0.0, 1.0).Draw(t, "temperature")
		req.Temperature = &temp
	}

	// Optionally add top_p
	if rapid.Bool().Draw(t, "hasTopP") {
		topP := rapid.Float64Range(0.0, 1.0).Draw(t, "topP")
		req.TopP = &topP
	}

	// Optionally add stream
	if rapid.Bool().Draw(t, "hasStream") {
		stream := rapid.Bool().Draw(t, "stream")
		req.Stream = &stream
	}

	// Optionally add stop sequences
	if rapid.Bool().Draw(t, "hasStopSequences") {
		numStops := rapid.IntRange(1, 3).Draw(t, "numStops")
		stops := make([]string, numStops)
		for i := 0; i < numStops; i++ {
			stops[i] = rapid.StringN(1, 10, 20).Draw(t, "stopSequence")
		}
		req.StopSequences = stops
	}

	// Optionally add tools
	if rapid.Bool().Draw(t, "hasTools") {
		numTools := rapid.IntRange(1, 3).Draw(t, "numTools")
		tools := make([]Tool, numTools)
		for i := 0; i < numTools; i++ {
			toolName := rapid.StringMatching(`^[a-z_][a-z0-9_]*$`).Draw(t, "toolName")
			tools[i] = Tool{
				Name:        toolName,
				Description: rapid.StringN(0, 50, 200).Draw(t, "toolDescription"),
				InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
			}
		}
		req.Tools = tools
	}

	return req
}


// generateValidInternalResponse generates a random valid internal response
func generateValidInternalResponse(t *rapid.T) *model.InternalLLMResponse {
	id := "msg_" + rapid.StringMatching(`^[a-zA-Z0-9]{24}$`).Draw(t, "id")
	modelName := rapid.SampledFrom([]string{
		"claude-3-opus-20240229",
		"claude-3-sonnet-20240229",
		"claude-3-haiku-20240307",
	}).Draw(t, "model")
	created := int64(rapid.IntRange(1600000000, 1800000000).Draw(t, "created"))

	// Generate choices
	numChoices := rapid.IntRange(1, 1).Draw(t, "numChoices") // Anthropic typically has 1 choice
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
func verifyRequestPreservation(t *rapid.T, original *MessageRequest, internal *model.InternalLLMRequest) {
	// Verify model
	if original.Model != internal.Model {
		t.Fatalf("model mismatch: expected %s, got %s", original.Model, internal.Model)
	}

	// Calculate expected message count (original messages + system if present)
	expectedMsgCount := len(original.Messages)
	if original.System != nil && original.System.GetText() != "" {
		expectedMsgCount++
	}

	// Verify messages count
	if expectedMsgCount != len(internal.Messages) {
		t.Fatalf("messages count mismatch: expected %d, got %d", expectedMsgCount, len(internal.Messages))
	}

	// Verify system message if present
	msgOffset := 0
	if original.System != nil && original.System.GetText() != "" {
		if internal.Messages[0].Role != "system" {
			t.Fatalf("expected first message to be system, got %s", internal.Messages[0].Role)
		}
		systemText := original.System.GetText()
		internalSystemText := internal.Messages[0].Content.GetText()
		if systemText != internalSystemText {
			t.Fatalf("system message content mismatch: expected %s, got %s", systemText, internalSystemText)
		}
		msgOffset = 1
	}

	// Verify each message
	for i, origMsg := range original.Messages {
		intMsg := internal.Messages[i+msgOffset]

		if origMsg.Role != intMsg.Role {
			t.Fatalf("message[%d] role mismatch: expected %s, got %s", i, origMsg.Role, intMsg.Role)
		}

		origContent := origMsg.Content.GetText()
		intContent := intMsg.Content.GetText()
		if origContent != intContent {
			t.Fatalf("message[%d] content mismatch: expected %s, got %s", i, origContent, intContent)
		}
	}

	// Verify max_tokens
	if internal.MaxTokens == nil {
		t.Fatalf("max_tokens should be set")
	}
	if original.MaxTokens != *internal.MaxTokens {
		t.Fatalf("max_tokens mismatch: expected %d, got %d", original.MaxTokens, *internal.MaxTokens)
	}

	// Verify temperature
	if (original.Temperature == nil) != (internal.Temperature == nil) {
		t.Fatalf("temperature presence mismatch")
	}
	if original.Temperature != nil && *original.Temperature != *internal.Temperature {
		t.Fatalf("temperature mismatch: expected %f, got %f", *original.Temperature, *internal.Temperature)
	}

	// Verify top_p
	if (original.TopP == nil) != (internal.TopP == nil) {
		t.Fatalf("top_p presence mismatch")
	}
	if original.TopP != nil && *original.TopP != *internal.TopP {
		t.Fatalf("top_p mismatch: expected %f, got %f", *original.TopP, *internal.TopP)
	}

	// Verify stream
	if (original.Stream == nil) != (internal.Stream == nil) {
		t.Fatalf("stream presence mismatch")
	}
	if original.Stream != nil && *original.Stream != *internal.Stream {
		t.Fatalf("stream mismatch: expected %v, got %v", *original.Stream, *internal.Stream)
	}

	// Verify stop sequences
	if len(original.StopSequences) > 0 {
		if internal.Stop == nil {
			t.Fatalf("stop sequences should be set")
		}
		if len(original.StopSequences) != len(internal.Stop.MultipleStop) {
			t.Fatalf("stop sequences count mismatch: expected %d, got %d", len(original.StopSequences), len(internal.Stop.MultipleStop))
		}
		for i, stop := range original.StopSequences {
			if stop != internal.Stop.MultipleStop[i] {
				t.Fatalf("stop sequence[%d] mismatch: expected %s, got %s", i, stop, internal.Stop.MultipleStop[i])
			}
		}
	}

	// Verify tools count
	if len(original.Tools) != len(internal.Tools) {
		t.Fatalf("tools count mismatch: expected %d, got %d", len(original.Tools), len(internal.Tools))
	}

	// Verify each tool
	for i, origTool := range original.Tools {
		intTool := internal.Tools[i]

		if intTool.Type != "function" {
			t.Fatalf("tool[%d] type mismatch: expected function, got %s", i, intTool.Type)
		}
		if origTool.Name != intTool.Function.Name {
			t.Fatalf("tool[%d] name mismatch: expected %s, got %s", i, origTool.Name, intTool.Function.Name)
		}
	}

	// Verify RawAPIFormat is set correctly
	if internal.RawAPIFormat != model.APIFormatAnthropic {
		t.Fatalf("RawAPIFormat mismatch: expected %s, got %s", model.APIFormatAnthropic, internal.RawAPIFormat)
	}
}


// verifyResponsePreservation verifies that the client response preserves semantic fields
func verifyResponsePreservation(t *rapid.T, original *model.InternalLLMResponse, client *MessageResponse) {
	// Verify ID
	if original.ID != client.ID {
		t.Fatalf("id mismatch: expected %s, got %s", original.ID, client.ID)
	}

	// Verify Type
	if client.Type != "message" {
		t.Fatalf("type mismatch: expected message, got %s", client.Type)
	}

	// Verify Role
	if client.Role != "assistant" {
		t.Fatalf("role mismatch: expected assistant, got %s", client.Role)
	}

	// Verify Model
	if original.Model != client.Model {
		t.Fatalf("model mismatch: expected %s, got %s", original.Model, client.Model)
	}

	// Verify content is preserved
	// Get text content from original
	var originalText string
	if len(original.Choices) > 0 && original.Choices[0].Message != nil {
		originalText = original.Choices[0].Message.Content.GetText()
	}

	// Get text content from client response
	var clientText string
	for _, block := range client.Content {
		if block.Type == "text" && block.Text != nil {
			clientText += *block.Text
		}
	}

	if originalText != clientText {
		t.Fatalf("content mismatch: expected %s, got %s", originalText, clientText)
	}

	// Verify stop reason is set
	if len(original.Choices) > 0 && original.Choices[0].FinishReason != nil {
		if client.StopReason == nil {
			t.Fatalf("stop_reason should be set")
		}
		// Verify stop reason mapping
		expectedStopReason := mapFinishReasonToStopReason(*original.Choices[0].FinishReason)
		if expectedStopReason != *client.StopReason {
			t.Fatalf("stop_reason mismatch: expected %s, got %s", expectedStopReason, *client.StopReason)
		}
	}

	// Verify usage
	if original.Usage != nil {
		if client.Usage == nil {
			t.Fatalf("usage should be set")
		}
		if original.Usage.PromptTokens != client.Usage.InputTokens {
			t.Fatalf("usage input_tokens mismatch: expected %d, got %d", original.Usage.PromptTokens, client.Usage.InputTokens)
		}
		if original.Usage.CompletionTokens != client.Usage.OutputTokens {
			t.Fatalf("usage output_tokens mismatch: expected %d, got %d", original.Usage.CompletionTokens, client.Usage.OutputTokens)
		}
	}
}

// mapFinishReasonToStopReason maps OpenAI finish reason to Anthropic stop reason
func mapFinishReasonToStopReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}
