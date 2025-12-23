package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"gpt-load/internal/transformer/model"
)

// MessagesInbound implements the Inbound interface for Anthropic Messages format
type MessagesInbound struct {
	mu sync.Mutex

	// Accumulated response for streaming
	accumulatedResponse *model.InternalLLMResponse

	// Accumulated content for each choice (for streaming)
	accumulatedContent map[int]string

	// Accumulated tool calls for each choice (for streaming)
	accumulatedToolCalls map[int][]model.ToolCall

	// Accumulated thinking content for each choice
	accumulatedThinking map[int]string

	// Current content block index being processed
	currentBlockIndex int

	// Current content block type
	currentBlockType string

	// Current tool call being built
	currentToolCall *model.ToolCall
}

// NewMessagesInbound creates a new MessagesInbound instance
func NewMessagesInbound() *MessagesInbound {
	return &MessagesInbound{
		accumulatedContent:   make(map[int]string),
		accumulatedToolCalls: make(map[int][]model.ToolCall),
		accumulatedThinking:  make(map[int]string),
	}
}


// TransformRequest converts Anthropic Messages format request to internal format
func (m *MessagesInbound) TransformRequest(ctx context.Context, body []byte) (*model.InternalLLMRequest, error) {
	var req MessageRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic request: %w", err)
	}

	// Build internal request
	internalReq := &model.InternalLLMRequest{
		Model:        req.Model,
		Messages:     make([]model.Message, 0),
		MaxTokens:    &req.MaxTokens,
		Temperature:  req.Temperature,
		TopP:         req.TopP,
		Stream:       req.Stream,
		RawAPIFormat: model.APIFormatAnthropic,
	}

	// Handle system message
	if req.System != nil {
		systemText := req.System.GetText()
		if systemText != "" {
			internalReq.Messages = append(internalReq.Messages, model.Message{
				Role: "system",
				Content: model.MessageContent{
					Content: &systemText,
				},
			})
		}
	}

	// Convert messages
	for _, msg := range req.Messages {
		internalMsg, err := m.convertMessage(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message: %w", err)
		}
		internalReq.Messages = append(internalReq.Messages, *internalMsg)
	}

	// Convert stop sequences
	if len(req.StopSequences) > 0 {
		internalReq.Stop = &model.Stop{
			MultipleStop: req.StopSequences,
		}
	}

	// Convert tools
	if len(req.Tools) > 0 {
		internalReq.Tools = make([]model.Tool, len(req.Tools))
		for i, tool := range req.Tools {
			internalReq.Tools[i] = model.Tool{
				Type: "function",
				Function: model.Function{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  tool.InputSchema,
				},
			}
		}
	}

	// Convert tool choice
	if req.ToolChoice != nil {
		internalReq.ToolChoice = m.convertToolChoice(req.ToolChoice)
	}

	// Convert metadata
	if req.Metadata != nil && req.Metadata.UserID != "" {
		internalReq.User = &req.Metadata.UserID
	}

	// Validate
	if err := internalReq.Validate(); err != nil {
		return nil, err
	}

	return internalReq, nil
}


// convertMessage converts an Anthropic message to internal format
func (m *MessagesInbound) convertMessage(msg Message) (*model.Message, error) {
	internalMsg := &model.Message{
		Role: msg.Role,
	}

	// Handle simple text content
	if msg.Content.Text != nil {
		internalMsg.Content = model.MessageContent{
			Content: msg.Content.Text,
		}
		return internalMsg, nil
	}

	// Handle content blocks
	if len(msg.Content.Blocks) > 0 {
		var textParts []model.MessageContentPart
		var toolCalls []model.ToolCall
		var reasoningContent string

		for _, block := range msg.Content.Blocks {
			switch block.Type {
			case "text":
				if block.Text != nil {
					textParts = append(textParts, model.MessageContentPart{
						Type: "text",
						Text: block.Text,
					})
				}

			case "image":
				if block.Source != nil {
					// Convert Anthropic image source to OpenAI image_url format
					dataURL := fmt.Sprintf("data:%s;base64,%s", block.Source.MediaType, block.Source.Data)
					textParts = append(textParts, model.MessageContentPart{
						Type: "image_url",
						ImageURL: &model.ImageURL{
							URL: dataURL,
						},
					})
				}

			case "tool_use":
				if block.ID != nil && block.Name != nil {
					var args string
					if block.Input != nil {
						args = string(*block.Input)
					}
					toolCalls = append(toolCalls, model.ToolCall{
						ID:   *block.ID,
						Type: "function",
						Function: model.FunctionCall{
							Name:      *block.Name,
							Arguments: args,
						},
						Index: len(toolCalls),
					})
				}

			case "tool_result":
				if block.ToolUseID != nil {
					internalMsg.ToolCallID = block.ToolUseID
					if block.Content != nil {
						internalMsg.Content = model.MessageContent{
							Content: block.Content,
						}
					}
				}

			case "thinking":
				if block.Thinking != nil {
					reasoningContent += *block.Thinking
				}
			}
		}

		// Set content
		if len(textParts) == 1 && textParts[0].Type == "text" {
			internalMsg.Content = model.MessageContent{
				Content: textParts[0].Text,
			}
		} else if len(textParts) > 0 {
			internalMsg.Content = model.MessageContent{
				MultipleContent: textParts,
			}
		}

		// Set tool calls
		if len(toolCalls) > 0 {
			internalMsg.ToolCalls = toolCalls
		}

		// Set reasoning content
		if reasoningContent != "" {
			internalMsg.ReasoningContent = &reasoningContent
		}
	}

	return internalMsg, nil
}

// convertToolChoice converts Anthropic tool choice to internal format
func (m *MessagesInbound) convertToolChoice(tc *ToolChoice) *model.ToolChoice {
	if tc == nil {
		return nil
	}

	switch tc.Type {
	case "auto":
		auto := "auto"
		return &model.ToolChoice{ToolChoice: &auto}
	case "any":
		required := "required"
		return &model.ToolChoice{ToolChoice: &required}
	case "tool":
		if tc.Name != nil {
			return &model.ToolChoice{
				NamedToolChoice: &model.NamedToolChoice{
					Type: "function",
					Function: model.ToolFunction{
						Name: *tc.Name,
					},
				},
			}
		}
	}

	return nil
}


// TransformResponse converts internal response to Anthropic Messages format
func (m *MessagesInbound) TransformResponse(ctx context.Context, response *model.InternalLLMResponse) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Store the response for GetInternalResponse
	m.accumulatedResponse = response

	// Handle error response
	if response.Error != nil {
		errResp := ErrorResponse{
			Type: "error",
			Error: ErrorDetail{
				Type:    response.Error.Detail.Type,
				Message: response.Error.Detail.Message,
			},
		}
		return json.Marshal(errResp)
	}

	// Build Anthropic response
	anthropicResp := MessageResponse{
		ID:    response.ID,
		Type:  "message",
		Role:  "assistant",
		Model: response.Model,
	}

	// Convert choices to content blocks
	var contentBlocks []ContentBlock
	for _, choice := range response.Choices {
		if choice.Message != nil {
			// Add thinking block if present
			if choice.Message.ReasoningContent != nil && *choice.Message.ReasoningContent != "" {
				contentBlocks = append(contentBlocks, ContentBlock{
					Type:     "thinking",
					Thinking: choice.Message.ReasoningContent,
				})
			}

			// Add text content
			text := choice.Message.Content.GetText()
			if text != "" {
				contentBlocks = append(contentBlocks, ContentBlock{
					Type: "text",
					Text: &text,
				})
			}

			// Add tool calls as tool_use blocks
			for _, tc := range choice.Message.ToolCalls {
				input := json.RawMessage(tc.Function.Arguments)
				contentBlocks = append(contentBlocks, ContentBlock{
					Type:  "tool_use",
					ID:    &tc.ID,
					Name:  &tc.Function.Name,
					Input: &input,
				})
			}

			// Set stop reason
			if choice.FinishReason != nil {
				stopReason := m.convertFinishReason(*choice.FinishReason)
				anthropicResp.StopReason = &stopReason
			}
		}
	}

	anthropicResp.Content = contentBlocks

	// Convert usage
	if response.Usage != nil {
		anthropicResp.Usage = &Usage{
			InputTokens:  response.Usage.PromptTokens,
			OutputTokens: response.Usage.CompletionTokens,
		}
	}

	return json.Marshal(anthropicResp)
}

// convertFinishReason converts OpenAI finish reason to Anthropic stop reason
func (m *MessagesInbound) convertFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "end_turn"
	default:
		return "end_turn"
	}
}


// TransformStream converts internal streaming response to Anthropic SSE format
func (m *MessagesInbound) TransformStream(ctx context.Context, stream *model.InternalLLMResponse) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Accumulate the streaming chunks
	m.accumulateStreamChunk(stream)

	// Build Anthropic SSE events
	var events []byte

	for _, choice := range stream.Choices {
		if choice.Delta != nil {
			// Handle text content
			text := choice.Delta.Content.GetText()
			if text != "" {
				event := StreamEvent{
					Type:  EventContentBlockDelta,
					Index: &choice.Index,
					Delta: &ContentBlockDelta{
						Type: "text_delta",
						Text: &text,
					},
				}
				eventData, err := json.Marshal(event)
				if err != nil {
					return nil, err
				}
				events = append(events, []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", EventContentBlockDelta, eventData))...)
			}

			// Handle thinking content
			if choice.Delta.ReasoningContent != nil && *choice.Delta.ReasoningContent != "" {
				event := StreamEvent{
					Type:  EventContentBlockDelta,
					Index: &choice.Index,
					Delta: &ContentBlockDelta{
						Type:     "thinking_delta",
						Thinking: choice.Delta.ReasoningContent,
					},
				}
				eventData, err := json.Marshal(event)
				if err != nil {
					return nil, err
				}
				events = append(events, []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", EventContentBlockDelta, eventData))...)
			}

			// Handle tool calls
			for _, tc := range choice.Delta.ToolCalls {
				if tc.Function.Arguments != "" {
					event := StreamEvent{
						Type:  EventContentBlockDelta,
						Index: &tc.Index,
						Delta: &ContentBlockDelta{
							Type:        "input_json_delta",
							PartialJSON: &tc.Function.Arguments,
						},
					}
					eventData, err := json.Marshal(event)
					if err != nil {
						return nil, err
					}
					events = append(events, []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", EventContentBlockDelta, eventData))...)
				}
			}
		}

		// Handle finish reason
		if choice.FinishReason != nil {
			stopReason := m.convertFinishReason(*choice.FinishReason)
			event := StreamEvent{
				Type: EventMessageDelta,
				MessageDelta: &MessageDelta{
					StopReason: &stopReason,
				},
			}
			if stream.Usage != nil {
				event.Usage = &Usage{
					InputTokens:  stream.Usage.PromptTokens,
					OutputTokens: stream.Usage.CompletionTokens,
				}
			}
			eventData, err := json.Marshal(event)
			if err != nil {
				return nil, err
			}
			events = append(events, []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", EventMessageDelta, eventData))...)
		}
	}

	return events, nil
}


// accumulateStreamChunk accumulates streaming chunks for later aggregation
func (m *MessagesInbound) accumulateStreamChunk(chunk *model.InternalLLMResponse) {
	// Initialize accumulated response if needed
	if m.accumulatedResponse == nil {
		m.accumulatedResponse = &model.InternalLLMResponse{
			ID:      chunk.ID,
			Object:  "chat.completion",
			Created: chunk.Created,
			Model:   chunk.Model,
			Choices: make([]model.Choice, 0),
		}
	}

	// Update model if provided
	if chunk.Model != "" {
		m.accumulatedResponse.Model = chunk.Model
	}

	// Accumulate usage if provided
	if chunk.Usage != nil {
		m.accumulatedResponse.Usage = chunk.Usage
	}

	// Accumulate content from each choice
	for _, choice := range chunk.Choices {
		if choice.Delta != nil {
			// Accumulate text content
			content := choice.Delta.Content.GetText()
			if content != "" {
				m.accumulatedContent[choice.Index] += content
			}

			// Accumulate thinking content
			if choice.Delta.ReasoningContent != nil {
				m.accumulatedThinking[choice.Index] += *choice.Delta.ReasoningContent
			}

			// Accumulate tool calls
			if len(choice.Delta.ToolCalls) > 0 {
				m.accumulateToolCalls(choice.Index, choice.Delta.ToolCalls)
			}
		}

		// Track finish reason
		if choice.FinishReason != nil {
			m.ensureChoiceExists(choice.Index)
			m.accumulatedResponse.Choices[choice.Index].FinishReason = choice.FinishReason
		}
	}
}

// accumulateToolCalls accumulates tool calls from streaming chunks
func (m *MessagesInbound) accumulateToolCalls(choiceIndex int, toolCalls []model.ToolCall) {
	if m.accumulatedToolCalls == nil {
		m.accumulatedToolCalls = make(map[int][]model.ToolCall)
	}

	existing := m.accumulatedToolCalls[choiceIndex]

	for _, tc := range toolCalls {
		// Find existing tool call by index
		found := false
		for i := range existing {
			if existing[i].Index == tc.Index {
				// Append arguments
				existing[i].Function.Arguments += tc.Function.Arguments
				if tc.Function.Name != "" {
					existing[i].Function.Name = tc.Function.Name
				}
				if tc.ID != "" {
					existing[i].ID = tc.ID
				}
				if tc.Type != "" {
					existing[i].Type = tc.Type
				}
				found = true
				break
			}
		}

		if !found {
			// Add new tool call
			existing = append(existing, tc)
		}
	}

	m.accumulatedToolCalls[choiceIndex] = existing
}

// ensureChoiceExists ensures a choice exists at the given index
func (m *MessagesInbound) ensureChoiceExists(index int) {
	for len(m.accumulatedResponse.Choices) <= index {
		m.accumulatedResponse.Choices = append(m.accumulatedResponse.Choices, model.Choice{
			Index: len(m.accumulatedResponse.Choices),
		})
	}
}


// GetInternalResponse returns the aggregated complete response
func (m *MessagesInbound) GetInternalResponse(ctx context.Context) (*model.InternalLLMResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.accumulatedResponse == nil {
		// Return empty response if nothing accumulated
		return &model.InternalLLMResponse{
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Choices: []model.Choice{},
		}, nil
	}

	// Build the final response with accumulated content
	result := &model.InternalLLMResponse{
		ID:      m.accumulatedResponse.ID,
		Object:  "chat.completion",
		Created: m.accumulatedResponse.Created,
		Model:   m.accumulatedResponse.Model,
		Usage:   m.accumulatedResponse.Usage,
		Error:   m.accumulatedResponse.Error,
		Choices: make([]model.Choice, 0),
	}

	// Build choices with accumulated content
	maxIndex := -1
	for idx := range m.accumulatedContent {
		if idx > maxIndex {
			maxIndex = idx
		}
	}
	for idx := range m.accumulatedToolCalls {
		if idx > maxIndex {
			maxIndex = idx
		}
	}
	for idx := range m.accumulatedThinking {
		if idx > maxIndex {
			maxIndex = idx
		}
	}
	for i := range m.accumulatedResponse.Choices {
		if i > maxIndex {
			maxIndex = i
		}
	}

	for i := 0; i <= maxIndex; i++ {
		content := m.accumulatedContent[i]
		toolCalls := m.accumulatedToolCalls[i]
		thinking := m.accumulatedThinking[i]

		var finishReason *string
		if i < len(m.accumulatedResponse.Choices) {
			finishReason = m.accumulatedResponse.Choices[i].FinishReason
		}

		msg := &model.Message{
			Role: "assistant",
			Content: model.MessageContent{
				Content: &content,
			},
			ToolCalls: toolCalls,
		}

		if thinking != "" {
			msg.ReasoningContent = &thinking
		}

		choice := model.Choice{
			Index:        i,
			Message:      msg,
			FinishReason: finishReason,
		}

		result.Choices = append(result.Choices, choice)
	}

	// If no choices were accumulated but we have stored choices, use them
	if len(result.Choices) == 0 && len(m.accumulatedResponse.Choices) > 0 {
		result.Choices = m.accumulatedResponse.Choices
	}

	return result, nil
}
