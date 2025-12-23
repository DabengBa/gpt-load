package openai

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"gpt-load/internal/transformer/model"
)

// ChatInbound implements the Inbound interface for OpenAI Chat format
type ChatInbound struct {
	mu sync.Mutex

	// Accumulated response for streaming
	accumulatedResponse *model.InternalLLMResponse

	// Accumulated content for each choice (for streaming)
	accumulatedContent map[int]string

	// Accumulated tool calls for each choice (for streaming)
	accumulatedToolCalls map[int][]model.ToolCall
}

// NewChatInbound creates a new ChatInbound instance
func NewChatInbound() *ChatInbound {
	return &ChatInbound{
		accumulatedContent:   make(map[int]string),
		accumulatedToolCalls: make(map[int][]model.ToolCall),
	}
}

// TransformRequest converts OpenAI Chat format request to internal format
func (c *ChatInbound) TransformRequest(ctx context.Context, body []byte) (*model.InternalLLMRequest, error) {
	var req model.InternalLLMRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	// Mark the original API format
	req.RawAPIFormat = model.APIFormatOpenAIChat

	// Validate the request
	if err := req.Validate(); err != nil {
		return nil, err
	}

	return &req, nil
}

// TransformResponse converts internal response to OpenAI Chat format
func (c *ChatInbound) TransformResponse(ctx context.Context, response *model.InternalLLMResponse) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Store the response for GetInternalResponse
	c.accumulatedResponse = response

	// The internal format is already OpenAI Chat compatible
	return json.Marshal(response)
}

// TransformStream converts internal streaming response to OpenAI SSE format
func (c *ChatInbound) TransformStream(ctx context.Context, stream *model.InternalLLMResponse) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Accumulate the streaming chunks
	c.accumulateStreamChunk(stream)

	// The internal format is already OpenAI Chat compatible
	// Just serialize and add SSE prefix
	data, err := json.Marshal(stream)
	if err != nil {
		return nil, err
	}

	// Return with SSE format: "data: {json}\n\n"
	return append([]byte("data: "), append(data, []byte("\n\n")...)...), nil
}

// accumulateStreamChunk accumulates streaming chunks for later aggregation
func (c *ChatInbound) accumulateStreamChunk(chunk *model.InternalLLMResponse) {
	// Initialize accumulated response if needed
	if c.accumulatedResponse == nil {
		c.accumulatedResponse = &model.InternalLLMResponse{
			ID:                chunk.ID,
			Object:            "chat.completion",
			Created:           chunk.Created,
			Model:             chunk.Model,
			SystemFingerprint: chunk.SystemFingerprint,
			ServiceTier:       chunk.ServiceTier,
			Choices:           make([]model.Choice, 0),
		}
	}

	// Update model if provided
	if chunk.Model != "" {
		c.accumulatedResponse.Model = chunk.Model
	}

	// Accumulate usage if provided
	if chunk.Usage != nil {
		c.accumulatedResponse.Usage = chunk.Usage
	}

	// Accumulate content from each choice
	for _, choice := range chunk.Choices {
		if choice.Delta != nil {
			// Accumulate text content
			content := choice.Delta.Content.GetText()
			if content != "" {
				c.accumulatedContent[choice.Index] += content
			}

			// Accumulate tool calls
			if len(choice.Delta.ToolCalls) > 0 {
				c.accumulateToolCalls(choice.Index, choice.Delta.ToolCalls)
			}
		}

		// Track finish reason
		if choice.FinishReason != nil {
			c.ensureChoiceExists(choice.Index)
			c.accumulatedResponse.Choices[choice.Index].FinishReason = choice.FinishReason
		}
	}
}

// accumulateToolCalls accumulates tool calls from streaming chunks
func (c *ChatInbound) accumulateToolCalls(choiceIndex int, toolCalls []model.ToolCall) {
	if c.accumulatedToolCalls == nil {
		c.accumulatedToolCalls = make(map[int][]model.ToolCall)
	}

	existing := c.accumulatedToolCalls[choiceIndex]

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

	c.accumulatedToolCalls[choiceIndex] = existing
}

// ensureChoiceExists ensures a choice exists at the given index
func (c *ChatInbound) ensureChoiceExists(index int) {
	for len(c.accumulatedResponse.Choices) <= index {
		c.accumulatedResponse.Choices = append(c.accumulatedResponse.Choices, model.Choice{
			Index: len(c.accumulatedResponse.Choices),
		})
	}
}

// GetInternalResponse returns the aggregated complete response
func (c *ChatInbound) GetInternalResponse(ctx context.Context) (*model.InternalLLMResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.accumulatedResponse == nil {
		// Return empty response if nothing accumulated
		return &model.InternalLLMResponse{
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Choices: []model.Choice{},
		}, nil
	}

	// Build the final response with accumulated content
	result := &model.InternalLLMResponse{
		ID:                c.accumulatedResponse.ID,
		Object:            "chat.completion",
		Created:           c.accumulatedResponse.Created,
		Model:             c.accumulatedResponse.Model,
		SystemFingerprint: c.accumulatedResponse.SystemFingerprint,
		ServiceTier:       c.accumulatedResponse.ServiceTier,
		Usage:             c.accumulatedResponse.Usage,
		Error:             c.accumulatedResponse.Error,
		Choices:           make([]model.Choice, 0),
	}

	// Build choices with accumulated content
	maxIndex := -1
	for idx := range c.accumulatedContent {
		if idx > maxIndex {
			maxIndex = idx
		}
	}
	for idx := range c.accumulatedToolCalls {
		if idx > maxIndex {
			maxIndex = idx
		}
	}
	for i, choice := range c.accumulatedResponse.Choices {
		if i > maxIndex {
			maxIndex = i
		}
		_ = choice
	}

	for i := 0; i <= maxIndex; i++ {
		content := c.accumulatedContent[i]
		toolCalls := c.accumulatedToolCalls[i]

		var finishReason *string
		if i < len(c.accumulatedResponse.Choices) {
			finishReason = c.accumulatedResponse.Choices[i].FinishReason
		}

		choice := model.Choice{
			Index: i,
			Message: &model.Message{
				Role: "assistant",
				Content: model.MessageContent{
					Content: &content,
				},
				ToolCalls: toolCalls,
			},
			FinishReason: finishReason,
		}

		result.Choices = append(result.Choices, choice)
	}

	// If no choices were accumulated but we have stored choices, use them
	if len(result.Choices) == 0 && len(c.accumulatedResponse.Choices) > 0 {
		result.Choices = c.accumulatedResponse.Choices
	}

	return result, nil
}
