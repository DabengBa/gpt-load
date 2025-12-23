package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"gpt-load/internal/transformer/model"
)

const (
	// AnthropicVersion is the API version header value
	AnthropicVersion = "2023-06-01"
	// DefaultMessagesPath is the default path for Anthropic Messages API
	DefaultMessagesPath = "/v1/messages"
)

// MessagesOutbound implements the Outbound interface for Anthropic Messages format
type MessagesOutbound struct{}

// NewMessagesOutbound creates a new MessagesOutbound instance
func NewMessagesOutbound() *MessagesOutbound {
	return &MessagesOutbound{}
}

// TransformRequest converts internal request to Anthropic Messages API HTTP request
func (m *MessagesOutbound) TransformRequest(ctx context.Context, request *model.InternalLLMRequest, baseUrl, key string) (*http.Request, error) {
	if request == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Build the request URL
	url := buildAnthropicMessagesURL(baseUrl)

	// Convert internal request to Anthropic format
	anthropicReq, err := m.convertToAnthropicRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	// Serialize the request body
	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", key)
	req.Header.Set("anthropic-version", AnthropicVersion)

	return req, nil
}


// convertToAnthropicRequest converts internal request to Anthropic MessageRequest
func (m *MessagesOutbound) convertToAnthropicRequest(req *model.InternalLLMRequest) (*MessageRequest, error) {
	anthropicReq := &MessageRequest{
		Model:       req.Model,
		Messages:    make([]Message, 0),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}

	// Set max_tokens (required for Anthropic)
	if req.MaxTokens != nil {
		anthropicReq.MaxTokens = *req.MaxTokens
	} else if req.MaxCompletionTokens != nil {
		anthropicReq.MaxTokens = *req.MaxCompletionTokens
	} else {
		// Default max_tokens if not specified
		anthropicReq.MaxTokens = 4096
	}

	// Convert messages, extracting system message
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			// Handle system message - Anthropic uses separate system field
			systemText := msg.Content.GetText()
			if systemText != "" {
				anthropicReq.System = &SystemContent{
					Text: &systemText,
				}
			}
			continue
		}

		anthropicMsg, err := m.convertMessage(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message: %w", err)
		}
		anthropicReq.Messages = append(anthropicReq.Messages, *anthropicMsg)
	}

	// Convert stop sequences
	if req.Stop != nil {
		if req.Stop.Stop != nil {
			anthropicReq.StopSequences = []string{*req.Stop.Stop}
		} else if len(req.Stop.MultipleStop) > 0 {
			anthropicReq.StopSequences = req.Stop.MultipleStop
		}
	}

	// Convert tools
	if len(req.Tools) > 0 {
		anthropicReq.Tools = make([]Tool, len(req.Tools))
		for i, tool := range req.Tools {
			anthropicReq.Tools[i] = Tool{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				InputSchema: tool.Function.Parameters,
			}
		}
	}

	// Convert tool choice
	if req.ToolChoice != nil {
		anthropicReq.ToolChoice = m.convertToolChoice(req.ToolChoice)
	}

	// Convert user metadata
	if req.User != nil {
		anthropicReq.Metadata = &Metadata{
			UserID: *req.User,
		}
	}

	return anthropicReq, nil
}

// convertMessage converts an internal message to Anthropic format
func (m *MessagesOutbound) convertMessage(msg model.Message) (*Message, error) {
	anthropicMsg := &Message{
		Role: msg.Role,
	}

	// Handle tool result message
	if msg.ToolCallID != nil {
		content := msg.Content.GetText()
		anthropicMsg.Content = MessageContent{
			Blocks: []ContentBlock{
				{
					Type:      "tool_result",
					ToolUseID: msg.ToolCallID,
					Content:   &content,
				},
			},
		}
		return anthropicMsg, nil
	}

	// Handle assistant message with tool calls
	if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
		blocks := make([]ContentBlock, 0)

		// Add text content if present
		text := msg.Content.GetText()
		if text != "" {
			blocks = append(blocks, ContentBlock{
				Type: "text",
				Text: &text,
			})
		}

		// Add tool_use blocks
		for _, tc := range msg.ToolCalls {
			input := json.RawMessage(tc.Function.Arguments)
			blocks = append(blocks, ContentBlock{
				Type:  "tool_use",
				ID:    &tc.ID,
				Name:  &tc.Function.Name,
				Input: &input,
			})
		}

		anthropicMsg.Content = MessageContent{
			Blocks: blocks,
		}
		return anthropicMsg, nil
	}

	// Handle simple text content
	if msg.Content.Content != nil {
		anthropicMsg.Content = MessageContent{
			Text: msg.Content.Content,
		}
		return anthropicMsg, nil
	}

	// Handle multiple content parts
	if len(msg.Content.MultipleContent) > 0 {
		blocks := make([]ContentBlock, 0, len(msg.Content.MultipleContent))
		for _, part := range msg.Content.MultipleContent {
			block, err := m.convertContentPart(part)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, *block)
		}
		anthropicMsg.Content = MessageContent{
			Blocks: blocks,
		}
		return anthropicMsg, nil
	}

	// Empty content - use empty text
	emptyText := ""
	anthropicMsg.Content = MessageContent{
		Text: &emptyText,
	}
	return anthropicMsg, nil
}


// convertContentPart converts an internal content part to Anthropic ContentBlock
func (m *MessagesOutbound) convertContentPart(part model.MessageContentPart) (*ContentBlock, error) {
	switch part.Type {
	case "text":
		return &ContentBlock{
			Type: "text",
			Text: part.Text,
		}, nil

	case "image_url":
		if part.ImageURL == nil {
			return nil, fmt.Errorf("image_url part missing ImageURL")
		}
		// Parse data URL to extract media type and base64 data
		source, err := parseDataURL(part.ImageURL.URL)
		if err != nil {
			return nil, fmt.Errorf("failed to parse image URL: %w", err)
		}
		return &ContentBlock{
			Type:   "image",
			Source: source,
		}, nil

	default:
		return nil, fmt.Errorf("unsupported content part type: %s", part.Type)
	}
}

// parseDataURL parses a data URL and returns an ImageSource
func parseDataURL(url string) (*ImageSource, error) {
	// Check if it's a data URL
	if !strings.HasPrefix(url, "data:") {
		// For regular URLs, we can't convert to base64 here
		// Return an error or handle differently based on requirements
		return nil, fmt.Errorf("only data URLs are supported for image conversion")
	}

	// Parse data URL format: data:[<mediatype>][;base64],<data>
	url = strings.TrimPrefix(url, "data:")

	// Find the comma separator
	commaIdx := strings.Index(url, ",")
	if commaIdx == -1 {
		return nil, fmt.Errorf("invalid data URL format")
	}

	metadata := url[:commaIdx]
	data := url[commaIdx+1:]

	// Parse media type and encoding
	parts := strings.Split(metadata, ";")
	mediaType := "application/octet-stream"
	isBase64 := false

	for i, part := range parts {
		if i == 0 && part != "" {
			mediaType = part
		} else if part == "base64" {
			isBase64 = true
		}
	}

	if !isBase64 {
		return nil, fmt.Errorf("only base64 encoded data URLs are supported")
	}

	return &ImageSource{
		Type:      "base64",
		MediaType: mediaType,
		Data:      data,
	}, nil
}

// convertToolChoice converts internal tool choice to Anthropic format
func (m *MessagesOutbound) convertToolChoice(tc *model.ToolChoice) *ToolChoice {
	if tc == nil {
		return nil
	}

	if tc.ToolChoice != nil {
		switch *tc.ToolChoice {
		case "auto":
			return &ToolChoice{Type: "auto"}
		case "required":
			return &ToolChoice{Type: "any"}
		case "none":
			return nil
		}
	}

	if tc.NamedToolChoice != nil {
		return &ToolChoice{
			Type: "tool",
			Name: &tc.NamedToolChoice.Function.Name,
		}
	}

	return nil
}

// buildAnthropicMessagesURL constructs the full URL for Anthropic Messages API
func buildAnthropicMessagesURL(baseUrl string) string {
	// Remove trailing slash from base URL
	baseUrl = strings.TrimSuffix(baseUrl, "/")

	// Check if the base URL already contains the path
	if strings.HasSuffix(baseUrl, "/messages") {
		return baseUrl
	}

	// Check if it ends with /v1
	if strings.HasSuffix(baseUrl, "/v1") {
		return baseUrl + "/messages"
	}

	// Otherwise, append the full path
	return baseUrl + "/v1/messages"
}


// TransformResponse converts Anthropic Messages API response to internal format
func (m *MessagesOutbound) TransformResponse(ctx context.Context, response *http.Response) (*model.InternalLLMResponse, error) {
	if response == nil {
		return nil, fmt.Errorf("response cannot be nil")
	}

	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check for error response
	if response.StatusCode >= 400 {
		return m.parseErrorResponse(response.StatusCode, body)
	}

	// Parse successful response
	var anthropicResp MessageResponse
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return m.convertToInternalResponse(&anthropicResp), nil
}

// convertToInternalResponse converts Anthropic response to internal format
func (m *MessagesOutbound) convertToInternalResponse(resp *MessageResponse) *model.InternalLLMResponse {
	internalResp := &model.InternalLLMResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resp.Model,
		Choices: make([]model.Choice, 1),
	}

	// Build the message from content blocks
	msg := &model.Message{
		Role: "assistant",
	}

	var textContent string
	var toolCalls []model.ToolCall
	var reasoningContent string

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text != nil {
				textContent += *block.Text
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
		case "thinking":
			if block.Thinking != nil {
				reasoningContent += *block.Thinking
			}
		}
	}

	msg.Content = model.MessageContent{
		Content: &textContent,
	}
	msg.ToolCalls = toolCalls
	if reasoningContent != "" {
		msg.ReasoningContent = &reasoningContent
	}

	// Convert stop reason to finish reason
	var finishReason *string
	if resp.StopReason != nil {
		fr := m.convertStopReason(*resp.StopReason)
		finishReason = &fr
	}

	internalResp.Choices[0] = model.Choice{
		Index:        0,
		Message:      msg,
		FinishReason: finishReason,
	}

	// Convert usage
	if resp.Usage != nil {
		internalResp.Usage = &model.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}
	}

	return internalResp
}

// convertStopReason converts Anthropic stop reason to OpenAI finish reason
func (m *MessagesOutbound) convertStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "stop_sequence":
		return "stop"
	default:
		return "stop"
	}
}

// parseErrorResponse parses an error response from Anthropic API
func (m *MessagesOutbound) parseErrorResponse(statusCode int, body []byte) (*model.InternalLLMResponse, error) {
	// Try to parse as Anthropic error format
	var errorResp ErrorResponse
	if err := json.Unmarshal(body, &errorResp); err != nil {
		// If parsing fails, create a generic error
		return &model.InternalLLMResponse{
			Object:  "error",
			Created: time.Now().Unix(),
			Error: &model.ResponseError{
				StatusCode: statusCode,
				Detail: model.ErrorDetail{
					Message: string(body),
					Type:    "api_error",
				},
			},
		}, nil
	}

	// Map Anthropic error type to OpenAI error type
	errorType := mapAnthropicErrorType(errorResp.Error.Type)

	return &model.InternalLLMResponse{
		Object:  "error",
		Created: time.Now().Unix(),
		Error: &model.ResponseError{
			StatusCode: statusCode,
			Detail: model.ErrorDetail{
				Message: errorResp.Error.Message,
				Type:    errorType,
			},
		},
	}, nil
}

// mapAnthropicErrorType maps Anthropic error types to OpenAI error types
func mapAnthropicErrorType(anthropicType string) string {
	switch anthropicType {
	case "invalid_api_key":
		return "invalid_api_key"
	case "rate_limit_error":
		return "rate_limit_exceeded"
	case "overloaded_error":
		return "server_error"
	case "invalid_request_error":
		return "invalid_request_error"
	case "authentication_error":
		return "invalid_api_key"
	case "permission_error":
		return "invalid_api_key"
	case "not_found_error":
		return "invalid_request_error"
	default:
		return "api_error"
	}
}


// TransformStream converts Anthropic SSE stream data to internal format
func (m *MessagesOutbound) TransformStream(ctx context.Context, eventData []byte) (*model.InternalLLMResponse, error) {
	// Parse the event line to extract event type and data
	eventType, data := parseSSEEvent(eventData)

	// Handle different event types
	switch eventType {
	case EventMessageStart:
		return m.handleMessageStart(data)
	case EventContentBlockStart:
		return m.handleContentBlockStart(data)
	case EventContentBlockDelta:
		return m.handleContentBlockDelta(data)
	case EventContentBlockStop:
		// No action needed for content block stop
		return nil, nil
	case EventMessageDelta:
		return m.handleMessageDelta(data)
	case EventMessageStop:
		// No action needed for message stop
		return nil, nil
	case EventPing:
		// No action needed for ping
		return nil, nil
	case EventError:
		return m.handleError(data)
	default:
		// Try to parse as raw JSON (for cases where event type is not in the data)
		return m.parseRawStreamData(eventData)
	}
}

// parseSSEEvent parses SSE event data to extract event type and JSON data
func parseSSEEvent(eventData []byte) (string, []byte) {
	lines := bytes.Split(eventData, []byte("\n"))

	var eventType string
	var data []byte

	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if bytes.HasPrefix(line, []byte("event:")) {
			eventType = string(bytes.TrimSpace(bytes.TrimPrefix(line, []byte("event:"))))
		} else if bytes.HasPrefix(line, []byte("data:")) {
			data = bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		}
	}

	// If no event type found, try to detect from data
	if eventType == "" && len(data) > 0 {
		var probe struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(data, &probe) == nil && probe.Type != "" {
			eventType = probe.Type
		}
	}

	// If still no data, use the raw event data
	if len(data) == 0 {
		data = bytes.TrimSpace(eventData)
	}

	return eventType, data
}

// handleMessageStart handles message_start event
func (m *MessagesOutbound) handleMessageStart(data []byte) (*model.InternalLLMResponse, error) {
	var event struct {
		Type    string          `json:"type"`
		Message MessageResponse `json:"message"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, nil // Ignore parse errors for stream events
	}

	return &model.InternalLLMResponse{
		ID:      event.Message.ID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   event.Message.Model,
		Choices: []model.Choice{
			{
				Index: 0,
				Delta: &model.Message{
					Role: "assistant",
				},
			},
		},
	}, nil
}

// handleContentBlockStart handles content_block_start event
func (m *MessagesOutbound) handleContentBlockStart(data []byte) (*model.InternalLLMResponse, error) {
	var event struct {
		Type         string       `json:"type"`
		Index        int          `json:"index"`
		ContentBlock ContentBlock `json:"content_block"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, nil
	}

	// For tool_use blocks, emit the tool call start
	if event.ContentBlock.Type == "tool_use" && event.ContentBlock.ID != nil && event.ContentBlock.Name != nil {
		return &model.InternalLLMResponse{
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Choices: []model.Choice{
				{
					Index: 0,
					Delta: &model.Message{
						ToolCalls: []model.ToolCall{
							{
								ID:    *event.ContentBlock.ID,
								Type:  "function",
								Index: event.Index,
								Function: model.FunctionCall{
									Name:      *event.ContentBlock.Name,
									Arguments: "",
								},
							},
						},
					},
				},
			},
		}, nil
	}

	return nil, nil
}

// handleContentBlockDelta handles content_block_delta event
func (m *MessagesOutbound) handleContentBlockDelta(data []byte) (*model.InternalLLMResponse, error) {
	var event struct {
		Type  string            `json:"type"`
		Index int               `json:"index"`
		Delta ContentBlockDelta `json:"delta"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, nil
	}

	resp := &model.InternalLLMResponse{
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Choices: []model.Choice{
			{
				Index: 0,
				Delta: &model.Message{},
			},
		},
	}

	switch event.Delta.Type {
	case "text_delta":
		if event.Delta.Text != nil {
			resp.Choices[0].Delta.Content = model.MessageContent{
				Content: event.Delta.Text,
			}
		}
	case "input_json_delta":
		if event.Delta.PartialJSON != nil {
			resp.Choices[0].Delta.ToolCalls = []model.ToolCall{
				{
					Index: event.Index,
					Function: model.FunctionCall{
						Arguments: *event.Delta.PartialJSON,
					},
				},
			}
		}
	case "thinking_delta":
		if event.Delta.Thinking != nil {
			resp.Choices[0].Delta.ReasoningContent = event.Delta.Thinking
		}
	}

	return resp, nil
}

// handleMessageDelta handles message_delta event
func (m *MessagesOutbound) handleMessageDelta(data []byte) (*model.InternalLLMResponse, error) {
	var event struct {
		Type  string `json:"type"`
		Delta struct {
			StopReason   *string `json:"stop_reason,omitempty"`
			StopSequence *string `json:"stop_sequence,omitempty"`
		} `json:"delta"`
		Usage *Usage `json:"usage,omitempty"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, nil
	}

	resp := &model.InternalLLMResponse{
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Choices: []model.Choice{
			{
				Index: 0,
				Delta: &model.Message{},
			},
		},
	}

	// Convert stop reason to finish reason
	if event.Delta.StopReason != nil {
		fr := m.convertStopReason(*event.Delta.StopReason)
		resp.Choices[0].FinishReason = &fr
	}

	// Convert usage
	if event.Usage != nil {
		resp.Usage = &model.Usage{
			PromptTokens:     event.Usage.InputTokens,
			CompletionTokens: event.Usage.OutputTokens,
			TotalTokens:      event.Usage.InputTokens + event.Usage.OutputTokens,
		}
	}

	return resp, nil
}

// handleError handles error event
func (m *MessagesOutbound) handleError(data []byte) (*model.InternalLLMResponse, error) {
	var event struct {
		Type  string      `json:"type"`
		Error ErrorDetail `json:"error"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, nil
	}

	return &model.InternalLLMResponse{
		Object:  "error",
		Created: time.Now().Unix(),
		Error: &model.ResponseError{
			StatusCode: 500,
			Detail: model.ErrorDetail{
				Message: event.Error.Message,
				Type:    mapAnthropicErrorType(event.Error.Type),
			},
		},
	}, nil
}

// parseRawStreamData attempts to parse raw stream data
func (m *MessagesOutbound) parseRawStreamData(data []byte) (*model.InternalLLMResponse, error) {
	// Try to parse as a generic event with type field
	var event struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, nil
	}

	// Route to appropriate handler based on type
	switch event.Type {
	case EventMessageStart:
		return m.handleMessageStart(data)
	case EventContentBlockStart:
		return m.handleContentBlockStart(data)
	case EventContentBlockDelta:
		return m.handleContentBlockDelta(data)
	case EventMessageDelta:
		return m.handleMessageDelta(data)
	case EventError:
		return m.handleError(data)
	}

	return nil, nil
}
