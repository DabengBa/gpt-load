package gemini

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

// MessagesOutbound implements the Outbound interface for Gemini format
type MessagesOutbound struct{}

// NewMessagesOutbound creates a new MessagesOutbound instance
func NewMessagesOutbound() *MessagesOutbound {
	return &MessagesOutbound{}
}

// TransformRequest converts internal request to Gemini API HTTP request
func (m *MessagesOutbound) TransformRequest(ctx context.Context, request *model.InternalLLMRequest, baseUrl, key string) (*http.Request, error) {
	if request == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Build the request URL with API key as query parameter
	url := buildGeminiURL(baseUrl, request.Model, key, request.IsStreaming())

	// Convert internal request to Gemini format
	geminiReq, err := m.convertToGeminiRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	// Serialize the request body
	body, err := json.Marshal(geminiReq)
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

	return req, nil
}


// convertToGeminiRequest converts internal request to Gemini GenerateContentRequest
func (m *MessagesOutbound) convertToGeminiRequest(req *model.InternalLLMRequest) (*GenerateContentRequest, error) {
	geminiReq := &GenerateContentRequest{
		Contents: make([]Content, 0),
	}

	// Convert messages
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			// Handle system message - Gemini uses separate systemInstruction field
			systemContent, err := m.convertMessageContent(msg)
			if err != nil {
				return nil, fmt.Errorf("failed to convert system message: %w", err)
			}
			geminiReq.SystemInstruction = systemContent
			continue
		}

		content, err := m.convertMessage(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message: %w", err)
		}
		geminiReq.Contents = append(geminiReq.Contents, *content)
	}

	// Set generation config
	geminiReq.GenerationConfig = m.buildGenerationConfig(req)

	// Convert tools
	if len(req.Tools) > 0 {
		geminiReq.Tools = m.convertTools(req.Tools)
	}

	// Convert tool choice
	if req.ToolChoice != nil {
		geminiReq.ToolConfig = m.convertToolChoice(req.ToolChoice)
	}

	return geminiReq, nil
}

// convertMessage converts an internal message to Gemini Content
func (m *MessagesOutbound) convertMessage(msg model.Message) (*Content, error) {
	content := &Content{
		Role:  convertRole(msg.Role),
		Parts: make([]Part, 0),
	}

	// Handle tool result message
	if msg.ToolCallID != nil {
		text := msg.Content.GetText()
		responseJSON, _ := json.Marshal(map[string]string{"result": text})
		content.Parts = append(content.Parts, Part{
			FunctionResponse: &FunctionResponse{
				Name:     *msg.ToolCallID, // Use tool call ID as function name
				Response: responseJSON,
			},
		})
		return content, nil
	}

	// Handle assistant message with tool calls
	if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
		// Add text content if present
		text := msg.Content.GetText()
		if text != "" {
			content.Parts = append(content.Parts, Part{Text: &text})
		}

		// Add function calls
		for _, tc := range msg.ToolCalls {
			args := json.RawMessage(tc.Function.Arguments)
			content.Parts = append(content.Parts, Part{
				FunctionCall: &FunctionCall{
					Name: tc.Function.Name,
					Args: args,
				},
			})
		}
		return content, nil
	}

	// Handle simple text content
	if msg.Content.Content != nil {
		content.Parts = append(content.Parts, Part{Text: msg.Content.Content})
		return content, nil
	}

	// Handle multiple content parts
	if len(msg.Content.MultipleContent) > 0 {
		for _, part := range msg.Content.MultipleContent {
			geminiPart, err := m.convertContentPart(part)
			if err != nil {
				return nil, err
			}
			content.Parts = append(content.Parts, *geminiPart)
		}
		return content, nil
	}

	// Empty content - use empty text
	emptyText := ""
	content.Parts = append(content.Parts, Part{Text: &emptyText})
	return content, nil
}

// convertMessageContent converts message content to Gemini Content (for system instruction)
func (m *MessagesOutbound) convertMessageContent(msg model.Message) (*Content, error) {
	content := &Content{
		Parts: make([]Part, 0),
	}

	if msg.Content.Content != nil {
		content.Parts = append(content.Parts, Part{Text: msg.Content.Content})
		return content, nil
	}

	if len(msg.Content.MultipleContent) > 0 {
		for _, part := range msg.Content.MultipleContent {
			if part.Type == "text" && part.Text != nil {
				content.Parts = append(content.Parts, Part{Text: part.Text})
			}
		}
	}

	return content, nil
}


// convertContentPart converts an internal content part to Gemini Part
func (m *MessagesOutbound) convertContentPart(part model.MessageContentPart) (*Part, error) {
	switch part.Type {
	case "text":
		return &Part{Text: part.Text}, nil

	case "image_url":
		if part.ImageURL == nil {
			return nil, fmt.Errorf("image_url part missing ImageURL")
		}
		// Parse data URL to extract media type and base64 data
		inlineData, err := parseDataURL(part.ImageURL.URL)
		if err != nil {
			return nil, fmt.Errorf("failed to parse image URL: %w", err)
		}
		return &Part{InlineData: inlineData}, nil

	default:
		return nil, fmt.Errorf("unsupported content part type: %s", part.Type)
	}
}

// parseDataURL parses a data URL and returns InlineData
func parseDataURL(url string) (*InlineData, error) {
	// Check if it's a data URL
	if !strings.HasPrefix(url, "data:") {
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
	mimeType := "application/octet-stream"
	isBase64 := false

	for i, part := range parts {
		if i == 0 && part != "" {
			mimeType = part
		} else if part == "base64" {
			isBase64 = true
		}
	}

	if !isBase64 {
		return nil, fmt.Errorf("only base64 encoded data URLs are supported")
	}

	return &InlineData{
		MimeType: mimeType,
		Data:     data,
	}, nil
}

// buildGenerationConfig builds the generation config from internal request
func (m *MessagesOutbound) buildGenerationConfig(req *model.InternalLLMRequest) *GenerationConfig {
	config := &GenerationConfig{}
	hasConfig := false

	if req.Temperature != nil {
		config.Temperature = req.Temperature
		hasConfig = true
	}

	if req.TopP != nil {
		config.TopP = req.TopP
		hasConfig = true
	}

	if req.MaxTokens != nil {
		config.MaxOutputTokens = req.MaxTokens
		hasConfig = true
	} else if req.MaxCompletionTokens != nil {
		config.MaxOutputTokens = req.MaxCompletionTokens
		hasConfig = true
	}

	if req.Stop != nil {
		if req.Stop.Stop != nil {
			config.StopSequences = []string{*req.Stop.Stop}
			hasConfig = true
		} else if len(req.Stop.MultipleStop) > 0 {
			config.StopSequences = req.Stop.MultipleStop
			hasConfig = true
		}
	}

	if !hasConfig {
		return nil
	}

	return config
}

// convertTools converts internal tools to Gemini format
func (m *MessagesOutbound) convertTools(tools []model.Tool) []Tool {
	declarations := make([]FunctionDeclaration, len(tools))
	for i, tool := range tools {
		declarations[i] = FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		}
	}
	return []Tool{{FunctionDeclarations: declarations}}
}

// convertToolChoice converts internal tool choice to Gemini format
func (m *MessagesOutbound) convertToolChoice(tc *model.ToolChoice) *ToolConfig {
	if tc == nil {
		return nil
	}

	config := &ToolConfig{
		FunctionCallingConfig: &FunctionCallingConfig{},
	}

	if tc.ToolChoice != nil {
		switch *tc.ToolChoice {
		case "auto":
			config.FunctionCallingConfig.Mode = FunctionCallingModeAuto
		case "required":
			config.FunctionCallingConfig.Mode = FunctionCallingModeAny
		case "none":
			config.FunctionCallingConfig.Mode = FunctionCallingModeNone
		}
	}

	if tc.NamedToolChoice != nil {
		config.FunctionCallingConfig.Mode = FunctionCallingModeAny
		config.FunctionCallingConfig.AllowedFunctionNames = []string{tc.NamedToolChoice.Function.Name}
	}

	return config
}

// convertRole converts OpenAI role to Gemini role
func convertRole(role string) string {
	switch role {
	case "assistant":
		return "model"
	case "user":
		return "user"
	default:
		return role
	}
}


// buildGeminiURL constructs the full URL for Gemini API
func buildGeminiURL(baseUrl, modelName, apiKey string, streaming bool) string {
	// Remove trailing slash from base URL
	baseUrl = strings.TrimSuffix(baseUrl, "/")

	// Determine the action based on streaming
	action := "generateContent"
	if streaming {
		action = "streamGenerateContent"
	}

	// Check if the base URL already contains the model path
	if strings.Contains(baseUrl, "/models/") {
		// URL already has model, just add action and key
		if strings.Contains(baseUrl, ":generateContent") || strings.Contains(baseUrl, ":streamGenerateContent") {
			return fmt.Sprintf("%s?key=%s", baseUrl, apiKey)
		}
		return fmt.Sprintf("%s:%s?key=%s", baseUrl, action, apiKey)
	}

	// Check if it ends with /v1beta or /v1
	if strings.HasSuffix(baseUrl, "/v1beta") || strings.HasSuffix(baseUrl, "/v1") {
		return fmt.Sprintf("%s/models/%s:%s?key=%s", baseUrl, modelName, action, apiKey)
	}

	// Otherwise, append the full path
	return fmt.Sprintf("%s/v1beta/models/%s:%s?key=%s", baseUrl, modelName, action, apiKey)
}

// TransformResponse converts Gemini API response to internal format
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
	var geminiResp GenerateContentResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return m.convertToInternalResponse(&geminiResp), nil
}

// convertToInternalResponse converts Gemini response to internal format
func (m *MessagesOutbound) convertToInternalResponse(resp *GenerateContentResponse) *model.InternalLLMResponse {
	internalResp := &model.InternalLLMResponse{
		ID:      fmt.Sprintf("gemini-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Choices: make([]model.Choice, 0),
	}

	// Convert candidates to choices
	for i, candidate := range resp.Candidates {
		choice := model.Choice{
			Index: i,
		}

		if candidate.Content != nil {
			msg := m.convertContentToMessage(candidate.Content)
			choice.Message = msg
		}

		// Convert finish reason
		if candidate.FinishReason != "" {
			fr := convertFinishReason(candidate.FinishReason)
			choice.FinishReason = &fr
		}

		internalResp.Choices = append(internalResp.Choices, choice)
	}

	// Convert usage
	if resp.UsageMetadata != nil {
		internalResp.Usage = &model.Usage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		}
	}

	return internalResp
}

// convertContentToMessage converts Gemini Content to internal Message
func (m *MessagesOutbound) convertContentToMessage(content *Content) *model.Message {
	msg := &model.Message{
		Role: "assistant",
	}

	var textContent string
	var toolCalls []model.ToolCall

	for i, part := range content.Parts {
		if part.Text != nil {
			textContent += *part.Text
		}
		if part.FunctionCall != nil {
			var args string
			if part.FunctionCall.Args != nil {
				args = string(part.FunctionCall.Args)
			}
			toolCalls = append(toolCalls, model.ToolCall{
				ID:   fmt.Sprintf("call_%d", i),
				Type: "function",
				Function: model.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: args,
				},
				Index: len(toolCalls),
			})
		}
	}

	msg.Content = model.MessageContent{
		Content: &textContent,
	}
	msg.ToolCalls = toolCalls

	return msg
}

// convertFinishReason converts Gemini finish reason to OpenAI format
func convertFinishReason(reason string) string {
	switch reason {
	case FinishReasonStop:
		return "stop"
	case FinishReasonMaxTokens:
		return "length"
	case FinishReasonSafety, FinishReasonRecitation, FinishReasonBlocklist, FinishReasonProhibitedContent, FinishReasonSpii:
		return "content_filter"
	default:
		return "stop"
	}
}


// parseErrorResponse parses an error response from Gemini API
func (m *MessagesOutbound) parseErrorResponse(statusCode int, body []byte) (*model.InternalLLMResponse, error) {
	// Try to parse as Gemini error format
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

	// Map Gemini error status to OpenAI error type
	errorType := mapGeminiErrorStatus(errorResp.Error.Status)

	return &model.InternalLLMResponse{
		Object:  "error",
		Created: time.Now().Unix(),
		Error: &model.ResponseError{
			StatusCode: statusCode,
			Detail: model.ErrorDetail{
				Message: errorResp.Error.Message,
				Type:    errorType,
				Code:    errorResp.Error.Status,
			},
		},
	}, nil
}

// mapGeminiErrorStatus maps Gemini error status to OpenAI error types
func mapGeminiErrorStatus(status string) string {
	switch status {
	case "INVALID_ARGUMENT":
		return "invalid_request_error"
	case "PERMISSION_DENIED", "UNAUTHENTICATED":
		return "invalid_api_key"
	case "RESOURCE_EXHAUSTED":
		return "rate_limit_exceeded"
	case "NOT_FOUND":
		return "invalid_request_error"
	case "INTERNAL", "UNAVAILABLE":
		return "server_error"
	default:
		return "api_error"
	}
}

// TransformStream converts Gemini streaming data to internal format
func (m *MessagesOutbound) TransformStream(ctx context.Context, eventData []byte) (*model.InternalLLMResponse, error) {
	// Gemini streaming returns JSON objects directly (not SSE format)
	// Each line is a complete JSON response

	trimmed := bytes.TrimSpace(eventData)
	if len(trimmed) == 0 {
		return nil, nil
	}

	// Handle array wrapper that Gemini sometimes uses
	if bytes.HasPrefix(trimmed, []byte("[")) {
		// Remove leading [ or ,
		trimmed = bytes.TrimLeft(trimmed, "[,")
		trimmed = bytes.TrimSpace(trimmed)
	}
	if bytes.HasSuffix(trimmed, []byte("]")) {
		trimmed = bytes.TrimRight(trimmed, "]")
		trimmed = bytes.TrimSpace(trimmed)
	}

	if len(trimmed) == 0 {
		return nil, nil
	}

	// Parse the JSON chunk
	var chunk GenerateContentResponse
	if err := json.Unmarshal(trimmed, &chunk); err != nil {
		// Try to parse as error
		var errorResp ErrorResponse
		if json.Unmarshal(trimmed, &errorResp) == nil && errorResp.Error.Message != "" {
			return &model.InternalLLMResponse{
				Object:  "error",
				Created: time.Now().Unix(),
				Error: &model.ResponseError{
					StatusCode: errorResp.Error.Code,
					Detail: model.ErrorDetail{
						Message: errorResp.Error.Message,
						Type:    mapGeminiErrorStatus(errorResp.Error.Status),
					},
				},
			}, nil
		}
		return nil, nil // Ignore parse errors for stream chunks
	}

	return m.convertStreamChunk(&chunk), nil
}

// convertStreamChunk converts a Gemini stream chunk to internal format
func (m *MessagesOutbound) convertStreamChunk(chunk *GenerateContentResponse) *model.InternalLLMResponse {
	resp := &model.InternalLLMResponse{
		ID:      fmt.Sprintf("gemini-%d", time.Now().UnixNano()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Choices: make([]model.Choice, 0),
	}

	for i, candidate := range chunk.Candidates {
		choice := model.Choice{
			Index: i,
			Delta: &model.Message{},
		}

		if candidate.Content != nil {
			for j, part := range candidate.Content.Parts {
				if part.Text != nil {
					choice.Delta.Content = model.MessageContent{
						Content: part.Text,
					}
				}
				if part.FunctionCall != nil {
					var args string
					if part.FunctionCall.Args != nil {
						args = string(part.FunctionCall.Args)
					}
					choice.Delta.ToolCalls = append(choice.Delta.ToolCalls, model.ToolCall{
						ID:   fmt.Sprintf("call_%d", j),
						Type: "function",
						Function: model.FunctionCall{
							Name:      part.FunctionCall.Name,
							Arguments: args,
						},
						Index: j,
					})
				}
			}
		}

		// Convert finish reason
		if candidate.FinishReason != "" {
			fr := convertFinishReason(candidate.FinishReason)
			choice.FinishReason = &fr
		}

		resp.Choices = append(resp.Choices, choice)
	}

	// Convert usage
	if chunk.UsageMetadata != nil {
		resp.Usage = &model.Usage{
			PromptTokens:     chunk.UsageMetadata.PromptTokenCount,
			CompletionTokens: chunk.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      chunk.UsageMetadata.TotalTokenCount,
		}
	}

	return resp
}
