package openai

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

// ChatOutbound implements the Outbound interface for OpenAI Chat format
type ChatOutbound struct{}

// NewChatOutbound creates a new ChatOutbound instance
func NewChatOutbound() *ChatOutbound {
	return &ChatOutbound{}
}

// TransformRequest converts internal request to OpenAI Chat API HTTP request
func (c *ChatOutbound) TransformRequest(ctx context.Context, request *model.InternalLLMRequest, baseUrl, key string) (*http.Request, error) {
	if request == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Build the request URL
	url := buildOpenAIChatURL(baseUrl)

	// Serialize the request body
	body, err := json.Marshal(request)
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
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", key))

	return req, nil
}

// TransformResponse converts OpenAI Chat API response to internal format
func (c *ChatOutbound) TransformResponse(ctx context.Context, response *http.Response) (*model.InternalLLMResponse, error) {
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
		return c.parseErrorResponse(response.StatusCode, body)
	}

	// Parse successful response
	var internalResp model.InternalLLMResponse
	if err := json.Unmarshal(body, &internalResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &internalResp, nil
}

// TransformStream converts OpenAI SSE stream data to internal format
func (c *ChatOutbound) TransformStream(ctx context.Context, eventData []byte) (*model.InternalLLMResponse, error) {
	// Handle [DONE] marker
	trimmed := bytes.TrimSpace(eventData)
	if bytes.Equal(trimmed, []byte("[DONE]")) {
		return nil, nil
	}

	// Parse the JSON chunk
	var chunk model.InternalLLMResponse
	if err := json.Unmarshal(trimmed, &chunk); err != nil {
		return nil, fmt.Errorf("failed to unmarshal stream chunk: %w", err)
	}

	// Set object type for streaming
	if chunk.Object == "" {
		chunk.Object = "chat.completion.chunk"
	}

	return &chunk, nil
}

// buildOpenAIChatURL constructs the full URL for OpenAI Chat API
func buildOpenAIChatURL(baseUrl string) string {
	// Remove trailing slash from base URL
	baseUrl = strings.TrimSuffix(baseUrl, "/")

	// Check if the base URL already contains the path
	if strings.HasSuffix(baseUrl, "/chat/completions") {
		return baseUrl
	}

	// Check if it ends with /v1
	if strings.HasSuffix(baseUrl, "/v1") {
		return baseUrl + "/chat/completions"
	}

	// Otherwise, append the full path
	return baseUrl + "/v1/chat/completions"
}

// parseErrorResponse parses an error response from OpenAI API
func (c *ChatOutbound) parseErrorResponse(statusCode int, body []byte) (*model.InternalLLMResponse, error) {
	// Try to parse as OpenAI error format
	var errorResp struct {
		Error model.ErrorDetail `json:"error"`
	}

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

	return &model.InternalLLMResponse{
		Object:  "error",
		Created: time.Now().Unix(),
		Error: &model.ResponseError{
			StatusCode: statusCode,
			Detail:     errorResp.Error,
		},
	}, nil
}
