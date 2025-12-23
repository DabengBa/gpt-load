package model

import (
	"encoding/json"
	"errors"
)

// InternalLLMRequest is the unified LLM request model
// Based on OpenAI Chat Completion format, extended to support other providers
type InternalLLMRequest struct {
	// Core fields
	Messages []Message `json:"messages"`
	Model    string    `json:"model"`

	// Generation parameters
	Temperature         *float64       `json:"temperature,omitempty"`
	TopP                *float64       `json:"top_p,omitempty"`
	MaxTokens           *int64         `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int64         `json:"max_completion_tokens,omitempty"`
	Stream              *bool          `json:"stream,omitempty"`
	StreamOptions       *StreamOptions `json:"stream_options,omitempty"`
	Stop                *Stop          `json:"stop,omitempty"`

	// Penalty parameters
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`

	// Tool calling
	Tools             []Tool      `json:"tools,omitempty"`
	ToolChoice        *ToolChoice `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool       `json:"parallel_tool_calls,omitempty"`

	// Reasoning/Thinking
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
	ReasoningBudget *int64 `json:"-"` // Internal field

	// Response format
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// Metadata
	Metadata map[string]string `json:"metadata,omitempty"`
	User     *string           `json:"user,omitempty"`
	Seed     *int64            `json:"seed,omitempty"`

	// Help fields (not sent to LLM service)
	RawAPIFormat        APIFormat         `json:"-"` // Original request format
	TransformerMetadata map[string]string `json:"-"` // Transformer-specific metadata
}

// Validate validates the request
func (r *InternalLLMRequest) Validate() error {
	if r.Model == "" {
		return errors.New("model is required")
	}
	if len(r.Messages) == 0 {
		return errors.New("messages are required")
	}
	return nil
}

// IsStreaming returns whether streaming is enabled
func (r *InternalLLMRequest) IsStreaming() bool {
	return r.Stream != nil && *r.Stream
}

// Message represents a message in the conversation
type Message struct {
	Role    string         `json:"role,omitempty"`
	Content MessageContent `json:"content,omitzero"`
	Name    *string        `json:"name,omitempty"`

	// Tool call related
	ToolCallID *string    `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`

	// Reasoning content (DeepSeek/Anthropic thinking)
	ReasoningContent   *string `json:"reasoning_content,omitempty"`
	ReasoningSignature *string `json:"-"` // Anthropic signature

	// Cache control (Anthropic)
	CacheControl *CacheControl `json:"-"`
}

// ClearHelpFields clears internal help fields
func (m *Message) ClearHelpFields() {
	m.ReasoningContent = nil
	m.ReasoningSignature = nil
}

// MessageContent represents message content (can be string or array of parts)
type MessageContent struct {
	Content         *string              `json:"content,omitempty"`
	MultipleContent []MessageContentPart `json:"multiple_content,omitempty"`
}

// IsEmpty returns whether the content is empty
func (c MessageContent) IsEmpty() bool {
	return c.Content == nil && len(c.MultipleContent) == 0
}

// GetText returns the text content
func (c MessageContent) GetText() string {
	if c.Content != nil {
		return *c.Content
	}
	// Concatenate text parts
	var text string
	for _, part := range c.MultipleContent {
		if part.Type == "text" && part.Text != nil {
			text += *part.Text
		}
	}
	return text
}

func (c MessageContent) MarshalJSON() ([]byte, error) {
	if len(c.MultipleContent) > 0 {
		// If only one text part, serialize as string
		if len(c.MultipleContent) == 1 && c.MultipleContent[0].Type == "text" && c.MultipleContent[0].Text != nil {
			return json.Marshal(c.MultipleContent[0].Text)
		}
		return json.Marshal(c.MultipleContent)
	}
	return json.Marshal(c.Content)
}

func (c *MessageContent) UnmarshalJSON(data []byte) error {
	// Try string first
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		c.Content = &str
		return nil
	}

	// Try array of parts
	var parts []MessageContentPart
	if err := json.Unmarshal(data, &parts); err == nil {
		c.MultipleContent = parts
		return nil
	}

	return errors.New("invalid content type: expected string or []MessageContentPart")
}

// MessageContentPart represents a part of message content
type MessageContentPart struct {
	Type     string    `json:"type"`
	Text     *string   `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`

	// Cache control (Anthropic)
	CacheControl *CacheControl `json:"-"`
}

// InternalLLMResponse is the unified LLM response model
type InternalLLMResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`

	// System fingerprint
	SystemFingerprint string `json:"system_fingerprint,omitempty"`

	// Service tier
	ServiceTier string `json:"service_tier,omitempty"`

	// Error information
	Error *ResponseError `json:"error,omitempty"`
}

// IsError returns whether the response is an error
func (r *InternalLLMResponse) IsError() bool {
	return r.Error != nil
}

// GetContent returns the content from the first choice
func (r *InternalLLMResponse) GetContent() string {
	if len(r.Choices) == 0 {
		return ""
	}
	choice := r.Choices[0]
	if choice.Message != nil {
		return choice.Message.Content.GetText()
	}
	if choice.Delta != nil {
		return choice.Delta.Content.GetText()
	}
	return ""
}

// Choice represents a choice in the response
type Choice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`      // Non-streaming
	Delta        *Message `json:"delta,omitempty"`        // Streaming
	FinishReason *string  `json:"finish_reason,omitempty"`
}
