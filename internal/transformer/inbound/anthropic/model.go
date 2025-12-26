package anthropic

import (
	"encoding/json"
	"errors"
)

// MessageRequest represents an Anthropic Messages API request
type MessageRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	MaxTokens int64     `json:"max_tokens"`

	// Optional fields
	System      *SystemContent `json:"system,omitempty"`
	Temperature *float64       `json:"temperature,omitempty"`
	TopP        *float64       `json:"top_p,omitempty"`
	TopK        *int64         `json:"top_k,omitempty"`
	Stream      *bool          `json:"stream,omitempty"`
	StopSequences []string     `json:"stop_sequences,omitempty"`

	// Tool calling
	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	// Metadata
	Metadata *Metadata `json:"metadata,omitempty"`
}

// SystemContent represents the system field which can be string or array of content blocks
type SystemContent struct {
	Text   *string        `json:"-"`
	Blocks []ContentBlock `json:"-"`
}

func (s SystemContent) MarshalJSON() ([]byte, error) {
	if s.Text != nil {
		return json.Marshal(s.Text)
	}
	if len(s.Blocks) > 0 {
		return json.Marshal(s.Blocks)
	}
	return []byte("null"), nil
}

func (s *SystemContent) UnmarshalJSON(data []byte) error {
	// Try string first
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		s.Text = &str
		return nil
	}

	// Try array of content blocks
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		s.Blocks = blocks
		return nil
	}

	return errors.New("invalid system content: expected string or []ContentBlock")
}


// GetText returns the system text content
func (s *SystemContent) GetText() string {
	if s == nil {
		return ""
	}
	if s.Text != nil {
		return *s.Text
	}
	// Concatenate text from blocks
	var text string
	for _, block := range s.Blocks {
		if block.Type == "text" && block.Text != nil {
			text += *block.Text
		}
	}
	return text
}

// Message represents a message in the Anthropic format
type Message struct {
	Role    string         `json:"role"`
	Content MessageContent `json:"-"`
}

// MarshalJSON implements custom JSON marshaling for Message
func (m Message) MarshalJSON() ([]byte, error) {
	contentBytes, err := m.Content.MarshalJSON()
	if err != nil {
		return nil, err
	}

	type Alias Message
	return json.Marshal(&struct {
		Alias
		Content json.RawMessage `json:"content"`
	}{
		Alias:   Alias(m),
		Content: contentBytes,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling for Message
func (m *Message) UnmarshalJSON(data []byte) error {
	type Alias Message
	aux := &struct {
		*Alias
		Content json.RawMessage `json:"content"`
	}{
		Alias: (*Alias)(m),
	}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	return m.Content.UnmarshalJSON(aux.Content)
}

// MessageContent represents message content (can be string or array of content blocks)
type MessageContent struct {
	Text   *string        `json:"-"`
	Blocks []ContentBlock `json:"-"`
	raw    json.RawMessage
}

func (c MessageContent) MarshalJSON() ([]byte, error) {
	if c.Text != nil {
		return json.Marshal(c.Text)
	}
	if len(c.Blocks) > 0 {
		return json.Marshal(c.Blocks)
	}
	return []byte("null"), nil
}

func (c *MessageContent) UnmarshalJSON(data []byte) error {
	c.raw = data

	// Try string first
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		c.Text = &str
		return nil
	}

	// Try array of content blocks
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		c.Blocks = blocks
		return nil
	}

	return errors.New("invalid content: expected string or []ContentBlock")
}

// GetText returns the text content
func (c *MessageContent) GetText() string {
	if c == nil {
		return ""
	}
	if c.Text != nil {
		return *c.Text
	}
	// Concatenate text from blocks
	var text string
	for _, block := range c.Blocks {
		if block.Type == "text" && block.Text != nil {
			text += *block.Text
		}
	}
	return text
}


// ContentBlock represents a content block in Anthropic format
type ContentBlock struct {
	Type string `json:"type"`

	// For text blocks
	Text *string `json:"text,omitempty"`

	// For image blocks
	Source *ImageSource `json:"source,omitempty"`

	// For tool_use blocks
	ID    *string          `json:"id,omitempty"`
	Name  *string          `json:"name,omitempty"`
	Input *json.RawMessage `json:"input,omitempty"`

	// For tool_result blocks
	ToolUseID *string `json:"tool_use_id,omitempty"`
	Content   *string `json:"content,omitempty"`
	IsError   *bool   `json:"is_error,omitempty"`

	// For thinking blocks (extended thinking)
	Thinking  *string `json:"thinking,omitempty"`
	Signature *string `json:"signature,omitempty"`

	// Cache control
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// ImageSource represents an image source in Anthropic format
type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// CacheControl represents cache control configuration
type CacheControl struct {
	Type string `json:"type"`
}

// Tool represents a tool definition in Anthropic format
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`

	// Cache control
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// ToolChoice represents tool choice configuration
type ToolChoice struct {
	Type                   string  `json:"type"`
	Name                   *string `json:"name,omitempty"`
	DisableParallelToolUse *bool   `json:"disable_parallel_tool_use,omitempty"`
}

// Metadata represents request metadata
type Metadata struct {
	UserID string `json:"user_id,omitempty"`
}


// MessageResponse represents an Anthropic Messages API response
type MessageResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []ContentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   *string        `json:"stop_reason,omitempty"`
	StopSequence *string        `json:"stop_sequence,omitempty"`
	Usage        *Usage         `json:"usage,omitempty"`
}

// Usage represents token usage in Anthropic format
type Usage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens,omitempty"`
}

// ErrorResponse represents an Anthropic error response
type ErrorResponse struct {
	Type  string      `json:"type"`
	Error ErrorDetail `json:"error"`
}

// ErrorDetail represents error details
type ErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Stream event types
const (
	EventMessageStart      = "message_start"
	EventContentBlockStart = "content_block_start"
	EventContentBlockDelta = "content_block_delta"
	EventContentBlockStop  = "content_block_stop"
	EventMessageDelta      = "message_delta"
	EventMessageStop       = "message_stop"
	EventPing              = "ping"
	EventError             = "error"
)

// StreamEvent represents a streaming event
type StreamEvent struct {
	Type string `json:"type"`

	// For message_start
	Message *MessageResponse `json:"message,omitempty"`

	// For content_block_start
	Index        *int          `json:"index,omitempty"`
	ContentBlock *ContentBlock `json:"content_block,omitempty"`

	// For content_block_delta
	Delta *ContentBlockDelta `json:"delta,omitempty"`

	// For message_delta
	MessageDelta *MessageDelta `json:"message_delta,omitempty"`
	Usage        *Usage        `json:"usage,omitempty"`

	// For error
	Error *ErrorDetail `json:"error,omitempty"`
}

// ContentBlockDelta represents a delta in content block
type ContentBlockDelta struct {
	Type string `json:"type"`

	// For text_delta
	Text *string `json:"text,omitempty"`

	// For input_json_delta (tool use)
	PartialJSON *string `json:"partial_json,omitempty"`

	// For thinking_delta
	Thinking *string `json:"thinking,omitempty"`
}

// MessageDelta represents a delta in message
type MessageDelta struct {
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}
