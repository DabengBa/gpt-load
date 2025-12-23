package model

import (
	"encoding/json"
	"errors"
)

// APIFormat defines the supported API formats
type APIFormat string

const (
	APIFormatOpenAIChat     APIFormat = "openai_chat"
	APIFormatOpenAIResponse APIFormat = "openai_response"
	APIFormatAnthropic      APIFormat = "anthropic"
	APIFormatGemini         APIFormat = "gemini"
)

// StreamOptions represents streaming configuration
type StreamOptions struct {
	// IncludeUsage indicates whether to include usage statistics in the final chunk
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// Stop represents stop sequences (can be string or []string)
type Stop struct {
	Stop         *string
	MultipleStop []string
}

func (s Stop) MarshalJSON() ([]byte, error) {
	if s.Stop != nil {
		return json.Marshal(s.Stop)
	}
	if len(s.MultipleStop) > 0 {
		return json.Marshal(s.MultipleStop)
	}
	return []byte("null"), nil
}

func (s *Stop) UnmarshalJSON(data []byte) error {
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		s.Stop = &str
		return nil
	}

	var strs []string
	if err := json.Unmarshal(data, &strs); err == nil {
		s.MultipleStop = strs
		return nil
	}

	return errors.New("invalid stop type: expected string or []string")
}

// Tool represents a function tool definition
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`

	// CacheControl is used for provider-specific cache control (e.g., Anthropic)
	CacheControl *CacheControl `json:"-"`
}

// Function represents a function definition
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
}

// FunctionCall represents a function call
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall represents a tool call in the response
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
	Index    int          `json:"index"`

	// CacheControl is used for provider-specific cache control
	CacheControl *CacheControl `json:"-"`
}

// ToolChoice represents the tool choice parameter
type ToolChoice struct {
	ToolChoice      *string          `json:"tool_choice,omitempty"`
	NamedToolChoice *NamedToolChoice `json:"named_tool_choice,omitempty"`
}

type NamedToolChoice struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name string `json:"name"`
}

func (t ToolChoice) MarshalJSON() ([]byte, error) {
	if t.ToolChoice != nil {
		return json.Marshal(t.ToolChoice)
	}
	return json.Marshal(t.NamedToolChoice)
}

func (t *ToolChoice) UnmarshalJSON(data []byte) error {
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		t.ToolChoice = &str
		return nil
	}

	var named NamedToolChoice
	if err := json.Unmarshal(data, &named); err == nil {
		t.NamedToolChoice = &named
		return nil
	}

	return errors.New("invalid tool choice type")
}

// CacheControl represents cache control configuration (for Anthropic)
type CacheControl struct {
	Type string `json:"-"`
	TTL  string `json:"-"`
}

// ResponseFormat specifies the format of the response
type ResponseFormat struct {
	Type string `json:"type"`
}

// Usage represents token usage statistics
type Usage struct {
	PromptTokens            int64                    `json:"prompt_tokens"`
	CompletionTokens        int64                    `json:"completion_tokens"`
	TotalTokens             int64                    `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails provides breakdown of prompt tokens
type PromptTokensDetails struct {
	AudioTokens  int64 `json:"audio_tokens"`
	CachedTokens int64 `json:"cached_tokens"`
}

// CompletionTokensDetails provides breakdown of completion tokens
type CompletionTokensDetails struct {
	AudioTokens              int64 `json:"audio_tokens"`
	ReasoningTokens          int64 `json:"reasoning_tokens"`
	AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
}

// ResponseError represents an error response
type ResponseError struct {
	StatusCode int         `json:"-"`
	Detail     ErrorDetail `json:"error"`
}

func (e ResponseError) Error() string {
	msg := "request failed"
	if e.Detail.Message != "" {
		msg = e.Detail.Message
	}
	if e.Detail.Code != "" {
		msg += " (code: " + e.Detail.Code + ")"
	}
	return msg
}

// ErrorDetail represents error details
type ErrorDetail struct {
	Code      string `json:"code,omitempty"`
	Message   string `json:"message"`
	Type      string `json:"type"`
	Param     string `json:"param,omitempty"`
	RequestID string `json:"request_id,omitempty"`
}

// ImageURL represents an image URL with optional detail level
type ImageURL struct {
	URL    string  `json:"url"`
	Detail *string `json:"detail,omitempty"`
}
