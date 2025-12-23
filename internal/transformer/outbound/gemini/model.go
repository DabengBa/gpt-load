package gemini

import "encoding/json"

// GenerateContentRequest represents a Gemini generateContent API request
type GenerateContentRequest struct {
	Contents          []Content          `json:"contents"`
	SystemInstruction *Content           `json:"systemInstruction,omitempty"`
	Tools             []Tool             `json:"tools,omitempty"`
	ToolConfig        *ToolConfig        `json:"toolConfig,omitempty"`
	GenerationConfig  *GenerationConfig  `json:"generationConfig,omitempty"`
	SafetySettings    []SafetySetting    `json:"safetySettings,omitempty"`
}

// Content represents a content block in Gemini format
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

// Part represents a part of content in Gemini format
type Part struct {
	// Text content
	Text *string `json:"text,omitempty"`

	// Inline data (for images)
	InlineData *InlineData `json:"inlineData,omitempty"`

	// Function call (in response)
	FunctionCall *FunctionCall `json:"functionCall,omitempty"`

	// Function response (in request)
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
}

// InlineData represents inline binary data (e.g., images)
type InlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// FunctionCall represents a function call in Gemini format
type FunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

// FunctionResponse represents a function response in Gemini format
type FunctionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}


// Tool represents a tool definition in Gemini format
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// FunctionDeclaration represents a function declaration in Gemini format
type FunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolConfig represents tool configuration
type ToolConfig struct {
	FunctionCallingConfig *FunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// FunctionCallingConfig represents function calling configuration
type FunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// GenerationConfig represents generation configuration
type GenerationConfig struct {
	Temperature       *float64 `json:"temperature,omitempty"`
	TopP              *float64 `json:"topP,omitempty"`
	TopK              *int64   `json:"topK,omitempty"`
	MaxOutputTokens   *int64   `json:"maxOutputTokens,omitempty"`
	StopSequences     []string `json:"stopSequences,omitempty"`
	CandidateCount    *int64   `json:"candidateCount,omitempty"`
	ResponseMimeType  string   `json:"responseMimeType,omitempty"`
	ResponseSchema    any      `json:"responseSchema,omitempty"`
}

// SafetySetting represents a safety setting
type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// GenerateContentResponse represents a Gemini generateContent API response
type GenerateContentResponse struct {
	Candidates     []Candidate     `json:"candidates,omitempty"`
	PromptFeedback *PromptFeedback `json:"promptFeedback,omitempty"`
	UsageMetadata  *UsageMetadata  `json:"usageMetadata,omitempty"`
}

// Candidate represents a response candidate
type Candidate struct {
	Content       *Content        `json:"content,omitempty"`
	FinishReason  string          `json:"finishReason,omitempty"`
	Index         int             `json:"index"`
	SafetyRatings []SafetyRating  `json:"safetyRatings,omitempty"`
}

// PromptFeedback represents feedback about the prompt
type PromptFeedback struct {
	BlockReason   string         `json:"blockReason,omitempty"`
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
}

// SafetyRating represents a safety rating
type SafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
	Blocked     bool   `json:"blocked,omitempty"`
}

// UsageMetadata represents token usage metadata
type UsageMetadata struct {
	PromptTokenCount     int64 `json:"promptTokenCount"`
	CandidatesTokenCount int64 `json:"candidatesTokenCount"`
	TotalTokenCount      int64 `json:"totalTokenCount"`
}

// ErrorResponse represents a Gemini error response
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail represents error details
type ErrorDetail struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// Finish reason constants
const (
	FinishReasonStop          = "STOP"
	FinishReasonMaxTokens     = "MAX_TOKENS"
	FinishReasonSafety        = "SAFETY"
	FinishReasonRecitation    = "RECITATION"
	FinishReasonOther         = "OTHER"
	FinishReasonBlocklist     = "BLOCKLIST"
	FinishReasonProhibitedContent = "PROHIBITED_CONTENT"
	FinishReasonSpii          = "SPII"
)

// Function calling mode constants
const (
	FunctionCallingModeAuto = "AUTO"
	FunctionCallingModeAny  = "ANY"
	FunctionCallingModeNone = "NONE"
)
