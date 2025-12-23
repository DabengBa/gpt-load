package model

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test MessageContent JSON serialization/deserialization

func TestMessageContent_UnmarshalJSON_String(t *testing.T) {
	jsonStr := `"Hello, world!"`
	var content MessageContent
	err := json.Unmarshal([]byte(jsonStr), &content)

	require.NoError(t, err)
	assert.NotNil(t, content.Content)
	assert.Equal(t, "Hello, world!", *content.Content)
	assert.Empty(t, content.MultipleContent)
}

func TestMessageContent_UnmarshalJSON_Array(t *testing.T) {
	jsonStr := `[{"type":"text","text":"Hello"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]`
	var content MessageContent
	err := json.Unmarshal([]byte(jsonStr), &content)

	require.NoError(t, err)
	assert.Nil(t, content.Content)
	assert.Len(t, content.MultipleContent, 2)
	assert.Equal(t, "text", content.MultipleContent[0].Type)
	assert.Equal(t, "Hello", *content.MultipleContent[0].Text)
	assert.Equal(t, "image_url", content.MultipleContent[1].Type)
	assert.Equal(t, "https://example.com/image.png", content.MultipleContent[1].ImageURL.URL)
}

func TestMessageContent_MarshalJSON_String(t *testing.T) {
	text := "Hello, world!"
	content := MessageContent{Content: &text}
	data, err := json.Marshal(content)

	require.NoError(t, err)
	assert.Equal(t, `"Hello, world!"`, string(data))
}

func TestMessageContent_MarshalJSON_SingleTextPart(t *testing.T) {
	// Single text part should be serialized as string
	text := "Hello"
	content := MessageContent{
		MultipleContent: []MessageContentPart{
			{Type: "text", Text: &text},
		},
	}
	data, err := json.Marshal(content)

	require.NoError(t, err)
	assert.Equal(t, `"Hello"`, string(data))
}

func TestMessageContent_MarshalJSON_MultipleParts(t *testing.T) {
	text := "Hello"
	content := MessageContent{
		MultipleContent: []MessageContentPart{
			{Type: "text", Text: &text},
			{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.com/image.png"}},
		},
	}
	data, err := json.Marshal(content)

	require.NoError(t, err)
	var result []map[string]interface{}
	err = json.Unmarshal(data, &result)
	require.NoError(t, err)
	assert.Len(t, result, 2)
}

func TestMessageContent_GetText(t *testing.T) {
	// Test with string content
	text := "Hello"
	content := MessageContent{Content: &text}
	assert.Equal(t, "Hello", content.GetText())

	// Test with multiple parts
	text1 := "Hello "
	text2 := "World"
	content2 := MessageContent{
		MultipleContent: []MessageContentPart{
			{Type: "text", Text: &text1},
			{Type: "text", Text: &text2},
		},
	}
	assert.Equal(t, "Hello World", content2.GetText())
}

func TestMessageContent_IsEmpty(t *testing.T) {
	// Empty content
	content := MessageContent{}
	assert.True(t, content.IsEmpty())

	// With string content
	text := "Hello"
	content2 := MessageContent{Content: &text}
	assert.False(t, content2.IsEmpty())

	// With multiple parts
	content3 := MessageContent{
		MultipleContent: []MessageContentPart{
			{Type: "text", Text: &text},
		},
	}
	assert.False(t, content3.IsEmpty())
}

// Test Stop JSON serialization/deserialization

func TestStop_UnmarshalJSON_String(t *testing.T) {
	jsonStr := `"stop"`
	var stop Stop
	err := json.Unmarshal([]byte(jsonStr), &stop)

	require.NoError(t, err)
	assert.NotNil(t, stop.Stop)
	assert.Equal(t, "stop", *stop.Stop)
	assert.Empty(t, stop.MultipleStop)
}

func TestStop_UnmarshalJSON_Array(t *testing.T) {
	jsonStr := `["stop1", "stop2"]`
	var stop Stop
	err := json.Unmarshal([]byte(jsonStr), &stop)

	require.NoError(t, err)
	assert.Nil(t, stop.Stop)
	assert.Equal(t, []string{"stop1", "stop2"}, stop.MultipleStop)
}

func TestStop_MarshalJSON_String(t *testing.T) {
	stopStr := "stop"
	stop := Stop{Stop: &stopStr}
	data, err := json.Marshal(stop)

	require.NoError(t, err)
	assert.Equal(t, `"stop"`, string(data))
}

func TestStop_MarshalJSON_Array(t *testing.T) {
	stop := Stop{MultipleStop: []string{"stop1", "stop2"}}
	data, err := json.Marshal(stop)

	require.NoError(t, err)
	assert.Equal(t, `["stop1","stop2"]`, string(data))
}

// Test ToolChoice JSON serialization/deserialization

func TestToolChoice_UnmarshalJSON_String(t *testing.T) {
	jsonStr := `"auto"`
	var tc ToolChoice
	err := json.Unmarshal([]byte(jsonStr), &tc)

	require.NoError(t, err)
	assert.NotNil(t, tc.ToolChoice)
	assert.Equal(t, "auto", *tc.ToolChoice)
	assert.Nil(t, tc.NamedToolChoice)
}

func TestToolChoice_UnmarshalJSON_Object(t *testing.T) {
	jsonStr := `{"type":"function","function":{"name":"get_weather"}}`
	var tc ToolChoice
	err := json.Unmarshal([]byte(jsonStr), &tc)

	require.NoError(t, err)
	assert.Nil(t, tc.ToolChoice)
	assert.NotNil(t, tc.NamedToolChoice)
	assert.Equal(t, "function", tc.NamedToolChoice.Type)
	assert.Equal(t, "get_weather", tc.NamedToolChoice.Function.Name)
}

func TestToolChoice_MarshalJSON_String(t *testing.T) {
	auto := "auto"
	tc := ToolChoice{ToolChoice: &auto}
	data, err := json.Marshal(tc)

	require.NoError(t, err)
	assert.Equal(t, `"auto"`, string(data))
}

func TestToolChoice_MarshalJSON_Object(t *testing.T) {
	tc := ToolChoice{
		NamedToolChoice: &NamedToolChoice{
			Type:     "function",
			Function: ToolFunction{Name: "get_weather"},
		},
	}
	data, err := json.Marshal(tc)

	require.NoError(t, err)
	var result map[string]interface{}
	err = json.Unmarshal(data, &result)
	require.NoError(t, err)
	assert.Equal(t, "function", result["type"])
}

// Test InternalLLMRequest

func TestInternalLLMRequest_Validate(t *testing.T) {
	// Valid request
	text := "Hello"
	req := InternalLLMRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: "user", Content: MessageContent{Content: &text}},
		},
	}
	assert.NoError(t, req.Validate())

	// Missing model
	req2 := InternalLLMRequest{
		Messages: []Message{
			{Role: "user", Content: MessageContent{Content: &text}},
		},
	}
	assert.Error(t, req2.Validate())

	// Missing messages
	req3 := InternalLLMRequest{
		Model: "gpt-4",
	}
	assert.Error(t, req3.Validate())
}

func TestInternalLLMRequest_IsStreaming(t *testing.T) {
	// Not streaming (nil)
	req := InternalLLMRequest{}
	assert.False(t, req.IsStreaming())

	// Not streaming (false)
	streamFalse := false
	req2 := InternalLLMRequest{Stream: &streamFalse}
	assert.False(t, req2.IsStreaming())

	// Streaming
	streamTrue := true
	req3 := InternalLLMRequest{Stream: &streamTrue}
	assert.True(t, req3.IsStreaming())
}

func TestInternalLLMRequest_JSONRoundTrip(t *testing.T) {
	text := "Hello, how are you?"
	temp := 0.7
	maxTokens := int64(100)
	stream := true

	req := InternalLLMRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: "system", Content: MessageContent{Content: ptrString("You are a helpful assistant.")}},
			{Role: "user", Content: MessageContent{Content: &text}},
		},
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		Stream:      &stream,
		Tools: []Tool{
			{
				Type: "function",
				Function: Function{
					Name:        "get_weather",
					Description: "Get weather information",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
				},
			},
		},
	}

	// Marshal
	data, err := json.Marshal(req)
	require.NoError(t, err)

	// Unmarshal
	var req2 InternalLLMRequest
	err = json.Unmarshal(data, &req2)
	require.NoError(t, err)

	// Verify
	assert.Equal(t, req.Model, req2.Model)
	assert.Len(t, req2.Messages, 2)
	assert.Equal(t, "system", req2.Messages[0].Role)
	assert.Equal(t, "user", req2.Messages[1].Role)
	assert.Equal(t, *req.Temperature, *req2.Temperature)
	assert.Equal(t, *req.MaxTokens, *req2.MaxTokens)
	assert.Equal(t, *req.Stream, *req2.Stream)
	assert.Len(t, req2.Tools, 1)
	assert.Equal(t, "get_weather", req2.Tools[0].Function.Name)
}

// Test InternalLLMResponse

func TestInternalLLMResponse_IsError(t *testing.T) {
	// No error
	resp := InternalLLMResponse{}
	assert.False(t, resp.IsError())

	// With error
	resp2 := InternalLLMResponse{
		Error: &ResponseError{
			StatusCode: 400,
			Detail:     ErrorDetail{Message: "Bad request"},
		},
	}
	assert.True(t, resp2.IsError())
}

func TestInternalLLMResponse_GetContent(t *testing.T) {
	// Empty choices
	resp := InternalLLMResponse{}
	assert.Equal(t, "", resp.GetContent())

	// With message
	text := "Hello!"
	resp2 := InternalLLMResponse{
		Choices: []Choice{
			{
				Index:   0,
				Message: &Message{Role: "assistant", Content: MessageContent{Content: &text}},
			},
		},
	}
	assert.Equal(t, "Hello!", resp2.GetContent())

	// With delta (streaming)
	resp3 := InternalLLMResponse{
		Choices: []Choice{
			{
				Index: 0,
				Delta: &Message{Content: MessageContent{Content: &text}},
			},
		},
	}
	assert.Equal(t, "Hello!", resp3.GetContent())
}

func TestInternalLLMResponse_JSONRoundTrip(t *testing.T) {
	text := "Hello!"
	finishReason := "stop"
	resp := InternalLLMResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1677652288,
		Model:   "gpt-4",
		Choices: []Choice{
			{
				Index:        0,
				Message:      &Message{Role: "assistant", Content: MessageContent{Content: &text}},
				FinishReason: &finishReason,
			},
		},
		Usage: &Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	// Marshal
	data, err := json.Marshal(resp)
	require.NoError(t, err)

	// Unmarshal
	var resp2 InternalLLMResponse
	err = json.Unmarshal(data, &resp2)
	require.NoError(t, err)

	// Verify
	assert.Equal(t, resp.ID, resp2.ID)
	assert.Equal(t, resp.Object, resp2.Object)
	assert.Equal(t, resp.Created, resp2.Created)
	assert.Equal(t, resp.Model, resp2.Model)
	assert.Len(t, resp2.Choices, 1)
	assert.Equal(t, "assistant", resp2.Choices[0].Message.Role)
	assert.Equal(t, "Hello!", resp2.Choices[0].Message.Content.GetText())
	assert.Equal(t, "stop", *resp2.Choices[0].FinishReason)
	assert.NotNil(t, resp2.Usage)
	assert.Equal(t, int64(10), resp2.Usage.PromptTokens)
	assert.Equal(t, int64(5), resp2.Usage.CompletionTokens)
	assert.Equal(t, int64(15), resp2.Usage.TotalTokens)
}

// Test ResponseError

func TestResponseError_Error(t *testing.T) {
	err := ResponseError{
		StatusCode: 400,
		Detail: ErrorDetail{
			Code:    "invalid_request",
			Message: "Invalid request",
		},
	}
	assert.Contains(t, err.Error(), "Invalid request")
	assert.Contains(t, err.Error(), "invalid_request")
}

// Helper function
func ptrString(s string) *string {
	return &s
}
