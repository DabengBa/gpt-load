package bifrost

import (
	"encoding/json"
	"strings"

	"gpt-load/internal/protocol"
)

// probeResponseHasGeneratedText reports whether the final response body carries
// visible generated text for the selected client protocol. It deliberately does
// not use a union parser: accepting another protocol's carrier would turn a
// valid but mismatched 2xx response into probe evidence.
func probeResponseHasGeneratedText(clientProtocol protocol.Protocol, body []byte) bool {
	if len(body) == 0 || !json.Valid(body) {
		return false
	}
	switch clientProtocol {
	case protocol.OpenAICompletions:
		var response struct {
			Object  string `json:"object"`
			Choices []struct {
				Message struct {
					Role    string          `json:"role"`
					Content json.RawMessage `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &response); err != nil || response.Object != "chat.completion" {
			return false
		}
		for _, choice := range response.Choices {
			if choice.Message.Role == "assistant" && chatMessageContentHasText(choice.Message.Content) {
				return true
			}
		}
	case protocol.OpenAIResponses:
		var response struct {
			Object string `json:"object"`
			Output []struct {
				Type    string `json:"type"`
				Content []struct {
					Type string  `json:"type"`
					Text *string `json:"text"`
				} `json:"content"`
			} `json:"output"`
		}
		if err := json.Unmarshal(body, &response); err != nil || response.Object != "response" {
			return false
		}
		for _, output := range response.Output {
			if output.Type != "message" {
				continue
			}
			for _, content := range output.Content {
				if content.Type == "output_text" && hasGeneratedText(content.Text) {
					return true
				}
			}
		}
	case protocol.Anthropic:
		var response struct {
			Type    string `json:"type"`
			Content []struct {
				Type string  `json:"type"`
				Text *string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal(body, &response); err != nil || response.Type != "message" {
			return false
		}
		for _, content := range response.Content {
			if content.Type == "text" && hasGeneratedText(content.Text) {
				return true
			}
		}
	case protocol.Gemini:
		var response struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						Text    *string `json:"text"`
						Thought bool    `json:"thought"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal(body, &response); err != nil {
			return false
		}
		for _, candidate := range response.Candidates {
			for _, part := range candidate.Content.Parts {
				if part.Thought {
					continue
				}
				if hasGeneratedText(part.Text) {
					return true
				}
			}
		}
	}
	return false
}

func chatMessageContentHasText(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return strings.TrimSpace(text) != ""
	}
	var parts []struct {
		Type string  `json:"type"`
		Text *string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return false
	}
	for _, part := range parts {
		if part.Type == "text" && hasGeneratedText(part.Text) {
			return true
		}
	}
	return false
}

func hasGeneratedText(text *string) bool {
	return text != nil && strings.TrimSpace(*text) != ""
}
