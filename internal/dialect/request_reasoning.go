package dialect

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
)

const maxReasoningValueBytes = 64

// SupportsReasoningEffortOverride limits group effort replacement to generation
// operations whose client protocol defines the corresponding field.
func SupportsReasoningEffortOverride(client protocol.Protocol, operation execution.Operation) bool {
	switch operation {
	case execution.OperationChatCompletion:
		return client == protocol.OpenAICompletions || client == protocol.Anthropic || client == protocol.Gemini
	case execution.OperationResponsesCreate:
		return client == protocol.OpenAIResponses
	default:
		return false
	}
}

// HasReasoningEffort reports whether the client request includes its protocol effort field.
func HasReasoningEffort(body []byte, client protocol.Protocol) bool {
	_, _, exists := jsonPathValue(body, reasoningEffortPath(client))
	return exists
}

// OverrideReasoningEffort replaces an existing client-protocol effort field.
// It deliberately does not inject reasoning into requests that omitted it.
func OverrideReasoningEffort(body []byte, effort string, client protocol.Protocol) ([]byte, bool, error) {
	start, end, exists := jsonPathValue(body, reasoningEffortPath(client))
	if !exists {
		return body, false, nil
	}
	encodedEffort, err := json.Marshal(effort)
	if err != nil {
		return nil, false, err
	}
	updated := make([]byte, 0, len(body)-end+start+len(encodedEffort))
	updated = append(updated, body[:start]...)
	updated = append(updated, encodedEffort...)
	updated = append(updated, body[end:]...)
	return updated, true, nil
}

// jsonPathValue finds the final matching field at each path level, matching
// encoding/json's object decoding behavior while retaining untouched bytes.
func jsonPathValue(body []byte, path []string) (int, int, bool) {
	if len(path) == 0 || !json.Valid(body) {
		return 0, 0, false
	}
	start := skipJSONWhitespace(body, 0)
	end := skipJSONValue(body, start)
	if end < 0 || skipJSONWhitespace(body, end) != len(body) {
		return 0, 0, false
	}
	for _, field := range path {
		var exists bool
		start, end, exists = jsonObjectFieldValue(body, start, end, field)
		if !exists {
			return 0, 0, false
		}
	}
	return start, end, true
}

func jsonObjectFieldValue(body []byte, start, end int, field string) (int, int, bool) {
	if start >= end || body[start] != '{' {
		return 0, 0, false
	}
	index := skipJSONWhitespace(body, start+1)
	valueStart, valueEnd := 0, 0
	found := false
	for index < end {
		if body[index] == '}' {
			return valueStart, valueEnd, found
		}
		if body[index] != '"' {
			return 0, 0, false
		}
		keyEnd := skipJSONString(body, index)
		if keyEnd < 0 {
			return 0, 0, false
		}
		var key string
		if json.Unmarshal(body[index:keyEnd], &key) != nil {
			return 0, 0, false
		}
		index = skipJSONWhitespace(body, keyEnd)
		if index >= end || body[index] != ':' {
			return 0, 0, false
		}
		index = skipJSONWhitespace(body, index+1)
		candidateStart := index
		candidateEnd := skipJSONValue(body, candidateStart)
		if candidateEnd < 0 || candidateEnd > end {
			return 0, 0, false
		}
		if key == field {
			valueStart, valueEnd, found = candidateStart, candidateEnd, true
		}
		index = skipJSONWhitespace(body, candidateEnd)
		if index >= end {
			return 0, 0, false
		}
		switch body[index] {
		case ',':
			index = skipJSONWhitespace(body, index+1)
		case '}':
			return valueStart, valueEnd, found
		default:
			return 0, 0, false
		}
	}
	return 0, 0, false
}

func skipJSONWhitespace(body []byte, index int) int {
	for index < len(body) {
		switch body[index] {
		case ' ', '\n', '\r', '\t':
			index++
		default:
			return index
		}
	}
	return index
}

func skipJSONValue(body []byte, start int) int {
	start = skipJSONWhitespace(body, start)
	if start >= len(body) {
		return -1
	}
	switch body[start] {
	case '"':
		return skipJSONString(body, start)
	case '{', '[':
		closers := make([]byte, 0, 4)
		if body[start] == '{' {
			closers = append(closers, '}')
		} else {
			closers = append(closers, ']')
		}
		for index := start + 1; index < len(body); index++ {
			switch body[index] {
			case '"':
				next := skipJSONString(body, index)
				if next < 0 {
					return -1
				}
				index = next - 1
			case '{':
				closers = append(closers, '}')
			case '[':
				closers = append(closers, ']')
			case '}', ']':
				if len(closers) == 0 || body[index] != closers[len(closers)-1] {
					return -1
				}
				closers = closers[:len(closers)-1]
				if len(closers) == 0 {
					return index + 1
				}
			}
		}
		return -1
	case ',', '}', ']':
		return -1
	default:
		index := start
		for index < len(body) {
			switch body[index] {
			case ' ', '\n', '\r', '\t', ',', '}', ']':
				return index
			default:
				index++
			}
		}
		return index
	}
}

func skipJSONString(body []byte, start int) int {
	for index := start + 1; index < len(body); index++ {
		switch body[index] {
		case '\\':
			index++
		case '"':
			return index + 1
		}
	}
	return -1
}

func reasoningEffortPath(client protocol.Protocol) []string {
	switch client {
	case protocol.OpenAICompletions:
		return []string{"reasoning_effort"}
	case protocol.OpenAIResponses:
		return []string{"reasoning", "effort"}
	case protocol.Anthropic:
		return []string{"output_config", "effort"}
	case protocol.Gemini:
		return []string{"generationConfig", "thinkingConfig", "thinkingLevel"}
	default:
		return nil
	}
}

func inspectOpenAICompletionsReasoning(body []byte) reasoning.Config {
	root, ok := reasoningObject(body)
	if !ok {
		return reasoning.Config{}
	}
	return reasoning.Config{Effort: reasoningString(root, "reasoning_effort")}
}

func inspectOpenAIResponsesReasoning(body []byte) reasoning.Config {
	root, ok := reasoningObject(body)
	if !ok {
		return reasoning.Config{}
	}
	nested, ok := reasoningNestedObject(root, "reasoning")
	if !ok {
		return reasoning.Config{}
	}
	return reasoning.Config{
		Mode:   reasoningString(nested, "mode"),
		Effort: reasoningString(nested, "effort"),
	}
}

func inspectAnthropicReasoning(body []byte) reasoning.Config {
	root, ok := reasoningObject(body)
	if !ok {
		return reasoning.Config{}
	}
	result := reasoning.Config{}
	if thinking, exists := reasoningNestedObject(root, "thinking"); exists {
		result.Mode = reasoningString(thinking, "type")
		result.BudgetTokens = reasoningInteger(thinking, "budget_tokens")
	}
	if outputConfig, exists := reasoningNestedObject(root, "output_config"); exists {
		result.Effort = reasoningString(outputConfig, "effort")
	}
	return result
}

func inspectGeminiReasoning(body []byte) reasoning.Config {
	root, ok := reasoningObject(body)
	if !ok {
		return reasoning.Config{}
	}
	generationConfig, ok := reasoningNestedObject(root, "generationConfig")
	if !ok {
		return reasoning.Config{}
	}
	thinkingConfig, ok := reasoningNestedObject(generationConfig, "thinkingConfig")
	if !ok {
		return reasoning.Config{}
	}
	return reasoning.Config{
		Effort:       reasoningString(thinkingConfig, "thinkingLevel"),
		BudgetTokens: reasoningInteger(thinkingConfig, "thinkingBudget"),
	}
}

func reasoningObject(body []byte) (map[string]json.RawMessage, bool) {
	if len(body) == 0 {
		return nil, false
	}
	object, err := decodeJSONObject(body)
	return object, err == nil
}

func reasoningNestedObject(
	object map[string]json.RawMessage,
	field string,
) (map[string]json.RawMessage, bool) {
	raw, exists := object[field]
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, false
	}
	nested, err := decodeJSONObject(raw)
	return nested, err == nil
}

func reasoningString(object map[string]json.RawMessage, field string) string {
	value, ok := jsonString(object, field)
	if !ok || len(value) == 0 || len(value) > maxReasoningValueBytes {
		return ""
	}
	for _, character := range value {
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			character == '_' || character == '-' || character == '.' {
			continue
		}
		return ""
	}
	return strings.ToLower(value)
}

func reasoningInteger(object map[string]json.RawMessage, field string) *int64 {
	raw, exists := object[field]
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	value, err := strconv.ParseInt(string(bytes.TrimSpace(raw)), 10, 64)
	if err != nil {
		return nil
	}
	return &value
}
