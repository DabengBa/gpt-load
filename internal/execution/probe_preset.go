package execution

import "encoding/json"

// ResponsesProbePreset constructs the code-owned Responses probe request.
func ResponsesProbePreset(model, prompt string, budget int) ([]byte, error) {
	return json.Marshal(map[string]any{
		"model": model,
		"input": []map[string]any{{"role": "user", "content": []map[string]any{{
			"type": "input_text", "text": prompt,
		}}}},
		"max_output_tokens": budget, "store": false, "stream": false,
	})
}

// ProbeQuestion is the code-owned prompt for generation probes.
const ProbeQuestion = "What is 2 + 2? Please answer briefly."

// ChatProbePreset constructs the ordinary chat shape used by a model probe.
func ChatProbePreset(model string, budget int) ([]byte, error) {
	return json.Marshal(map[string]any{
		"model": model, "messages": []map[string]string{{"role": "user", "content": ProbeQuestion}},
		"max_tokens": budget,
	})
}

// GeminiProbePreset constructs the ordinary Gemini generation request.
func GeminiProbePreset(budget int) ([]byte, error) {
	return json.Marshal(map[string]any{
		"contents":         []any{map[string]any{"role": "user", "parts": []map[string]string{{"text": ProbeQuestion}}}},
		"generationConfig": map[string]int{"maxOutputTokens": budget},
	})
}
