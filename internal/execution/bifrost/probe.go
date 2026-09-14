package bifrost

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"

	"gpt-load/internal/execution"
	"gpt-load/internal/platform/contentcoding"
	"gpt-load/internal/protocol"
)

// probeQuestion is the single code-owned low-cost question every probe asks. It
// is short, deterministic, and clearly answerable, so a non-empty answer is
// strong evidence that the upstream actually generated text instead of merely
// accepting the request.
const probeQuestion = "What is 2 + 2? Please answer briefly."

// probeDefaultOutputTokens is the fallback output budget for a probe that did
// not carry a contract value. Generation probes always carry one; the fallback
// only protects a miswired caller from omitting the field entirely.
const probeDefaultOutputTokens = 16

// probeOutputTokenBudget returns the code-owned output budget selected by the
// channel probe contract.
func probeOutputTokenBudget(spec execution.AttemptSpec) int {
	if spec.ProbeMaxOutputTokens > 0 {
		return spec.ProbeMaxOutputTokens
	}
	return probeDefaultOutputTokens
}

// normalizeProbeAttemptResult records how a successful probe response resolved:
// whether it carried usable generated text (ProbeAnswerPresent) and whether it
// parsed as the selected protocol's wire shape at all (ProbeResponseInvalid). It
// only sets these probe fields and never changes ordinary result semantics or
// health decisions.
func normalizeProbeAttemptResult(spec execution.AttemptSpec, result *execution.AttemptResult) {
	if result == nil || spec.Operation != execution.OperationProbe {
		return
	}
	result.ProbeAnswerPresent = false
	result.ProbeResponseInvalid = false
	if result.Error != nil ||
		result.StatusCode < http.StatusOK ||
		result.StatusCode >= http.StatusMultipleChoices {
		return
	}
	body, err := decodeProbeResponseBody(result)
	if err != nil {
		// The response could not even be materialized as the selected protocol's
		// JSON. That is response corruption, not merely a missing answer.
		result.ProbeResponseInvalid = true
		return
	}
	extraction := probeAnswerPresent(spec.ClientProtocol, body)
	result.ProbeAnswerPresent = extraction.present
	result.ProbeResponseInvalid = !extraction.valid
}

// decodeProbeResponseBody materializes the probe response bytes. Native
// passthrough responses may still be content-encoded; the probe only needs the
// decoded JSON to extract text.
func decodeProbeResponseBody(result *execution.AttemptResult) ([]byte, error) {
	if result == nil {
		return nil, nil
	}
	encoding, err := contentcoding.ParseContentEncoding(result.Header.Values("Content-Encoding"))
	if err != nil {
		return nil, err
	}
	return contentcoding.DecodeLimited(
		encoding,
		result.Body,
		execution.UnaryResponseBodyLimit(protocol.OpenAICompletions),
	)
}

// probeExtraction is one protocol-shape extraction outcome: valid reports that
// the body parsed as the selected protocol's wire shape (even when it carried no
// text); present reports that it carried non-empty generated text.
type probeExtraction struct {
	present bool
	valid   bool
}

// probeAnswerPresent extracts non-empty generated text from a probe response.
// The executor serializes each protocol into a different wire shape: the
// converted Responses path emits the client protocol's own shape, the generic
// chat path emits the normalized OpenAI chat shape, and native passthrough
// emits the raw upstream shape. The selected client protocol picks the first
// shape, then the normalized OpenAI chat shape covers every converted chat
// route. A body matching neither shape is invalid, not merely empty.
func probeAnswerPresent(clientProtocol protocol.Protocol, body []byte) probeExtraction {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return probeExtraction{}
	}
	var envelope map[string]json.RawMessage
	if json.Unmarshal(trimmed, &envelope) != nil {
		return probeExtraction{}
	}
	switch clientProtocol {
	case protocol.OpenAIResponses:
		if extraction := openAIResponsesExtraction(envelope); extraction.valid {
			return extraction
		}
	case protocol.Anthropic:
		if extraction := anthropicExtraction(envelope); extraction.valid {
			return extraction
		}
	case protocol.Gemini:
		if extraction := geminiExtraction(envelope); extraction.valid {
			return extraction
		}
	}
	return openAIChatExtraction(envelope)
}

func openAIChatExtraction(envelope map[string]json.RawMessage) probeExtraction {
	raw, ok := envelope["choices"]
	if !ok {
		return probeExtraction{}
	}
	var choices []struct {
		Message json.RawMessage `json:"message"`
		Text    string          `json:"text"`
	}
	if json.Unmarshal(raw, &choices) != nil {
		return probeExtraction{}
	}
	for _, choice := range choices {
		if hasProbeText(choice.Text) {
			return probeExtraction{present: true, valid: true}
		}
		var message struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(choice.Message, &message) != nil {
			continue
		}
		if hasProbeText(probeContentText(message.Content)) {
			return probeExtraction{present: true, valid: true}
		}
	}
	return probeExtraction{valid: true}
}

// probeContentText accepts the string or content-block array spellings a chat
// response can use for one message's content.
func probeContentText(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}
	if trimmed[0] == '"' {
		var text string
		if json.Unmarshal(trimmed, &text) != nil {
			return ""
		}
		return text
	}
	if trimmed[0] != '[' {
		return ""
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(trimmed, &blocks) != nil {
		return ""
	}
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if hasProbeText(block.Text) {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "")
}

func openAIResponsesExtraction(envelope map[string]json.RawMessage) probeExtraction {
	outputRaw, hasOutput := envelope["output"]
	outputTextRaw, hasOutputText := envelope["output_text"]
	if !hasOutput && !hasOutputText {
		return probeExtraction{}
	}
	if hasOutputText {
		var outputText string
		if json.Unmarshal(outputTextRaw, &outputText) != nil {
			return probeExtraction{}
		}
		if hasProbeText(outputText) {
			return probeExtraction{present: true, valid: true}
		}
	}
	if hasOutput {
		var output []struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(outputRaw, &output) != nil {
			return probeExtraction{}
		}
		for _, item := range output {
			if hasProbeText(responsesContentText(item.Content)) {
				return probeExtraction{present: true, valid: true}
			}
		}
	}
	return probeExtraction{valid: true}
}

func responsesContentText(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}
	if trimmed[0] == '"' {
		var text string
		if json.Unmarshal(trimmed, &text) != nil {
			return ""
		}
		return text
	}
	if trimmed[0] != '[' {
		return ""
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(trimmed, &blocks) != nil {
		return ""
	}
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if block.Type != "output_text" && block.Type != "text" {
			continue
		}
		if hasProbeText(block.Text) {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "")
}

func anthropicExtraction(envelope map[string]json.RawMessage) probeExtraction {
	raw, ok := envelope["content"]
	if !ok {
		return probeExtraction{}
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return probeExtraction{}
	}
	for _, block := range blocks {
		if block.Type == "text" && hasProbeText(block.Text) {
			return probeExtraction{present: true, valid: true}
		}
	}
	return probeExtraction{valid: true}
}

func geminiExtraction(envelope map[string]json.RawMessage) probeExtraction {
	raw, ok := envelope["candidates"]
	if !ok {
		return probeExtraction{}
	}
	var candidates []struct {
		Content struct {
			Parts []struct {
				Text    string `json:"text"`
				Thought bool   `json:"thought"`
			} `json:"parts"`
		} `json:"content"`
	}
	if json.Unmarshal(raw, &candidates) != nil {
		return probeExtraction{}
	}
	for _, candidate := range candidates {
		for _, part := range candidate.Content.Parts {
			// Gemini marks internal reasoning with thought:true; it is not
			// user-facing generated text and must not count as an answer.
			if part.Thought {
				continue
			}
			if hasProbeText(part.Text) {
				return probeExtraction{present: true, valid: true}
			}
		}
	}
	return probeExtraction{valid: true}
}

func hasProbeText(value string) bool {
	return strings.TrimSpace(value) != ""
}
