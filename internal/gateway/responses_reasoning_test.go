package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

// Deliberately put status before type and separate them with other fields.
const reasoningStatusInput = `[{"status":"completed","id":"r1","type":"reasoning","summary":[{"type":"summary_text","text":"keep <&>"}],"encrypted_content":"opaque"},{"summary":[],"status":null,"id":"r2","type":"reasoning"},{"status":"completed","role":"assistant","type":"message","content":[{"type":"output_text","text":"history"}]},{"status":"completed","call_id":"call_1","name":"lookup","arguments":"{}","type":"function_call"}]`
const reasoningStatusFilteredInput = `[{"id":"r1","type":"reasoning","summary":[{"type":"summary_text","text":"keep <&>"}],"encrypted_content":"opaque"},{"summary":[],"id":"r2","type":"reasoning"},{"status":"completed","role":"assistant","type":"message","content":[{"type":"output_text","text":"history"}]},{"status":"completed","call_id":"call_1","name":"lookup","arguments":"{}","type":"function_call"}]`

func TestResponsesReasoningStatusReorderedFields(t *testing.T) {
	if strings.Contains(reasoningStatusInput, `"type":"reasoning","status"`) {
		t.Fatal("negative control does not reproduce the old assertion blind spot")
	}
	got, err := removeResponsesReasoningStatus([]byte(`{"input":` + reasoningStatusInput + `}`))
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, string(got), `{"input":`+reasoningStatusFilteredInput+`}`)
}

func TestRemoveResponsesReasoningStatus(t *testing.T) {
	input := []byte(`{"model":"gpt-5","input":[{"type":"reasoning","id":"r1","status":"completed","summary":[]},{"type":"message","status":"completed","content":[]},{"type":"function_call","status":"completed","call_id":"call_1"},{"type":"reasoning","id":"r2","status":null,"encrypted_content":"opaque"}]}`)

	got, err := removeResponsesReasoningStatus(input)
	if err != nil {
		t.Fatal(err)
	}

	var object map[string]any
	if err := json.Unmarshal(got, &object); err != nil {
		t.Fatal(err)
	}
	items := object["input"].([]any)
	if _, exists := items[0].(map[string]any)["status"]; exists {
		t.Fatal("reasoning status was not removed")
	}
	if _, exists := items[3].(map[string]any)["status"]; exists {
		t.Fatal("null reasoning status was not removed")
	}
	if items[1].(map[string]any)["status"] != "completed" {
		t.Fatal("message status was removed")
	}
	if items[2].(map[string]any)["status"] != "completed" {
		t.Fatal("function call status was removed")
	}
}

func TestRemoveResponsesReasoningStatusPreservesHTMLCharacters(t *testing.T) {
	input := []byte(`{"input":[{"type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"` + strings.Repeat("<&>", 40) + `"}]}]}`)

	got, err := removeResponsesReasoningStatus(input)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(got), `\u003c`) || strings.Contains(string(got), `\u0026`) {
		t.Fatalf("reasoning content was HTML-escaped: %s", got)
	}
}

func TestRemoveResponsesReasoningStatusLeavesPayloadWithoutReasoningStatusUnchanged(t *testing.T) {
	for _, payload := range []string{
		`{ "model":"gpt-5", "input":"hello" }`,
		`{ "input": null }`,
		`{ "model":"gpt-5" }`,
		`{ "input": [] }`,
		`{ "input": [{"type":"reasoning","summary":[]},{"status":"completed","type":"message","content":[]}] }`,
		// Preserve existing behavior; this is not a claim that object input is supported by Responses.
		`{ "input": {"status":"completed","type":"reasoning","summary":[]} }`,
	} {
		t.Run(payload, func(t *testing.T) {
			input := []byte(payload)
			got, err := removeResponsesReasoningStatus(input)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(input) {
				t.Fatalf("payload = %s, want unchanged %s", got, input)
			}
		})
	}
}
