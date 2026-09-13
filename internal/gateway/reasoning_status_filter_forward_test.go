package gateway

import (
	"bytes"
	"testing"

	"gpt-load/internal/state"
)

func TestNewExecutionAttemptSpecFiltersReasoningStatusWhenGroupEnabled(t *testing.T) {
	input := responsesExecutionForwardInput()
	input.Group.ResponsesReasoningStatusFilterEnabled = true
	input.Request.Body = []byte(`{"model":"public","input":[{"type":"reasoning","status":"completed","summary":[]},{"type":"function_call","status":"completed","call_id":"call_1"}]}`)

	spec, err := newExecutionAttemptSpec(input)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(spec.Body, []byte(`"type":"reasoning","status"`)) {
		t.Fatalf("reasoning status was forwarded: %s", spec.Body)
	}
	if !bytes.Contains(spec.Body, []byte(`"type":"function_call","status":"completed"`)) {
		t.Fatalf("function call status was removed: %s", spec.Body)
	}
}

func TestNewExecutionAttemptSpecKeepsReasoningStatusWhenGroupDisabled(t *testing.T) {
	input := responsesExecutionForwardInput()
	input.Group = state.GroupView{}
	input.Request.Body = []byte(`{"model":"public","input":[{"type":"reasoning","status":"completed","summary":[]}]}`)

	spec, err := newExecutionAttemptSpec(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(spec.Body, []byte(`"status":"completed"`)) {
		t.Fatalf("reasoning status was removed while disabled: %s", spec.Body)
	}
}
