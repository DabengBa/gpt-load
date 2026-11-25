package control

import (
	"net/http"
	"testing"

	"github.com/tiktoken-go/tokenizer"

	"gpt-load/internal/execution"
	"gpt-load/internal/usage"
)

func TestModelTraceContextPromptTokenEstimate(t *testing.T) {
	for _, encoding := range []tokenizer.Encoding{tokenizer.O200kBase, tokenizer.Cl100kBase} {
		codec, err := tokenizer.Get(encoding)
		if err != nil {
			t.Fatal(err)
		}
		count, err := codec.Count(modelTraceContextPrompt)
		if err != nil || int64(count) != modelTraceContextExpectedTokens {
			t.Fatalf("%s count = %d, %v; want %d", encoding, count, err, modelTraceContextExpectedTokens)
		}
	}
}

func TestModelTraceContextCheckUsage(t *testing.T) {
	for _, tc := range []struct {
		name   string
		code   int
		state  usage.State
		input  int64
		status modelTraceContextStatus
	}{
		{"within tolerance", http.StatusOK, usage.StateComplete, 30, modelTraceContextNoAnomaly},
		{"above tolerance", http.StatusOK, usage.StateComplete, 31, modelTraceContextSuspected},
		{"below estimate", http.StatusOK, usage.StateComplete, 5, modelTraceContextNoAnomaly},
		{"missing input", http.StatusOK, usage.StateComplete, 0, modelTraceContextUnavailable},
		{"partial usage", http.StatusOK, usage.StatePartial, 50, modelTraceContextUnavailable},
		{"failed response", http.StatusTooManyRequests, usage.StateComplete, 50, modelTraceContextUnavailable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := evaluateModelTraceContextCheck(execution.AttemptResult{
				StatusCode: tc.code,
				Usage: &execution.UsageEvidence{Normalized: usage.Result{
					State: tc.state, Tokens: usage.Tokens{UncachedInput: tc.input, Output: 100},
				}},
			})
			if got.Status != tc.status {
				t.Fatalf("status = %q, want %q", got.Status, tc.status)
			}
			if tc.status == modelTraceContextUnavailable {
				if got.ReportedInputTokens != nil || got.Delta != nil {
					t.Fatalf("unavailable check returned reported values: %#v", got)
				}
			} else if got.ReportedInputTokens == nil || *got.ReportedInputTokens != tc.input ||
				got.Delta == nil || *got.Delta != tc.input-modelTraceContextExpectedTokens {
				t.Fatalf("check = %#v, want input %d and its delta", got, tc.input)
			}
		})
	}
}
