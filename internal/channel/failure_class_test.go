package channel

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

func TestClassifyFailureUsesMinimalAllowlist(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		value      string
		want       FailureClass
	}{
		{name: "unauthorized", statusCode: http.StatusUnauthorized, want: FailureClassCredential},
		{name: "explicit credential", statusCode: http.StatusForbidden, value: "api_key_invalid", want: FailureClassCredential},
		{name: "not found", statusCode: http.StatusNotFound, want: FailureClassModel},
		{name: "explicit model", statusCode: http.StatusForbidden, value: "model_not_available", want: FailureClassModel},
		{name: "rate limited", statusCode: http.StatusTooManyRequests, want: FailureClassRateLimited},
		{name: "generic forbidden", statusCode: http.StatusForbidden, value: "permission_denied", want: FailureClassUnknown},
		{name: "billing forbidden", statusCode: http.StatusPaymentRequired, value: "billing disabled", want: FailureClassUnknown},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := ClassifyFailure(test.statusCode, test.value); got != test.want {
				t.Fatalf("ClassifyFailure() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestFailureHintUsesSharedEvidenceVocabulary(t *testing.T) {
	if got := FailureHint(http.StatusUnauthorized); got != execution.FailureHintInvalidCredential {
		t.Fatalf("credential hint = %q", got)
	}
	if got := FailureHint(http.StatusNotFound); got != execution.FailureHintModelUnavailable {
		t.Fatalf("model hint = %q", got)
	}
	if got := FailureHint(http.StatusForbidden, "permission_denied"); got != "" {
		t.Fatalf("generic forbidden hint = %q, want empty", got)
	}
}
