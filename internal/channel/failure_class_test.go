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
		{name: "payment required is billing", statusCode: http.StatusPaymentRequired, value: "billing disabled", want: FailureClassBilling},
		{name: "quota code beats 429 status", statusCode: http.StatusTooManyRequests, value: "insufficient_quota", want: FailureClassBilling},
		{name: "insufficient balance message", statusCode: http.StatusBadRequest, value: "Your credit balance is too low", want: FailureClassBilling},
		{name: "chinese balance message", statusCode: http.StatusForbidden, value: "当前账户余额不足，请充值", want: FailureClassBilling},
		{name: "self-healing quota stays rate limited", statusCode: http.StatusTooManyRequests, value: "quota_exceeded", want: FailureClassRateLimited},
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
