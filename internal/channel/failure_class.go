package channel

import (
	"net/http"
	"strings"

	"gpt-load/internal/execution"
)

// FailureClass is the smallest provider-neutral classification needed by
// routing health state. Unknown means the evidence is intentionally left
// unscoped rather than guessing from a generic provider error.
type FailureClass string

const (
	FailureClassUnknown     FailureClass = "unknown"
	FailureClassCredential  FailureClass = "credential"
	FailureClassModel       FailureClass = "model"
	FailureClassRateLimited FailureClass = "rate_limited"
)

// ClassifyFailure applies the bounded provider error allowlist. Status codes
// are considered only where their route meaning is unambiguous; generic 403
// permission errors remain unknown.
func ClassifyFailure(statusCode int, values ...string) FailureClass {
	markers := strings.ToLower(strings.Join(values, " "))
	if statusCode == http.StatusUnauthorized || containsFailureMarker(markers,
		"invalid_api_key", "api_key_invalid", "authentication_error",
		"authentication failed", "invalid credential", "api key not valid") {
		return FailureClassCredential
	}
	if statusCode == http.StatusNotFound || containsFailureMarker(markers,
		"model_not_found", "model not found", "model_not_available",
		"model unavailable", "deployment_not_found", "unsupported_model",
		"model_access_denied", "model permission denied", "model not authorized") {
		return FailureClassModel
	}
	if statusCode == http.StatusTooManyRequests || containsFailureMarker(markers,
		"rate_limit", "rate limit", "too_many_requests", "quota_exceeded",
		"resource_exhausted", "throttl") {
		return FailureClassRateLimited
	}
	return FailureClassUnknown
}

// FailureHint converts the bounded classification into the shared execution
// evidence vocabulary. Rate limiting deliberately carries no scope hint so the
// existing health policy remains compatible for generic 429 responses.
func FailureHint(statusCode int, values ...string) execution.FailureHint {
	switch ClassifyFailure(statusCode, values...) {
	case FailureClassCredential:
		return execution.FailureHintInvalidCredential
	case FailureClassModel:
		return execution.FailureHintModelUnavailable
	case FailureClassRateLimited:
		return execution.FailureHintRateLimited
	default:
		return ""
	}
}

func containsFailureMarker(value string, markers ...string) bool {
	for _, marker := range markers {
		if strings.Contains(value, marker) {
			return true
		}
	}
	return false
}
