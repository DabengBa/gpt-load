package bifrost

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"unicode/utf8"

	"gpt-load/internal/execution"
	platformredact "gpt-load/internal/platform/redact"
)

func openAIErrorTypeCode(body []byte) (string, string) {
	envelope := providerErrorEnvelope(body)
	return envelope.typeValue, envelope.codeValue
}

type providerErrorValues struct {
	typeValue    string
	codeValue    string
	statusValue  string
	messageValue string
}

// providerErrorEnvelope reads the type/code/status/message fields of an upstream
// error body. The canonical OpenAI shape nests them under "error", while the
// reseller gateways this deployment talks to answer with a top-level
// {"code","message"} object. The nested shape keeps precedence so a wrapper can
// never override it; both shapes carry the same failure markers.
func providerErrorEnvelope(body []byte) providerErrorValues {
	type errorFields struct {
		Type    json.RawMessage `json:"type"`
		Code    json.RawMessage `json:"code"`
		Status  json.RawMessage `json:"status"`
		Message string          `json:"message"`
	}
	var payload struct {
		errorFields
		Error errorFields `json:"error"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return providerErrorValues{}
	}
	preferred := func(nested, top string) string {
		if nested != "" {
			return nested
		}
		return top
	}
	return providerErrorValues{
		typeValue:    preferred(evidenceScalar(payload.Error.Type), evidenceScalar(payload.errorFields.Type)),
		codeValue:    preferred(evidenceScalar(payload.Error.Code), evidenceScalar(payload.errorFields.Code)),
		statusValue:  preferred(evidenceScalar(payload.Error.Status), evidenceScalar(payload.errorFields.Status)),
		messageValue: preferred(payload.Error.Message, payload.errorFields.Message),
	}
}

func evidenceScalar(raw json.RawMessage) string {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return ""
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var number json.Number
	if err := decoder.Decode(&number); err == nil {
		return number.String()
	}
	return ""
}

func redactSecrets(body []byte, secrets []string) []byte {
	clone := append([]byte(nil), body...)
	for _, secret := range secrets {
		if secret != "" {
			clone = bytes.ReplaceAll(clone, []byte(secret), []byte("[REDACTED]"))
		}
	}
	return clone
}

func evidenceHeaders(headers http.Header) http.Header {
	if len(headers) == 0 {
		return nil
	}
	result := make(http.Header)
	for name, values := range headers {
		canonical := http.CanonicalHeaderKey(name)
		if canonical == "Request-Id" || canonical == "X-Request-Id" || canonical == "Retry-After" ||
			canonical == "Traceparent" || strings.HasPrefix(canonical, "Ratelimit-") || strings.HasPrefix(canonical, "X-Ratelimit-") {
			result[canonical] = append([]string(nil), values...)
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

const errorEvidenceTruncatedMarker = "...[truncated]"

func errorMessageSummary(body []byte, fallback string, secrets []string) string {
	summary := platformredact.ExtractErrorMessage(body)
	if summary == "" {
		summary = fallback
	}
	summary = strings.ToValidUTF8(summary, "\uFFFD")
	redactor := platformredact.New()
	summary = redactor.String(summary, secrets...)
	summary = strings.Join(strings.Fields(summary), " ")
	summary = redactor.String(summary, secrets...)
	if summary == "" {
		summary = fallback
	}
	return truncateErrorEvidenceSummary(summary)
}

func truncateErrorEvidenceSummary(summary string) string {
	if utf8.RuneCountInString(summary) <= execution.MaxErrorSummaryLength {
		return summary
	}
	marker := []rune(errorEvidenceTruncatedMarker)
	limit := execution.MaxErrorSummaryLength - len(marker)
	if limit <= 0 {
		return string(marker[:execution.MaxErrorSummaryLength])
	}
	return string([]rune(summary)[:limit]) + errorEvidenceTruncatedMarker
}

func sanitizeEvidenceValue(value string, secrets []string) string {
	value = strings.TrimSpace(value)
	if value == "" || containsSecret(value, secrets) || strings.ContainsAny(value, "\r\n\x00") {
		return ""
	}
	const limit = 128
	if utf8.RuneCountInString(value) <= limit {
		return value
	}
	runes := []rune(value)
	return string(runes[:limit])
}

func containsSecret(value string, secrets []string) bool {
	for _, secret := range secrets {
		if secret != "" && strings.Contains(value, secret) {
			return true
		}
	}
	return false
}
