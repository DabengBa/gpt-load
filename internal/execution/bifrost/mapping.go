package bifrost

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"

	"gpt-load/internal/execution"
	platformheader "gpt-load/internal/platform/httpheader"
)

func responseHeaders(primary map[string]string, bifrostContext *schemas.BifrostContext, stream bool) http.Header {
	combined := make(map[string]string, len(primary)+4)
	for key, value := range primary {
		combined[key] = value
	}
	if bifrostContext != nil {
		if contextHeaders, ok := bifrostContext.Value(schemas.BifrostContextKeyProviderResponseHeaders).(map[string]string); ok {
			for key, value := range contextHeaders {
				combined[key] = value
			}
		}
	}
	connectionTokens := make(map[string]struct{})
	for key, value := range combined {
		if !strings.EqualFold(key, "Connection") {
			continue
		}
		for _, token := range strings.Split(value, ",") {
			if normalized := strings.ToLower(strings.TrimSpace(token)); normalized != "" {
				connectionTokens[normalized] = struct{}{}
			}
		}
	}
	secrets := directKeySecrets(bifrostContext)
	headers := make(http.Header)
	for key, value := range combined {
		if strings.ContainsAny(value, "\r\n\x00") {
			continue
		}
		canonical := http.CanonicalHeaderKey(key)
		if unsafeProviderResponseHeader(canonical, value, secrets, connectionTokens) {
			continue
		}
		headers.Set(canonical, value)
	}
	if stream {
		headers.Set("Content-Type", "text/event-stream")
	} else if headers.Get("Content-Type") == "" {
		headers.Set("Content-Type", "application/json")
	}
	for _, canonical := range []string{
		"Openai-Request-Id", "X-Ms-Request-Id", "X-Amzn-Requestid", "X-Goog-Request-Id",
	} {
		if headers.Get("X-Request-Id") == "" && headers.Get(canonical) != "" {
			headers.Set("X-Request-Id", headers.Get(canonical))
		}
	}
	return headers
}

func directKeySecrets(bifrostContext *schemas.BifrostContext) []string {
	if bifrostContext == nil {
		return nil
	}
	switch key := bifrostContext.Value(schemas.BifrostContextKeyDirectKey).(type) {
	case schemas.Key:
		keyCopy := key
		return keySecrets(&keyCopy)
	case *schemas.Key:
		if key != nil {
			return keySecrets(key)
		}
	}
	return nil
}

func unsafeProviderResponseHeader(
	canonical string,
	value string,
	secrets []string,
	connectionTokens map[string]struct{},
) bool {
	lowered := strings.ToLower(canonical)
	if _, found := connectionTokens[strings.ToLower(canonical)]; found {
		return true
	}
	if platformheader.IsCredentialName(canonical) ||
		strings.EqualFold(canonical, "Set-Cookie") ||
		strings.EqualFold(canonical, "Set-Cookie2") ||
		strings.Contains(lowered, "secret") ||
		strings.HasPrefix(strings.ToLower(canonical), "x-bf-") ||
		strings.HasPrefix(strings.ToLower(canonical), "x-gptload-") ||
		containsSecret(value, secrets) {
		return true
	}
	switch canonical {
	case "Connection", "Proxy-Connection", "Keep-Alive", "Proxy-Authenticate",
		"Proxy-Authorization", "Te", "Trailer", "Transfer-Encoding", "Upgrade":
		return true
	default:
		return false
	}
}

func passthroughHTTPError(status int, headers http.Header, body []byte, secrets []string) *execution.ErrorEvidence {
	typeValue, codeValue := openAIErrorTypeCode(body)
	typeValue = sanitizeEvidenceValue(typeValue, secrets)
	codeValue = sanitizeEvidenceValue(codeValue, secrets)
	requestID := upstreamRequestID(headers)
	evidence := &execution.ErrorEvidence{
		Kind:       execution.ErrorKindHTTP,
		Hint:       failureHintFromHTTP(status, body),
		StatusCode: status,
		Type:       typeValue,
		Code:       codeValue,
		Summary:    errorMessageSummary(body, errorSummary(execution.ErrorKindHTTP, status), secrets),
		RequestID:  requestID,
		RetryAfter: retryAfter(headers),
		Header:     evidenceHeaders(headers),
	}
	annotateBifrostErrorEvidence(evidence)
	return evidence
}

func failureHintFromHTTP(status int, body []byte) execution.FailureHint {
	envelope := providerErrorEnvelope(body)
	return neutralFailureHint(status, envelope.typeValue, envelope.codeValue, envelope.statusValue, envelope.messageValue)
}

func neutralFailureHint(status int, values ...string) execution.FailureHint {
	markers := strings.ToLower(strings.Join(values, " "))
	switch {
	case execution.InsufficientBalanceSignal(status, values...):
		return execution.FailureHintInsufficientBalance
	case status == http.StatusUnauthorized:
		return execution.FailureHintInvalidCredential
	case containsAnyMarker(markers,
		"model_not_found", "model not found", "model_not_available",
		"model unavailable", "deployment_not_found", "unsupported_model"):
		return execution.FailureHintModelUnavailable
	case status == http.StatusTooManyRequests || containsAnyMarker(markers,
		"rate_limit", "rate limit", "too_many_requests", "quota_exceeded",
		"resource_exhausted", "throttl"):
		return execution.FailureHintRateLimited
	case containsAnyMarker(markers,
		"invalid_api_key", "api_key_invalid", "authentication_error",
		"authentication failed",
		"invalid credential", "api key not valid"):
		return execution.FailureHintInvalidCredential
	case containsAnyMarker(markers,
		"invalid_request_error", "context_length_exceeded"):
		return execution.FailureHintRequestRejected
	case containsAnyMarker(markers,
		"server_is_overloaded", "service_unavailable", "server overloaded"):
		return execution.FailureHintHostError
	case status >= http.StatusInternalServerError && status <= 599:
		return execution.FailureHintHostError
	default:
		return ""
	}
}

func annotateBifrostErrorEvidence(evidence *execution.ErrorEvidence) {
	if evidence == nil {
		return
	}
	switch evidence.Kind {
	case execution.ErrorKindTransport, execution.ErrorKindTimeout,
		execution.ErrorKindHTTP, execution.ErrorKindProvider:
		evidence.OriginHint = execution.ErrorOriginUpstream
	case execution.ErrorKindCanceled:
		evidence.OriginHint = execution.ErrorOriginDownstream
	case execution.ErrorKindInvalidRequest:
		evidence.OriginHint = execution.ErrorOriginClient
	case execution.ErrorKindConversionUnsupported, execution.ErrorKindInternal:
		evidence.OriginHint = execution.ErrorOriginInternal
	}
	switch evidence.Hint {
	case execution.FailureHintInvalidCredential,
		execution.FailureHintRefreshRequired,
		execution.FailureHintReauthorizationRequired,
		execution.FailureHintInsufficientBalance:
		evidence.ScopeHint = execution.ErrorScopeCredential
	case execution.FailureHintRequestRejected:
		evidence.OriginHint = execution.ErrorOriginClient
		evidence.ScopeHint = execution.ErrorScopeRequest
	case execution.FailureHintCandidateUnavailable,
		execution.FailureHintModelUnavailable:
		evidence.ScopeHint = execution.ErrorScopeModel
	case execution.FailureHintHostError:
		evidence.ScopeHint = execution.ErrorScopeGroup
	case execution.FailureHintRateLimited:
		// A rate-limit marker alone does not establish request, model, or
		// credential scope.
	default:
		switch evidence.Kind {
		case execution.ErrorKindTransport, execution.ErrorKindTimeout,
			execution.ErrorKindConversionUnsupported:
			evidence.ScopeHint = execution.ErrorScopeGroup
		case execution.ErrorKindCanceled, execution.ErrorKindInvalidRequest:
			evidence.ScopeHint = execution.ErrorScopeRequest
		case execution.ErrorKindHTTP:
			if evidence.StatusCode >= http.StatusBadRequest &&
				evidence.StatusCode < http.StatusInternalServerError &&
				evidence.StatusCode != http.StatusTooManyRequests {
				evidence.ScopeHint = execution.ErrorScopeRequest
			}
		}
	}
}

func containsAnyMarker(value string, markers ...string) bool {
	for _, marker := range markers {
		if strings.Contains(value, marker) {
			return true
		}
	}
	return false
}

func upstreamRequestID(headers http.Header) string {
	if headers == nil {
		return ""
	}
	if value := headers.Get("X-Request-Id"); value != "" {
		return value
	}
	return headers.Get("Request-Id")
}

func retryAfter(headers http.Header) time.Duration {
	value := strings.TrimSpace(headers.Get("Retry-After"))
	if value == "" {
		return 0
	}
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second
	}
	if deadline, err := http.ParseTime(value); err == nil {
		duration := time.Until(deadline)
		if duration > 0 {
			return duration
		}
	}
	return 0
}

func unaryErrorResult(bifrostError *schemas.BifrostError, bifrostContext *schemas.BifrostContext, secrets []string) execution.AttemptResult {
	headers := responseHeaders(nil, bifrostContext, false)
	evidence, started, status := errorEvidence(bifrostError, bifrostContext, secrets, false, 0, headers, false)
	result := execution.AttemptResult{
		DispatchState:     sdkErrorDispatchState(bifrostError, started),
		ResponseStarted:   started,
		StatusCode:        status,
		Header:            nil,
		UpstreamRequestID: "",
		Usage:             errorUsageEvidence(bifrostError, nil),
		Error:             evidence,
	}
	if started {
		result.Header = headers
		result.UpstreamRequestID = upstreamRequestID(headers)
		result.Body = encodeSafeErrorBody(evidence)
	}
	return result
}

func convertedUnaryErrorResult(
	bifrostError *schemas.BifrostError,
	bifrostContext *schemas.BifrostContext,
	secrets []string,
) execution.AttemptResult {
	result := unaryErrorResult(bifrostError, bifrostContext, secrets)
	if convertedSDKPreflightFailure(bifrostError, result.ResponseStarted) {
		if bifrostError.ExtraFields.BilledUsage == nil {
			result.DispatchState = execution.DispatchNotSent
		}
		result.Error = convertedSerializationEvidence()
	}
	return result
}

func streamErrorResult(
	bifrostError *schemas.BifrostError,
	bifrostContext *schemas.BifrostContext,
	secrets []string,
	alreadyStarted bool,
	initialStatus int,
	headers http.Header,
	model string,
	usageEvidence *execution.UsageEvidence,
) execution.StreamResult {
	if headers == nil {
		headers = responseHeaders(nil, bifrostContext, true)
	}
	evidence, started, status := errorEvidence(bifrostError, bifrostContext, secrets, alreadyStarted, initialStatus, headers, true)
	result := execution.StreamResult{
		DispatchState: sdkErrorDispatchState(bifrostError, alreadyStarted || started),
		Model:         model,
		Usage:         errorUsageEvidence(bifrostError, usageEvidence),
		Error:         evidence,
	}
	if started {
		result.ResponseStarted = true
		result.StatusCode = status
		result.Header = headers.Clone()
		result.UpstreamRequestID = upstreamRequestID(headers)
	}
	return result
}

func markPromotedStreamRejectionReplaySafe(result *execution.StreamResult) {
	if result == nil || result.Error == nil ||
		!streamCapacityRejection(result.Error.Code) {
		return
	}
	result.Error.ReplaySafety = execution.ReplaySafetyRejectedBeforeProcessing
}

func streamCapacityRejection(codeValue string) bool {
	codeValue = strings.ToLower(strings.TrimSpace(codeValue))
	return codeValue == "server_is_overloaded" || codeValue == "rate_limit_exceeded"
}

func convertedStreamErrorResult(
	bifrostError *schemas.BifrostError,
	bifrostContext *schemas.BifrostContext,
	secrets []string,
	alreadyStarted bool,
	initialStatus int,
	headers http.Header,
	model string,
	usageEvidence *execution.UsageEvidence,
) execution.StreamResult {
	result := streamErrorResult(
		bifrostError,
		bifrostContext,
		secrets,
		alreadyStarted,
		initialStatus,
		headers,
		model,
		usageEvidence,
	)
	if !alreadyStarted && convertedSDKPreflightFailure(bifrostError, result.ResponseStarted) {
		if bifrostError.ExtraFields.BilledUsage == nil {
			result.DispatchState = execution.DispatchNotSent
		}
		result.Error = convertedSerializationEvidence()
	}
	return result
}

func convertedSDKPreflightFailure(bifrostError *schemas.BifrostError, responseStarted bool) bool {
	return !responseStarted && bifrostError != nil && bifrostError.IsBifrostError &&
		strings.EqualFold(strings.TrimSpace(errorType(bifrostError)), "invalid_request_error")
}

func convertedSerializationEvidence() *execution.ErrorEvidence {
	evidence := &execution.ErrorEvidence{
		Kind:    execution.ErrorKindConversionUnsupported,
		Code:    execution.ErrorCodeTargetSerializationFailed,
		Summary: "target request serialization failed",
	}
	annotateBifrostErrorEvidence(evidence)
	return evidence
}

func sdkErrorDispatchState(bifrostError *schemas.BifrostError, responseStarted bool) execution.DispatchState {
	if responseStarted || bifrostError == nil || bifrostError.Error == nil || bifrostError.Error.Error == nil {
		return execution.DispatchMaybeSent
	}
	underlying := bifrostError.Error.Error
	if bifrostError.ExtraFields.BilledUsage != nil {
		return execution.DispatchMaybeSent
	}
	var dnsError *net.DNSError
	if errors.As(underlying, &dnsError) {
		return execution.DispatchNotSent
	}
	var operationError *net.OpError
	if errors.As(underlying, &operationError) && strings.EqualFold(operationError.Op, "dial") {
		return execution.DispatchNotSent
	}
	message := underlying.Error()
	for _, marker := range []string{
		"connection to unspecified IP ",
		"connection to link-local IP ",
		"connection to private IP ",
		"blocked connection to non-public address ",
		"no usable address resolved for ",
	} {
		if strings.Contains(message, marker) {
			return execution.DispatchNotSent
		}
	}
	return execution.DispatchMaybeSent
}

func errorEvidence(
	bifrostError *schemas.BifrostError,
	bifrostContext *schemas.BifrostContext,
	secrets []string,
	alreadyStarted bool,
	initialStatus int,
	headers http.Header,
	stream bool,
) (*execution.ErrorEvidence, bool, int) {
	if bifrostError == nil {
		evidence := &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: "execution runtime returned an invalid response"}
		if alreadyStarted {
			evidence.RequestID = upstreamRequestID(headers)
			return evidence, true, initialStatus
		}
		return evidence, false, 0
	}
	typeValue := errorType(bifrostError)
	codeValue := ""
	messageValue := ""
	if bifrostError.Error != nil && bifrostError.Error.Code != nil {
		codeValue = *bifrostError.Error.Code
	}
	if bifrostError.Error != nil {
		messageValue = bifrostError.Error.Message
	}
	typeValue = sanitizeEvidenceValue(typeValue, secrets)
	codeValue = sanitizeEvidenceValue(codeValue, secrets)
	synthetic := typeValue == schemas.RequestTimedOut || typeValue == schemas.RequestCancelled || typeValue == schemas.ProviderConnectionFailed
	status := 0
	if bifrostError.StatusCode != nil && *bifrostError.StatusCode >= 100 && *bifrostError.StatusCode <= 599 {
		status = *bifrostError.StatusCode
	}
	started := alreadyStarted || (status > 0 && !synthetic && !bifrostError.IsBifrostError)
	resultStatus := status
	if alreadyStarted {
		resultStatus = initialStatus
	} else if !started {
		resultStatus = 0
	}
	kind := classifyError(bifrostError, typeValue, started, stream)
	requestID := ""
	if started {
		requestID = upstreamRequestID(headers)
	}
	evidenceStatus := resultStatus
	if alreadyStarted && status != initialStatus {
		evidenceStatus = 0
	}
	summary := errorSummary(kind, errorSummaryStatus(status, resultStatus))
	evidence := &execution.ErrorEvidence{
		Kind:       kind,
		Hint:       neutralFailureHint(status, typeValue, codeValue, messageValue),
		StatusCode: evidenceStatus,
		Type:       typeValue,
		Code:       codeValue,
		Summary:    errorMessageSummary([]byte(messageValue), summary, secrets),
		RequestID:  requestID,
		RetryAfter: retryAfter(headers),
		Header:     evidenceHeaders(headers),
	}
	if !started {
		evidence.StatusCode = 0
		evidence.RequestID = ""
		evidence.RetryAfter = 0
		evidence.Header = nil
	}
	annotateBifrostErrorEvidence(evidence)
	return evidence, started, resultStatus
}

func errorSummaryStatus(upstreamStatus, resultStatus int) int {
	if upstreamStatus > 0 {
		return upstreamStatus
	}
	return resultStatus
}

func errorType(bifrostError *schemas.BifrostError) string {
	if bifrostError == nil {
		return ""
	}
	if bifrostError.Error != nil && bifrostError.Error.Type != nil {
		return *bifrostError.Error.Type
	}
	if bifrostError.Type != nil {
		return *bifrostError.Type
	}
	return ""
}

func classifyError(bifrostError *schemas.BifrostError, typeValue string, started bool, stream bool) execution.ErrorKind {
	if stream && bifrostError != nil && bifrostError.Error != nil &&
		errors.Is(bifrostError.Error.Error, providerUtils.ErrStreamIdleTimeout) {
		return execution.ErrorKindTimeout
	}
	switch typeValue {
	case schemas.RequestTimedOut:
		return execution.ErrorKindTimeout
	case schemas.RequestCancelled:
		return execution.ErrorKindCanceled
	case schemas.ProviderConnectionFailed:
		return execution.ErrorKindTransport
	case "invalid_request_error":
		if !started && bifrostError != nil && bifrostError.IsBifrostError {
			return execution.ErrorKindInvalidRequest
		}
	}
	if started && bifrostError != nil && bifrostError.StatusCode != nil && *bifrostError.StatusCode >= 400 {
		return execution.ErrorKindHTTP
	}
	if bifrostError != nil && bifrostError.IsBifrostError {
		return execution.ErrorKindInternal
	}
	return execution.ErrorKindProvider
}

func errorSummary(kind execution.ErrorKind, status int) string {
	switch kind {
	case execution.ErrorKindTransport:
		return "upstream transport failed"
	case execution.ErrorKindTimeout:
		return "upstream request timed out"
	case execution.ErrorKindCanceled:
		return "upstream request canceled"
	case execution.ErrorKindHTTP:
		return fmt.Sprintf("upstream returned HTTP %d", status)
	case execution.ErrorKindProvider:
		return "upstream provider failed"
	case execution.ErrorKindInvalidRequest:
		return "upstream rejected the request"
	case execution.ErrorKindConversionUnsupported:
		return "target protocol conversion is not supported"
	default:
		return "execution adapter failed"
	}
}

func keySecrets(key *schemas.Key) []string {
	if key == nil {
		return nil
	}
	seen := make(map[string]struct{})
	result := make([]string, 0, 12)
	appendSecret := func(secret *schemas.SecretVar) {
		if secret == nil {
			return
		}
		value := secret.GetValue()
		if value == "" {
			return
		}
		if _, exists := seen[value]; exists {
			return
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	appendSecret(&key.Value)
	if config := key.AzureKeyConfig; config != nil {
		appendSecret(config.ClientID)
		appendSecret(config.ClientSecret)
		appendSecret(config.TenantID)
	}
	if config := key.BedrockKeyConfig; config != nil {
		appendSecret(&config.AccessKey)
		appendSecret(&config.SecretKey)
		appendSecret(config.SessionToken)
		appendSecret(config.RoleARN)
		appendSecret(config.ExternalID)
		appendSecret(config.RoleSessionName)
	}
	if config := key.VertexKeyConfig; config != nil {
		appendSecret(&config.AuthCredentials)
	}
	return result
}

func notSentUnaryFailure(kind execution.ErrorKind, summary string) execution.AttemptResult {
	return execution.AttemptResult{
		DispatchState: execution.DispatchNotSent,
		Error:         &execution.ErrorEvidence{Kind: kind, Summary: summary},
	}
}

func notSentConversionFailure(code, summary string) execution.AttemptResult {
	return execution.AttemptResult{
		DispatchState: execution.DispatchNotSent,
		Error: &execution.ErrorEvidence{
			Kind:    execution.ErrorKindConversionUnsupported,
			Code:    code,
			Summary: summary,
		},
	}
}

func attemptedUnaryFailure(kind execution.ErrorKind, summary string) execution.AttemptResult {
	return execution.AttemptResult{
		DispatchState: execution.DispatchMaybeSent,
		Error:         &execution.ErrorEvidence{Kind: kind, Summary: summary},
	}
}

func startedUnaryFailure(status int, headers http.Header, kind execution.ErrorKind, summary string) execution.AttemptResult {
	requestID := upstreamRequestID(headers)
	return execution.AttemptResult{
		DispatchState:     execution.DispatchMaybeSent,
		ResponseStarted:   true,
		StatusCode:        status,
		Header:            headers.Clone(),
		UpstreamRequestID: requestID,
		Error: &execution.ErrorEvidence{
			Kind:      kind,
			Summary:   summary,
			RequestID: requestID,
		},
	}
}

func notSentStreamFailure(kind execution.ErrorKind, summary string) execution.StreamResult {
	return execution.StreamResult{
		DispatchState: execution.DispatchNotSent,
		Error:         &execution.ErrorEvidence{Kind: kind, Summary: summary},
	}
}

func attemptedStreamFailure(kind execution.ErrorKind, summary string) execution.StreamResult {
	return execution.StreamResult{
		DispatchState: execution.DispatchMaybeSent,
		Error:         &execution.ErrorEvidence{Kind: kind, Summary: summary},
	}
}

func streamFromAttemptFailure(failure execution.AttemptResult) execution.StreamResult {
	return execution.StreamResult{
		DispatchState: failure.DispatchState,
		Error:         cloneError(failure.Error),
	}
}

func streamSinkFailure(headers http.Header, requestID, model string, usageEvidence *execution.UsageEvidence) execution.StreamResult {
	return execution.StreamResult{
		DispatchState:     execution.DispatchMaybeSent,
		ResponseStarted:   true,
		StatusCode:        http.StatusOK,
		Header:            headers.Clone(),
		Model:             model,
		UpstreamRequestID: requestID,
		Usage:             cloneUsage(usageEvidence),
		Error: &execution.ErrorEvidence{
			Kind:      execution.ErrorKindCanceled,
			Summary:   "stream consumer stopped",
			RequestID: requestID,
		},
	}
}

func cloneUsage(source *execution.UsageEvidence) *execution.UsageEvidence {
	if source == nil {
		return nil
	}
	clone := source.Clone()
	return &clone
}

func cloneError(source *execution.ErrorEvidence) *execution.ErrorEvidence {
	if source == nil {
		return nil
	}
	clone := source.Clone()
	return &clone
}
