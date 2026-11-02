package gateway

import (
	"bytes"
	"net/http"
	"reflect"
	"strconv"

	"gpt-load/internal/platform/contentcoding"
	"gpt-load/internal/platform/redact"
)

func (forwarder *responseProcessor) prepareErrorRepresentation(
	input ForwardInput,
	headers http.Header,
	wire []byte,
	secrets []string,
) preparedErrorRepresentation {
	if int64(len(wire)) > maxErrorResponseBodyBytes {
		return forwarder.failClosedErrorRepresentation(input, headers, secrets)
	}

	encoding, err := contentcoding.ParseContentEncoding(
		headerFieldValues(headers, "Content-Encoding"),
	)
	if err != nil {
		return forwarder.failClosedErrorRepresentation(input, headers, secrets)
	}
	originalPlain, err := contentcoding.DecodeLimited(
		encoding,
		wire,
		maxDecompressedErrorBodyBytes,
	)
	if err != nil {
		return forwarder.failClosedErrorRepresentation(input, headers, secrets)
	}

	patternSafePlain := forwarder.redactor.Bytes(originalPlain)
	if int64(len(patternSafePlain)) > maxDecompressedErrorBodyBytes {
		return forwarder.failClosedErrorRepresentation(input, headers, secrets)
	}
	safePlain, residualCredential, ok := redactCredentialLiterals(
		patternSafePlain,
		secrets,
		maxDecompressedErrorBodyBytes,
	)
	if !ok || residualCredential {
		return forwarder.failClosedErrorRepresentation(input, headers, secrets)
	}

	classificationPlain := bytes.Clone(safePlain)
	downstreamPlain := safePlain
	if needsModelRewrite(input) {
		downstreamPlain, ok = rewriteBoundedLiteral(
			safePlain,
			input.UpstreamModelID,
			input.ExternalModel,
			maxDecompressedErrorBodyBytes,
		)
		if !ok || credentialLiteralsRemain(downstreamPlain, secrets) {
			return forwarder.failClosedErrorRepresentation(input, headers, secrets)
		}
	}

	changed := encoding != contentcoding.Identity ||
		!bytes.Equal(originalPlain, downstreamPlain)
	preparedHeaders := headers.Clone()
	if preparedHeaders == nil {
		preparedHeaders = make(http.Header)
	}
	representationHeadersChanged := len(headerFieldValues(preparedHeaders, "Content-Encoding")) > 0 ||
		!hasSingleHeaderFieldValue(
			preparedHeaders,
			"Content-Length",
			strconv.Itoa(len(downstreamPlain)),
		)
	deleteHeaderField(preparedHeaders, "Content-Encoding")
	if changed {
		invalidateRewrittenBodyHeaders(preparedHeaders)
	}
	beforeSanitize := preparedHeaders.Clone()
	preparedHeaders = sanitizeForwardResponseHeaders(preparedHeaders, input, secrets...)
	representationHeadersChanged = representationHeadersChanged ||
		!reflect.DeepEqual(beforeSanitize, preparedHeaders)
	deleteHeaderField(preparedHeaders, "Content-Encoding")
	deleteHeaderField(preparedHeaders, "Content-Length")
	preparedHeaders.Set("Content-Length", strconv.Itoa(len(downstreamPlain)))
	if representationHeadersChanged {
		deleteHeaderField(preparedHeaders, "Signature")
		deleteHeaderField(preparedHeaders, "Signature-Input")
	}
	if representationMetadataContainsSecrets(preparedHeaders, secrets) {
		return forwarder.failClosedErrorRepresentation(input, headers, secrets)
	}

	return preparedErrorRepresentation{
		headers:        preparedHeaders,
		downstream:     bytes.Clone(downstreamPlain),
		classification: classificationPlain,
		changed:        changed,
	}
}

func (forwarder *responseProcessor) failClosedErrorRepresentation(
	input ForwardInput,
	headers http.Header,
	secrets []string,
) preparedErrorRepresentation {
	preparedHeaders := headers.Clone()
	if preparedHeaders == nil {
		preparedHeaders = make(http.Header)
	}
	deleteHeaderField(preparedHeaders, "Content-Encoding")
	invalidateRewrittenBodyHeaders(preparedHeaders)
	preparedHeaders = sanitizeForwardResponseHeaders(preparedHeaders, input, secrets...)
	deleteHeaderField(preparedHeaders, "Content-Encoding")
	invalidateRewrittenBodyHeaders(preparedHeaders)
	body := []byte(redact.Placeholder)
	preparedHeaders.Set("Content-Type", "text/plain; charset=utf-8")
	preparedHeaders.Set("Content-Length", strconv.Itoa(len(body)))
	return preparedErrorRepresentation{
		headers:        preparedHeaders,
		downstream:     bytes.Clone(body),
		classification: bytes.Clone(body),
		changed:        true,
	}
}
