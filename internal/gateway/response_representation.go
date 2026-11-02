package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"reflect"
	"strconv"

	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/contentcoding"
	"gpt-load/internal/platform/redact"
	"gpt-load/internal/protocol"
)

type preparedSuccessRepresentation struct {
	headers          http.Header
	downstream       []byte
	inspectable      []byte
	modelObservation responseModelObservation
	changed          bool
}

type preparedErrorRepresentation struct {
	headers        http.Header
	downstream     []byte
	classification []byte
	changed        bool
}

type responseProcessor struct {
	redactor *redact.Redactor
}

func (forwarder *responseProcessor) prepareSuccessRepresentation(
	input ForwardInput,
	status int,
	headers http.Header,
	wire []byte,
	secrets []string,
) (preparedSuccessRepresentation, error) {
	bodyLimit := execution.UnaryResponseBodyLimit(input.ClientProtocol)
	if int64(len(wire)) > bodyLimit {
		return preparedSuccessRepresentation{}, successRepresentationProtocolError("wire body exceeds limit")
	}

	encoding, err := contentcoding.ParseContentEncoding(
		headerFieldValues(headers, "Content-Encoding"),
	)
	if err != nil {
		return preparedSuccessRepresentation{}, successRepresentationProtocolError("unsupported or malformed Content-Encoding")
	}
	opaqueRepresentation := input.ClientProtocol == protocol.OpenAIImages ||
		embeddingsProtocol(input.ClientProtocol) || input.ClientProtocol == protocol.Rerank
	var originalPlain []byte
	if opaqueRepresentation && encoding == contentcoding.Identity {
		// The buffered attempt result owns wire for the duration of this terminal
		// representation step. Keep the identity bytes instead of copying a large
		// base64 payload before exact credential inspection.
		originalPlain = wire
	} else {
		originalPlain, err = contentcoding.DecodeLimited(
			encoding,
			wire,
			bodyLimit,
		)
		if err != nil {
			return preparedSuccessRepresentation{}, successRepresentationProtocolError("decompress response body")
		}
	}
	modelTracker := newResponseModelTracker(input.Dialect, input.UpstreamModelID)
	modelTracker.observe(originalPlain)

	var safePlain []byte
	if opaqueRepresentation {
		if opaqueCredentialLiteralsRemain(input.ClientProtocol, originalPlain, secrets) {
			return preparedSuccessRepresentation{}, successRepresentationProtocolError("credential remains in response body")
		}
		safePlain = originalPlain
	} else {
		patternSafePlain := forwarder.redactor.Bytes(originalPlain)
		if int64(len(patternSafePlain)) > bodyLimit {
			return preparedSuccessRepresentation{}, successRepresentationProtocolError("redacted response body exceeds limit")
		}
		var residualCredential bool
		var ok bool
		safePlain, residualCredential, ok = redactCredentialLiterals(
			patternSafePlain,
			secrets,
			bodyLimit,
		)
		if !ok {
			return preparedSuccessRepresentation{}, successRepresentationProtocolError("redact response body")
		}
		if residualCredential {
			return preparedSuccessRepresentation{}, successRepresentationProtocolError("credential remains in response body")
		}
	}
	if input.ResponsesStoreDowngraded {
		safePlain, err = normalizeStatelessResponsesSuccess(safePlain)
		if err != nil {
			return preparedSuccessRepresentation{}, successRepresentationProtocolError(
				"normalize stateless Responses response",
			)
		}
	}

	inspectablePlain := safePlain
	if !opaqueRepresentation {
		inspectablePlain = bytes.Clone(safePlain)
	}
	downstreamPlain := safePlain
	if needsModelRewrite(input) {
		rewriteModel := true
		if embeddingsProtocol(input.ClientProtocol) {
			if required, valid := embeddingsResponseModelRewriteRequired(
				safePlain,
				input.ExternalModel,
			); valid {
				rewriteModel = required
			}
		}
		if rewriteModel {
			rewriter, supported := input.Dialect.(dialect.ModelRewriter)
			if !supported {
				return preparedSuccessRepresentation{}, successRepresentationProtocolError("dialect does not support model rewrite")
			}
			downstreamPlain, err = rewriter.RewriteResponseModel(safePlain, input.ExternalModel)
			if err != nil {
				return preparedSuccessRepresentation{}, successRepresentationProtocolError("rewrite response model")
			}
			if int64(len(downstreamPlain)) > bodyLimit {
				return preparedSuccessRepresentation{}, successRepresentationProtocolError("rewritten response body exceeds limit")
			}
			var credentialRemains bool
			if opaqueRepresentation {
				credentialRemains = opaqueCredentialLiteralsRemain(
					input.ClientProtocol,
					downstreamPlain,
					secrets,
				)
			} else {
				credentialRemains = credentialLiteralsRemain(downstreamPlain, secrets)
			}
			if credentialRemains {
				return preparedSuccessRepresentation{}, successRepresentationProtocolError("credential remains after model rewrite")
			}
		}
	}

	// The execution adapter may already have projected the selected upstream
	// model back to the client alias. Conservatively treat every aliased route
	// as a representation rewrite so upstream validators/digests are never
	// forwarded for bytes whose identity changed before this boundary.
	changed := encoding != contentcoding.Identity || needsModelRewrite(input) ||
		!bytes.Equal(originalPlain, downstreamPlain)
	if status == http.StatusPartialContent && changed {
		return preparedSuccessRepresentation{}, successRepresentationProtocolError("changed partial response")
	}

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
		return preparedSuccessRepresentation{}, successRepresentationProtocolError("credential collision in rebuilt representation metadata")
	}
	if status == http.StatusPartialContent && !hasSafeContentRange(preparedHeaders, len(downstreamPlain)) {
		return preparedSuccessRepresentation{}, successRepresentationProtocolError("partial response lacks safe Content-Range")
	}

	downstream := downstreamPlain
	if !embeddingsProtocol(input.ClientProtocol) {
		downstream = bytes.Clone(downstreamPlain)
	}
	return preparedSuccessRepresentation{
		headers:          preparedHeaders,
		downstream:       downstream,
		inspectable:      inspectablePlain,
		modelObservation: modelTracker.observation(),
		changed:          changed,
	}, nil
}

func embeddingsResponseModelRewriteRequired(body []byte, model string) (bool, bool) {
	var envelope struct {
		Model json.RawMessage `json:"model"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return false, false
	}
	raw := bytes.TrimSpace(envelope.Model)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return false, true
	}
	var current string
	if err := json.Unmarshal(raw, &current); err != nil {
		return false, false
	}
	return current != model, true
}
