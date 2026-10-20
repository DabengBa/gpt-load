package bifrost

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

const (
	// Bifrost 1.8.4 compares a compressed Content-Length before decoding. Force
	// every Embeddings success body through its decoded streaming reader, then
	// enforce GPT-Load's protocol limit while materializing the raw JSON once.
	embeddingDecodedResponseThresholdBytes = int64(1)
	embeddingDecodedResponsePrefetchBytes  = 2
)

func buildEmbeddingRequest(
	spec execution.AttemptSpec,
	provider schemas.ModelProvider,
) (*schemas.BifrostEmbeddingRequest, error) {
	sanitized, err := dialect.NewOpenAIEmbeddings().SanitizeRequestForAttempt(
		&dialect.ParsedRequest{
			Method: spec.Method, Path: spec.Path, RawQuery: spec.RawQuery,
			Header: spec.Header.Clone(), Body: bytes.Clone(spec.Body),
		},
		spec.UpstreamModel,
	)
	if err != nil {
		return nil, err
	}
	input, err := embeddingInputMarker(sanitized.Body)
	if err != nil {
		return nil, err
	}
	return &schemas.BifrostEmbeddingRequest{
		Provider:       provider,
		Model:          spec.UpstreamModel,
		Input:          *input,
		Fallbacks:      nil,
		RawRequestBody: bytes.Clone(sanitized.Body),
	}, nil
}

// embeddingInputMarker preserves only the validated input shape needed by
// Bifrost's typed preflight. The selected provider receives RawRequestBody, so
// GPT-Load does not impose Go integer bounds on token IDs or rebuild the wire.
//
// Bifrost v1.13.0 replaced the old text/texts/embedding/embeddings union with an
// item list, so every shape becomes the one equivalent item: a string input or a
// text array is one text part, a token array is one token part, and a token matrix
// is one token part per row.
//
// Every shape that the item list cannot describe — an empty selection and a token
// id outside Go's integer range — is reported as the placeholder item
// placeholderEmbeddingInput returns. Bifrost rejects a zero-length input before
// dispatch, so returning no items would refuse a request the dialect validator
// already accepted; the raw body, not the marker, decides what the provider
// receives.
func embeddingInputMarker(body []byte) (*schemas.EmbeddingInput, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, fmt.Errorf("decode Embeddings request")
	}
	raw := bytes.TrimSpace(object["input"])
	if len(raw) == 0 {
		return nil, fmt.Errorf("Embeddings input is required")
	}
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, fmt.Errorf("decode Embeddings text input")
		}
		if text == "" {
			return placeholderEmbeddingInput(), nil
		}
		return &schemas.EmbeddingInput{embeddingTextItem([]string{text})}, nil
	}
	var values []json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil || values == nil {
		return nil, fmt.Errorf("decode Embeddings array input")
	}
	if len(values) == 0 {
		return placeholderEmbeddingInput(), nil
	}
	first := bytes.TrimSpace(values[0])
	if len(first) == 0 {
		return nil, fmt.Errorf("decode Embeddings array input")
	}
	switch first[0] {
	case '"':
		var texts []string
		if err := json.Unmarshal(raw, &texts); err != nil {
			return nil, fmt.Errorf("decode Embeddings text array")
		}
		// An empty text selects nothing the item list can describe, and an empty
		// text part fails Bifrost's content validation before dispatch.
		if countNonEmptyTexts(texts) != len(texts) || len(texts) == 0 {
			return placeholderEmbeddingInput(), nil
		}
		return &schemas.EmbeddingInput{embeddingTextItem(texts)}, nil
	case '[':
		var rows [][]int
		// A row without tokens produces an empty token part, which Bifrost rejects
		// before dispatch, so the whole selection is described by the placeholder.
		if err := json.Unmarshal(raw, &rows); err != nil || len(rows) == 0 ||
			countNonEmptyTokenRows(rows) != len(rows) {
			return placeholderEmbeddingInput(), nil
		}
		items := make([]schemas.EmbeddingInputItem, len(rows))
		for index, row := range rows {
			items[index] = schemas.EmbeddingInputItem{Content: schemas.EmbeddingContent{{
				Type:   schemas.EmbeddingContentPartTypeTokens,
				Tokens: row,
			}}}
		}
		input := schemas.EmbeddingInput(items)
		return &input, nil
	default:
		var tokens []int
		if err := json.Unmarshal(raw, &tokens); err != nil || len(tokens) == 0 {
			return placeholderEmbeddingInput(), nil
		}
		return &schemas.EmbeddingInput{{Content: schemas.EmbeddingContent{{
			Type:   schemas.EmbeddingContentPartTypeTokens,
			Tokens: tokens,
		}}}}, nil
	}
}

// placeholderEmbeddingInput is the marker for a selection the item list cannot
// describe. It carries one text part holding a single space: non-empty, so
// Bifrost's length and content checks accept it, and never reaching a provider
// because the raw body is the wire authority.
//
// A space rather than an empty string keeps the part valid under
// EmbeddingContentPart.Validate, which rejects an empty text payload.
func placeholderEmbeddingInput() *schemas.EmbeddingInput {
	placeholder := " "
	return &schemas.EmbeddingInput{{
		Content: schemas.EmbeddingContent{{
			Type: schemas.EmbeddingContentPartTypeText,
			Text: &placeholder,
		}},
	}}
}

// countNonEmptyTexts reports how many texts carry content. Bifrost's content
// validation requires a non-empty text payload, so an empty text cannot be
// described by a text part.
func countNonEmptyTexts(texts []string) int {
	count := 0
	for _, text := range texts {
		if text != "" {
			count++
		}
	}
	return count
}

// countNonEmptyTokenRows reports how many matrix rows carry tokens, mirroring
// Bifrost's requirement that a token part hold at least one token id.
func countNonEmptyTokenRows(rows [][]int) int {
	count := 0
	for _, row := range rows {
		if len(row) > 0 {
			count++
		}
	}
	return count
}

// embeddingTextItem describes a text selection as the single text part the
// v1.13.0 item list expresses it with. The text is kept verbatim in the part so
// the marker can never disagree with the raw body about what was selected.
func embeddingTextItem(texts []string) schemas.EmbeddingInputItem {
	return schemas.EmbeddingInputItem{Content: schemas.EmbeddingContent{{
		Type: schemas.EmbeddingContentPartTypeText,
		Text: schemas.Ptr(strings.Join(texts, "\n")),
	}}}
}

func embeddingTypedTarget(
	providerKind channel.ProviderKind,
	targetConfig json.RawMessage,
	rawQuery string,
) (string, error) {
	resourcePath := "/v1/embeddings"
	baseURL, configured, err := targetBaseURL(targetConfig)
	if err != nil {
		return "", err
	}
	switch providerKind {
	case channel.ProviderOpenAI:
	case channel.ProviderMultiProtocolGateway:
		if !configured {
			return "", fmt.Errorf("multi-protocol gateway Embeddings target is required")
		}
		return resolveMultiProtocolGatewayTargetURL(baseURL, resourcePath, rawQuery)
	case channel.ProviderOpenRouter:
		// OpenRouter concatenates the configured provider base URL with the
		// context path, so the per-request override must remain relative.
		return appendTypedQuery(resourcePath, rawQuery), nil
	case channel.ProviderOpenAICompatible:
		if !configured {
			return "", fmt.Errorf("compatible Embeddings target is required")
		}
		resourcePath = "/embeddings"
	default:
		return "", fmt.Errorf("unsupported Embeddings provider")
	}
	if configured {
		return resolveTypedTargetURL(baseURL, resourcePath, rawQuery)
	}
	return appendTypedQuery(resourcePath, rawQuery), nil
}

func enableEmbeddingWireFidelity(ctx *schemas.BifrostContext) {
	if ctx == nil {
		return
	}
	ctx.SetValue(schemas.BifrostContextKeyAllowPerRequestRawOverride, true)
	ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
	ctx.SetValue(schemas.BifrostContextKeySendBackRawResponse, true)
	ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
	ctx.SetValue(
		schemas.BifrostContextKeyLargeResponseThreshold,
		embeddingDecodedResponseThresholdBytes,
	)
	ctx.SetValue(
		schemas.BifrostContextKeyLargePayloadPrefetchSize,
		embeddingDecodedResponsePrefetchBytes,
	)
}

func (r *Runtime) executeEmbedding(
	parent context.Context,
	spec execution.AttemptSpec,
	prepared preparedAttempt,
) execution.AttemptResult {
	requestContext, requestCancel := boundedRequestContext(parent, spec.Timeouts.Request)
	defer requestCancel()
	callContext, callCancel := context.WithCancel(requestContext)
	defer callCancel()

	bifrostContext := r.newSDKContext(callContext, spec, prepared.directKey)
	enableEmbeddingWireFidelity(bifrostContext)
	setTypedRequestURL(bifrostContext, prepared.typedURL)
	outcomeChannel := make(chan embeddingSDKResult, 1)
	go func() {
		response, bifrostError := r.core.EmbeddingRequest(bifrostContext, prepared.embeddingRequest)
		outcomeChannel <- embeddingSDKResult{response: response, err: bifrostError}
	}()

	var outcome embeddingSDKResult
	select {
	case outcome = <-outcomeChannel:
		if requestContext.Err() != nil {
			return unaryContextFailure(requestContext, false)
		}
	case <-callContext.Done():
		return unaryContextFailure(requestContext, false)
	}
	if outcome.err != nil {
		result := unaryErrorResult(outcome.err, bifrostContext, prepared.secrets)
		clearEmbeddingErrorRaw(outcome.err)
		return result
	}
	var materializedHeaders http.Header
	if large, _ := bifrostContext.Value(schemas.BifrostContextKeyLargeResponseMode).(bool); large {
		materializedHeaders = responseHeaders(nil, bifrostContext, false)
		body, tooLarge, materializeErr := materializeEmbeddingLargeResponse(
			bifrostContext,
			r.unaryResponseBodyLimit(spec),
		)
		if requestContext.Err() != nil {
			return unaryContextFailure(requestContext, false)
		}
		if tooLarge {
			return startedUnaryFailure(
				http.StatusOK,
				materializedHeaders,
				execution.ErrorKindInternal,
				"upstream response exceeds size limit",
			)
		}
		if materializeErr != nil {
			return startedUnaryFailure(
				http.StatusOK,
				materializedHeaders,
				execution.ErrorKindProvider,
				"read upstream Embeddings response",
			)
		}
		var headerErr error
		materializedHeaders, headerErr = normalizeMaterializedEmbeddingHeaders(
			materializedHeaders,
			len(body),
		)
		if headerErr != nil {
			return startedUnaryFailure(
				http.StatusOK,
				materializedHeaders,
				execution.ErrorKindProvider,
				"unsupported Embeddings response encoding",
			)
		}
		decoded, err := materializedEmbeddingResponse(body)
		if err != nil {
			return startedUnaryFailure(
				http.StatusOK,
				materializedHeaders,
				execution.ErrorKindProvider,
				"decode upstream Embeddings response",
			)
		}
		outcome.response = decoded
	}
	if outcome.response == nil {
		return attemptedUnaryFailure(execution.ErrorKindInternal, "execution runtime returned no Embeddings response")
	}
	body, ok := takeRawEmbeddingResponse(outcome.response)
	headers := materializedHeaders
	if headers == nil {
		headers = responseHeaders(outcome.response.ExtraFields.ProviderResponseHeaders, bifrostContext, false)
	}
	if !ok {
		return startedUnaryFailure(
			http.StatusOK,
			headers,
			execution.ErrorKindInternal,
			"execution runtime returned no raw Embeddings response",
		)
	}
	// The raw body is the response authority. Clear typed data defensively before
	// alias projection and gateway processing.
	outcome.response.Data = nil
	model := openAIResponseModel(body, "")
	if needsClientModelAlias(spec) {
		var err error
		body, err = rewriteClientResponseModel(spec.ClientProtocol, body, spec.ClientModel)
		if err != nil {
			return startedUnaryFailure(http.StatusOK, headers, execution.ErrorKindInternal, "rewrite Embeddings response model")
		}
	}
	return execution.AttemptResult{
		DispatchState:     execution.DispatchMaybeSent,
		ResponseStarted:   true,
		StatusCode:        http.StatusOK,
		Header:            headers,
		Body:              body,
		Model:             model,
		UpstreamRequestID: upstreamRequestID(headers),
	}
}

func materializedEmbeddingResponse(body []byte) (*schemas.BifrostEmbeddingResponse, error) {
	if !json.Valid(body) {
		return nil, fmt.Errorf("invalid Embeddings response JSON")
	}
	response := &schemas.BifrostEmbeddingResponse{}
	response.ExtraFields.RawResponse = json.RawMessage(body)
	return response, nil
}

func materializeEmbeddingLargeResponse(
	ctx *schemas.BifrostContext,
	limit int64,
) (body []byte, tooLarge bool, err error) {
	if ctx == nil || limit <= 0 {
		return nil, false, fmt.Errorf("invalid Embeddings response limit")
	}
	reader, ok := ctx.Value(schemas.BifrostContextKeyLargeResponseReader).(*providerUtils.LargeResponseReader)
	if !ok || reader == nil || reader.Resp == nil {
		return nil, false, fmt.Errorf("missing Embeddings response reader")
	}
	stopCancellation := providerUtils.SetupStreamCancellation(ctx, reader.Resp.BodyStream(), nil)
	defer func() {
		stopCancellation()
		if closeErr := reader.Close(); err == nil && closeErr != nil {
			err = closeErr
		}
		clearEmbeddingLargeResponseContext(ctx)
	}()

	body, err = io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		abortEmbeddingLargeResponse(ctx, reader, err)
		return nil, false, err
	}
	if int64(len(body)) > limit {
		abortEmbeddingLargeResponse(
			ctx,
			reader,
			fmt.Errorf("Embeddings response exceeds size limit"),
		)
		return nil, true, nil
	}
	return body, false, nil
}

func abortEmbeddingLargeResponse(
	ctx *schemas.BifrostContext,
	reader *providerUtils.LargeResponseReader,
	cause error,
) {
	if ctx == nil || reader == nil || reader.Resp == nil {
		return
	}
	if closed, _ := ctx.GetAndSetValue(
		schemas.BifrostContextKeyConnectionClosed,
		true,
	).(bool); closed {
		return
	}
	bodyStream := reader.Resp.BodyStream()
	if closer, ok := bodyStream.(interface{ CloseWithError(error) error }); ok {
		_ = closer.CloseWithError(cause)
		return
	}
	if closer, ok := bodyStream.(io.Closer); ok {
		_ = closer.Close()
	}
}

func clearEmbeddingLargeResponseContext(ctx *schemas.BifrostContext) {
	if ctx == nil {
		return
	}
	for _, key := range []schemas.BifrostContextKey{
		schemas.BifrostContextKeyLargeResponseMode,
		schemas.BifrostContextKeyLargeResponseReader,
		schemas.BifrostContextKeyLargeResponseContentLength,
		schemas.BifrostContextKeyLargeResponseContentType,
		schemas.BifrostContextKeyLargePayloadResponsePreview,
	} {
		ctx.ClearValue(key)
	}
}

func normalizeMaterializedEmbeddingHeaders(
	headers http.Header,
	bodyLength int,
) (http.Header, error) {
	result := headers.Clone()
	if result == nil {
		result = make(http.Header)
	}
	// The Bifrost decoded reader does not retain a reliable signal that the
	// upstream representation was compressed. Treat every materialized body as
	// changed so stale validators can never describe different wire bytes.
	for _, name := range []string{
		"ETag", "Digest", "Content-MD5", "Content-Range",
		"Content-Digest", "Repr-Digest", "Signature", "Signature-Input",
	} {
		result.Del(name)
	}
	encoding := strings.ToLower(strings.TrimSpace(result.Get("Content-Encoding")))
	switch encoding {
	case "", "identity":
		result.Del("Content-Encoding")
	case "gzip":
		result.Del("Content-Encoding")
	default:
		return result, fmt.Errorf("unsupported Content-Encoding %q", encoding)
	}
	result.Del("Content-Length")
	result.Set("Content-Length", strconv.Itoa(bodyLength))
	return result, nil
}

func takeRawEmbeddingResponse(response *schemas.BifrostEmbeddingResponse) ([]byte, bool) {
	if response == nil {
		return nil, false
	}
	raw := response.ExtraFields.RawResponse
	response.ExtraFields.RawResponse = nil
	switch value := raw.(type) {
	case json.RawMessage:
		return value, len(value) > 0
	case []byte:
		return value, len(value) > 0
	case string:
		return []byte(value), value != ""
	default:
		return nil, false
	}
}

func clearEmbeddingErrorRaw(bifrostError *schemas.BifrostError) {
	if bifrostError == nil {
		return
	}
	bifrostError.ExtraFields.RawRequest = nil
	bifrostError.ExtraFields.RawResponse = nil
}

func normalizeEmbeddingsAttemptResult(spec execution.AttemptSpec, result *execution.AttemptResult) {
	if result == nil || (spec.ClientProtocol != protocol.OpenAIEmbeddings && spec.ClientProtocol != protocol.GeminiEmbeddings) {
		return
	}
	if result.Error != nil && result.Error.ReplaySafety == "" {
		result.Error.ReplaySafety = execution.ReplaySafetyUnknown
	}
}
