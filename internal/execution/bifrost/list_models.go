package bifrost

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/providers/gemini"
	"github.com/maximhq/bifrost/core/providers/openai"
	"github.com/maximhq/bifrost/core/schemas"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

type listModelsSDKResult struct {
	response *schemas.BifrostListModelsResponse
	err      *schemas.BifrostError
}

func (r *Runtime) executeListModels(
	parent context.Context,
	spec execution.AttemptSpec,
	prepared preparedAttempt,
) execution.AttemptResult {
	requestContext, requestCancel := boundedRequestContext(parent, spec.Timeouts.Request)
	defer requestCancel()
	callContext, callCancel := context.WithCancel(requestContext)
	defer callCancel()

	bifrostContext := r.newSDKContext(callContext, spec, prepared.directKey)
	setTypedRequestURL(bifrostContext, prepared.typedURL)
	outcomeChannel := make(chan listModelsSDKResult, 1)
	go func() {
		response, bifrostError := r.core.ListModelsRequest(bifrostContext, prepared.listModelsRequest)
		outcomeChannel <- listModelsSDKResult{response: response, err: bifrostError}
	}()

	var outcome listModelsSDKResult
	select {
	case outcome = <-outcomeChannel:
		if requestContext.Err() != nil {
			return unaryContextFailure(requestContext, false)
		}
	case <-callContext.Done():
		return unaryContextFailure(requestContext, false)
	}
	if outcome.err != nil {
		return unaryErrorResult(outcome.err, bifrostContext, prepared.secrets)
	}
	if failure := largeUnaryResponseFailure(bifrostContext); failure != nil {
		return *failure
	}
	if outcome.response == nil {
		return attemptedUnaryFailure(execution.ErrorKindInternal, "execution runtime returned no model list")
	}
	body, err := encodeListModelsResponse(prepared.clientProtocol, prepared.provider, outcome.response)
	if err != nil {
		return startedUnaryFailure(http.StatusOK, responseHeaders(outcome.response.ExtraFields.ProviderResponseHeaders, bifrostContext, false), execution.ErrorKindInternal, "encode model list")
	}
	headers := responseHeaders(outcome.response.ExtraFields.ProviderResponseHeaders, bifrostContext, false)
	return execution.AttemptResult{
		DispatchState:     execution.DispatchMaybeSent,
		ResponseStarted:   true,
		StatusCode:        http.StatusOK,
		Header:            headers,
		Body:              body,
		UpstreamRequestID: upstreamRequestID(headers),
	}
}

func encodeListModelsResponse(clientProtocol protocol.Protocol, provider schemas.ModelProvider, response *schemas.BifrostListModelsResponse) ([]byte, error) {
	response = sanitizeListModelsResponse(provider, response)
	var wire any
	switch clientProtocol {
	case protocol.OpenAICompletions:
		converted := openai.ToOpenAIListModelsResponse(response)
		if converted != nil {
			converted.Object = "list"
		}
		wire = converted
	case protocol.Anthropic:
		wire = anthropic.ToAnthropicListModelsResponse(response)
	case protocol.Gemini:
		wire = gemini.ToGeminiListModelsResponse(response)
	default:
		return nil, fmt.Errorf("unsupported model list protocol")
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("marshal model list")
	}
	return body, nil
}

func sanitizeListModelsResponse(provider schemas.ModelProvider, response *schemas.BifrostListModelsResponse) *schemas.BifrostListModelsResponse {
	if response == nil {
		return nil
	}
	clone := *response
	switch provider {
	case schemas.OpenRouter:
		clone.Data = withoutOpenRouterUnservedModels(response.Data)
	default:
		clone.Data = append([]schemas.Model(nil), response.Data...)
	}
	prefix := string(provider) + "/"
	for index := range clone.Data {
		clone.Data[index].ID = strings.TrimPrefix(clone.Data[index].ID, prefix)
	}
	clone.ExtraFields = schemas.BifrostResponseExtraFields{}
	clone.KeyStatuses = nil
	return &clone
}

// withoutOpenRouterUnservedModels keeps GPT-Load's published model list to the
// models whose declared output modalities its proxies serve: chat and
// embeddings. Bifrost v1.13.0 merges OpenRouter's decision catalog
// (/v1/models?output_modalities=decisions) into ListModels, and those models are
// served only by the native decisions endpoint, which no GPT-Load route uses.
//
// A model is published when at least one of its declared modalities is servable,
// so a mixed entry such as ["text","decisions"] stays. Entries without a usable
// modality declaration are dropped: a decisions-only entry declares no servable
// output, and an entry that declares nothing (v1.13.0's merged {id,object,owned_by}
// items decode without an architecture) cannot be shown to be chat or embed.
func withoutOpenRouterUnservedModels(models []schemas.Model) []schemas.Model {
	kept := make([]schemas.Model, 0, len(models))
	for _, model := range models {
		if model.Architecture == nil {
			continue
		}
		for _, modality := range model.Architecture.OutputModalities {
			if servableOutputModality(modality) {
				kept = append(kept, model)
				break
			}
		}
	}
	return kept
}

// servableOutputModality reports whether GPT-Load proxies a model whose declared
// output modality is modality. Modality names are compared case-insensitively
// and ignore surrounding space, matching what providers actually return
// ("Embedding", "EMBEDDINGS").
func servableOutputModality(modality string) bool {
	switch strings.ToLower(strings.TrimSpace(modality)) {
	case "text", "embedding", "embeddings":
		return true
	default:
		return false
	}
}
