package bifrost

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/maximhq/bifrost/core/schemas"

	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/usage"
)

type openAIChatResponse struct {
	ID                string                          `json:"id"`
	Choices           []schemas.BifrostResponseChoice `json:"choices"`
	Created           int                             `json:"created"`
	Model             string                          `json:"model"`
	Object            string                          `json:"object"`
	ServiceTier       *schemas.BifrostServiceTier     `json:"service_tier,omitempty"`
	SystemFingerprint string                          `json:"system_fingerprint,omitempty"`
	Usage             *openAIUsage                    `json:"usage,omitempty"`
}

type openAIUsage struct {
	PromptTokens            int                                  `json:"prompt_tokens"`
	PromptTokensDetails     *openAIPromptTokensDetails           `json:"prompt_tokens_details,omitempty"`
	CompletionTokens        int                                  `json:"completion_tokens"`
	CompletionTokensDetails *schemas.ChatCompletionTokensDetails `json:"completion_tokens_details,omitempty"`
	TotalTokens             int                                  `json:"total_tokens"`
}

type openAIPromptTokensDetails struct {
	AudioTokens      int `json:"audio_tokens,omitempty"`
	CachedTokens     int `json:"cached_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

func encodeChatResponse(response *schemas.BifrostChatResponse) ([]byte, *execution.UsageEvidence, error) {
	if response == nil {
		return nil, nil, fmt.Errorf("response is nil")
	}
	wireUsage := toOpenAIUsage(response.Usage)
	wire := openAIChatResponse{
		ID:                response.ID,
		Choices:           response.Choices,
		Created:           response.Created,
		Model:             response.Model,
		Object:            response.Object,
		ServiceTier:       response.ServiceTier,
		SystemFingerprint: response.SystemFingerprint,
		Usage:             wireUsage,
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal OpenAI response")
	}
	if wireUsage == nil {
		return body, nil, nil
	}
	rawUsage, err := json.Marshal(wireUsage)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal OpenAI usage")
	}
	normalized, err := dialect.NewOpenAI().ExtractUsage(body)
	if err != nil {
		return nil, nil, fmt.Errorf("normalize OpenAI usage")
	}
	return body, &execution.UsageEvidence{Normalized: normalized, Raw: rawUsage}, nil
}

func toOpenAIUsage(source *schemas.BifrostLLMUsage) *openAIUsage {
	if source == nil {
		return nil
	}
	result := &openAIUsage{
		PromptTokens:            source.PromptTokens,
		CompletionTokens:        source.CompletionTokens,
		CompletionTokensDetails: source.CompletionTokensDetails,
		TotalTokens:             source.TotalTokens,
	}
	if source.PromptTokensDetails != nil {
		result.PromptTokensDetails = &openAIPromptTokensDetails{
			AudioTokens:      source.PromptTokensDetails.AudioTokens,
			CachedTokens:     source.PromptTokensDetails.CachedReadTokens,
			CacheWriteTokens: source.PromptTokensDetails.CachedWriteTokens,
		}
	}
	return result
}

func usageEvidenceFromPassthrough(source *schemas.BifrostPassthroughUsage) (*execution.UsageEvidence, error) {
	if source == nil || source.LLMUsage == nil {
		return nil, nil
	}
	wireUsage := toOpenAIUsage(source.LLMUsage)
	rawUsage, err := json.Marshal(wireUsage)
	if err != nil {
		return nil, fmt.Errorf("marshal OpenAI usage")
	}
	syntheticBody, err := json.Marshal(struct {
		Usage *openAIUsage `json:"usage"`
	}{Usage: wireUsage})
	if err != nil {
		return nil, fmt.Errorf("marshal OpenAI usage response")
	}
	normalized, err := dialect.NewOpenAI().ExtractUsage(syntheticBody)
	if err != nil {
		return nil, fmt.Errorf("normalize OpenAI usage")
	}
	return &execution.UsageEvidence{Normalized: normalized, Raw: rawUsage}, nil
}

func usageEvidenceFromResponses(source *schemas.ResponsesResponseUsage) (*execution.UsageEvidence, error) {
	if source == nil {
		return nil, nil
	}
	rawUsage, err := json.Marshal(source)
	if err != nil {
		return nil, fmt.Errorf("marshal Responses usage")
	}
	syntheticBody, err := json.Marshal(struct {
		Usage *schemas.ResponsesResponseUsage `json:"usage"`
	}{Usage: source})
	if err != nil {
		return nil, fmt.Errorf("marshal Responses usage response")
	}
	normalized, err := dialect.NewOpenAIResponses().ExtractUsage(syntheticBody)
	if err != nil {
		return nil, fmt.Errorf("normalize Responses usage")
	}
	return &execution.UsageEvidence{Normalized: normalized, Raw: rawUsage}, nil
}

// BilledUsage is a cumulative provider snapshot, not a delta to add to the stream.
func errorUsageEvidence(bifrostError *schemas.BifrostError, previous *execution.UsageEvidence) *execution.UsageEvidence {
	if bifrostError == nil || bifrostError.ExtraFields.BilledUsage == nil {
		return cloneUsage(previous)
	}
	evidence, err := billedUsageEvidence(bifrostError.ExtraFields.BilledUsage)
	if err != nil {
		evidence = cloneUsage(previous)
		if evidence == nil {
			evidence = &execution.UsageEvidence{Normalized: usage.Result{State: usage.StateMissing}}
		}
		evidence.Normalized.Diagnostics.Add(usage.DiagnosticInvalidPayload)
		return evidence
	}
	if evidence.Normalized.State == usage.StateComplete {
		evidence.Normalized.State = usage.StatePartial
	}
	return evidence
}

func billedUsageEvidence(source *schemas.BifrostLLMUsage) (*execution.UsageEvidence, error) {
	evidence, err := usageEvidenceFromResponses(source.ToResponsesResponseUsage())
	if err != nil || evidence == nil || source.PromptTokensDetails == nil {
		return evidence, err
	}
	details := source.PromptTokensDetails
	write := int64(details.CachedWriteTokens)
	var five, hour int64
	if details.CachedWriteTokenDetails != nil {
		five = int64(details.CachedWriteTokenDetails.CachedWriteTokens5m)
		hour = int64(details.CachedWriteTokenDetails.CachedWriteTokens1h)
	}
	split, ok := usage.CheckedAdd(five, hour)
	if !ok || write < 0 {
		return nil, fmt.Errorf("invalid billed cache write usage")
	}
	if write == 0 {
		write = split
	}
	unknown, ok := usage.SubtractCached(write, split)
	if !ok {
		return nil, fmt.Errorf("billed cache write details exceed aggregate")
	}
	uncached, ok := usage.SubtractCached(evidence.Normalized.Tokens.UncachedInput, write)
	if !ok {
		return nil, fmt.Errorf("billed cache writes exceed input")
	}
	evidence.Normalized.Tokens.UncachedInput = uncached
	evidence.Normalized.Tokens.CacheWrite5M = five
	evidence.Normalized.Tokens.CacheWrite1H = hour
	evidence.Normalized.Tokens.CacheWriteUnknown = unknown
	return evidence, nil
}

func openAIResponseModel(body []byte, fallback string) string {
	var response struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &response); err == nil && strings.TrimSpace(response.Model) != "" {
		return response.Model
	}
	return fallback
}

func frameSSE(payload []byte) []byte {
	framed := make([]byte, 0, len(payload)+8)
	framed = append(framed, "data: "...)
	framed = append(framed, payload...)
	framed = append(framed, '\n', '\n')
	return framed
}

func encodeSafeErrorBody(evidence *execution.ErrorEvidence) []byte {
	if evidence == nil {
		return nil
	}
	body, err := json.Marshal(struct {
		Error struct {
			Type    string `json:"type,omitempty"`
			Code    string `json:"code,omitempty"`
			Message string `json:"message"`
		} `json:"error"`
	}{
		Error: struct {
			Type    string `json:"type,omitempty"`
			Code    string `json:"code,omitempty"`
			Message string `json:"message"`
		}{Type: evidence.Type, Code: evidence.Code, Message: evidence.Summary},
	})
	if err != nil {
		return nil
	}
	return body
}
