package reasoning

import (
	"errors"
	"fmt"
	"slices"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/providers/gemini"
	"github.com/maximhq/bifrost/core/providers/openai"
	"github.com/maximhq/bifrost/core/schemas"
)

var ErrUnsupportedEffort = errors.New("reasoning effort unsupported")

// EffortCapability is the secret-free control-plane projection for one model.
type EffortCapability struct {
	Known     bool     `json:"known"`
	Supported bool     `json:"supported"`
	Levels    []string `json:"levels"`
	Reason    string   `json:"reason,omitempty"`
}

// ProjectEffortCapability uses the same Bifrost model capability methods as request conversion.
func ProjectEffortCapability(provider string, model string) EffortCapability {
	providerID, ok := modelProvider(provider)
	if !ok || model == "" {
		return EffortCapability{Reason: "provider or model capability is unknown"}
	}
	caps := schemas.ResolveModelCaps(providerID, model)
	fallback, fallbackSupported := providerEffortFallback(providerID, caps.Model())
	levels := caps.ReasoningEffortLevels(fallback)
	supported := caps.SupportsReasoningEffort(fallbackSupported) && len(levels) > 0
	if providerID == schemas.Anthropic {
		supported = caps.SupportsNativeEffort(anthropic.DefaultSupportsNativeEffort(caps.Model()))
		levels = slices.DeleteFunc(slices.Clone(levels), func(level string) bool {
			return level == "none" || anthropic.MapBifrostEffortToAnthropic(level) != level
		})
	}
	return EffortCapability{
		Known: caps.Record() != nil || fallbackSupported, Supported: supported, Levels: slices.Clone(levels),
		Reason: capabilityReason(caps, supported),
	}
}

// ValidateEffort requires the exact requested label; it never normalizes or downgrades it.
func ValidateEffort(provider string, model, effort string) error {
	if effort == "" {
		return nil
	}
	projection := ProjectEffortCapability(provider, model)
	if !projection.Supported {
		return fmt.Errorf("%w: %s", ErrUnsupportedEffort, projection.Reason)
	}
	providerID, _ := modelProvider(provider)
	caps := schemas.ResolveModelCaps(providerID, model)
	valid := effort != "none" || caps.CanDisableReasoning(false)
	if effort != "none" {
		valid = slices.Contains(projection.Levels, effort)
	}
	if !valid {
		return fmt.Errorf("%w: requested effort is not supported", ErrUnsupportedEffort)
	}
	return nil
}

func providerEffortFallback(provider schemas.ModelProvider, model string) (*schemas.EffortControl, bool) {
	switch provider {
	case schemas.OpenAI:
		if openai.IsOpenAIReasoningModel(model) {
			return openai.DefaultEffortControl(model), true
		}
	case schemas.Gemini:
		fallback := gemini.DefaultEffortControl(model)
		return fallback, fallback != nil
	case schemas.Anthropic:
		return nil, anthropic.DefaultSupportsNativeEffort(model)
	}
	return nil, false
}

func capabilityReason(caps schemas.ModelCaps, supported bool) string {
	if supported {
		return "supported by Bifrost model capabilities"
	}
	if caps.Record() == nil {
		return "Bifrost has no supported model effort capability"
	}
	return "Bifrost model capabilities do not support categorical reasoning effort"
}

func modelProvider(provider string) (schemas.ModelProvider, bool) {
	switch provider {
	case "openai", "azure_openai", "codex":
		return schemas.OpenAI, true
	case "anthropic", "claude":
		return schemas.Anthropic, true
	case "gemini", "google_vertex":
		return schemas.Gemini, true
	case "deepseek":
		return schemas.DeepSeek, true
	case "xai", "grok":
		return schemas.XAI, true
	case "openrouter":
		return schemas.OpenRouter, true
	case "groq":
		return schemas.Groq, true
	case "aws_bedrock":
		return schemas.Bedrock, true
	default:
		return "", false
	}
}
