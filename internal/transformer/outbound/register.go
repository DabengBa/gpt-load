package outbound

import (
	"gpt-load/internal/transformer/model"
	"gpt-load/internal/transformer/outbound/anthropic"
	"gpt-load/internal/transformer/outbound/gemini"
	"gpt-load/internal/transformer/outbound/openai"
)

// OutboundType defines the type of outbound transformer
type OutboundType int

const (
	// OutboundTypeOpenAIChat represents OpenAI Chat format
	OutboundTypeOpenAIChat OutboundType = iota
	// OutboundTypeOpenAIResponse represents OpenAI Response format
	OutboundTypeOpenAIResponse
	// OutboundTypeAnthropic represents Anthropic Messages format
	OutboundTypeAnthropic
	// OutboundTypeGemini represents Gemini format
	OutboundTypeGemini
)

// String returns the string representation of OutboundType
func (t OutboundType) String() string {
	switch t {
	case OutboundTypeOpenAIChat:
		return "openai_chat"
	case OutboundTypeOpenAIResponse:
		return "openai_response"
	case OutboundTypeAnthropic:
		return "anthropic"
	case OutboundTypeGemini:
		return "gemini"
	default:
		return "unknown"
	}
}

// outboundFactories maps OutboundType to factory functions that create Outbound instances
var outboundFactories = map[OutboundType]func() model.Outbound{
	OutboundTypeOpenAIChat: func() model.Outbound {
		return openai.NewChatOutbound()
	},
	OutboundTypeAnthropic: func() model.Outbound {
		return anthropic.NewMessagesOutbound()
	},
	OutboundTypeGemini: func() model.Outbound {
		return gemini.NewMessagesOutbound()
	},
	// Note: OutboundTypeOpenAIResponse is not yet implemented
	// It will be added when the OpenAI Response format transformer is created
}

// GetOutbound returns an Outbound transformer instance for the given type.
// Returns nil if the type is not registered.
func GetOutbound(outboundType OutboundType) model.Outbound {
	if factory, ok := outboundFactories[outboundType]; ok {
		return factory()
	}
	return nil
}

// IsRegistered checks if an OutboundType is registered in the factory
func IsRegistered(outboundType OutboundType) bool {
	_, ok := outboundFactories[outboundType]
	return ok
}

// RegisteredTypes returns a slice of all registered OutboundTypes
func RegisteredTypes() []OutboundType {
	types := make([]OutboundType, 0, len(outboundFactories))
	for t := range outboundFactories {
		types = append(types, t)
	}
	return types
}

// OutboundTypeFromAPIFormat converts an APIFormat to the corresponding OutboundType.
// Returns OutboundTypeOpenAIChat as default if the format is not recognized.
func OutboundTypeFromAPIFormat(format model.APIFormat) OutboundType {
	switch format {
	case model.APIFormatOpenAIChat:
		return OutboundTypeOpenAIChat
	case model.APIFormatOpenAIResponse:
		return OutboundTypeOpenAIResponse
	case model.APIFormatAnthropic:
		return OutboundTypeAnthropic
	case model.APIFormatGemini:
		return OutboundTypeGemini
	default:
		return OutboundTypeOpenAIChat
	}
}
