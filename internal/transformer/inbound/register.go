package inbound

import (
	"gpt-load/internal/transformer/inbound/anthropic"
	"gpt-load/internal/transformer/inbound/openai"
	"gpt-load/internal/transformer/model"
)

// InboundType defines the type of inbound transformer
type InboundType int

const (
	// InboundTypeOpenAIChat represents OpenAI Chat format
	InboundTypeOpenAIChat InboundType = iota
	// InboundTypeOpenAIResponse represents OpenAI Response format
	InboundTypeOpenAIResponse
	// InboundTypeAnthropic represents Anthropic Messages format
	InboundTypeAnthropic
)

// String returns the string representation of InboundType
func (t InboundType) String() string {
	switch t {
	case InboundTypeOpenAIChat:
		return "openai_chat"
	case InboundTypeOpenAIResponse:
		return "openai_response"
	case InboundTypeAnthropic:
		return "anthropic"
	default:
		return "unknown"
	}
}

// inboundFactories maps InboundType to factory functions that create Inbound instances
var inboundFactories = map[InboundType]func() model.Inbound{
	InboundTypeOpenAIChat: func() model.Inbound {
		return openai.NewChatInbound()
	},
	InboundTypeAnthropic: func() model.Inbound {
		return anthropic.NewMessagesInbound()
	},
	// Note: InboundTypeOpenAIResponse is not yet implemented
	// It will be added when the OpenAI Response format transformer is created
}

// GetInbound returns an Inbound transformer instance for the given type.
// Returns nil if the type is not registered.
func GetInbound(inboundType InboundType) model.Inbound {
	if factory, ok := inboundFactories[inboundType]; ok {
		return factory()
	}
	return nil
}

// IsRegistered checks if an InboundType is registered in the factory
func IsRegistered(inboundType InboundType) bool {
	_, ok := inboundFactories[inboundType]
	return ok
}

// RegisteredTypes returns a slice of all registered InboundTypes
func RegisteredTypes() []InboundType {
	types := make([]InboundType, 0, len(inboundFactories))
	for t := range inboundFactories {
		types = append(types, t)
	}
	return types
}

// InboundTypeFromAPIFormat converts an APIFormat to the corresponding InboundType.
// Returns InboundTypeOpenAIChat as default if the format is not recognized.
func InboundTypeFromAPIFormat(format model.APIFormat) InboundType {
	switch format {
	case model.APIFormatOpenAIChat:
		return InboundTypeOpenAIChat
	case model.APIFormatOpenAIResponse:
		return InboundTypeOpenAIResponse
	case model.APIFormatAnthropic:
		return InboundTypeAnthropic
	default:
		return InboundTypeOpenAIChat
	}
}
