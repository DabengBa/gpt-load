// Package protocol defines protocol identifiers shared by runtime domains.
package protocol

type Protocol string

const (
	OpenAICompletions Protocol = "openai-completions"
	OpenAIResponses   Protocol = "openai-responses"
	OpenAIImages      Protocol = "openai-images"
	OpenAIEmbeddings  Protocol = "openai-embeddings"
	Rerank            Protocol = "rerank"
	Anthropic         Protocol = "anthropic"
	Gemini            Protocol = "gemini"
)

func (p Protocol) Valid() bool {
	switch p {
	case OpenAICompletions, OpenAIResponses, OpenAIImages, OpenAIEmbeddings, Rerank, Anthropic, Gemini:
		return true
	default:
		return false
	}
}

func (p Protocol) DataPlaneEnabled() bool {
	switch p {
	case OpenAICompletions, OpenAIResponses, OpenAIImages, OpenAIEmbeddings, Rerank, Anthropic, Gemini:
		return true
	default:
		return false
	}
}

func (p Protocol) SupportsModelOptionalRequests() bool {
	return p == OpenAIResponses
}

// SupportsGeneratedText reports whether a protocol response carries generated
// text that a probe can verify. Embeddings, Rerank, and Images return data
// instead of generated text, so they never satisfy a probe contract.
func (p Protocol) SupportsGeneratedText() bool {
	switch p {
	case OpenAICompletions, OpenAIResponses, Anthropic, Gemini:
		return true
	default:
		return false
	}
}

func DataPlaneProtocols() []Protocol {
	return []Protocol{
		OpenAICompletions,
		OpenAIResponses,
		OpenAIImages,
		OpenAIEmbeddings,
		Rerank,
		Anthropic,
		Gemini,
	}
}
