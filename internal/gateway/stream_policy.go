package gateway

import (
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

// streamDelivery is the fixed HTTP/SSE delivery outcome for one parsed
// streaming request.
type streamDelivery uint8

const (
	// streamDeliveryNotStreaming marks a non-streaming request. The handler
	// routes those through Forward and never consults the stream policy.
	streamDeliveryNotStreaming streamDelivery = iota
	// streamDeliveryBuffered releases the upstream payload only after the
	// protocol terminal state has been validated.
	streamDeliveryBuffered
	// streamDeliveryLiveException keeps the real-time pass-through path for
	// the protocols that cannot use the buffered gate.
	streamDeliveryLiveException
	// streamDeliveryReject refuses the stream before any provider dispatch.
	streamDeliveryReject
)

// evaluateStreamDelivery is the single entry point of the HTTP/SSE stream
// delivery policy. It classifies an already parsed request by client protocol,
// operation and stream flag. Malformed requests are rejected by the dialect
// before this function is reached, and non-streaming requests use Forward
// instead of this policy.
func evaluateStreamDelivery(
	clientProtocol protocol.Protocol,
	operation execution.Operation,
	stream bool,
) (streamDelivery, *reason) {
	if !stream {
		return streamDeliveryNotStreaming, nil
	}
	switch clientProtocol {
	case protocol.OpenAICompletions, protocol.Anthropic:
		if operation == execution.OperationChatCompletion {
			return streamDeliveryBuffered, nil
		}
		return streamDeliveryReject, &reasonStreamingOperationUnsupported
	case protocol.OpenAIResponses:
		if operation == execution.OperationResponsesCreate {
			return streamDeliveryBuffered, nil
		}
		return streamDeliveryReject, &reasonStreamingOperationUnsupported
	case protocol.Gemini:
		if operation == execution.OperationChatCompletion {
			return streamDeliveryLiveException, nil
		}
		return streamDeliveryReject, &reasonStreamingOperationUnsupported
	case protocol.OpenAIImages:
		if operation == execution.OperationImagesGenerate || operation == execution.OperationImagesEdit {
			return streamDeliveryLiveException, nil
		}
		return streamDeliveryReject, &reasonStreamingOperationUnsupported
	default:
		return streamDeliveryReject, &reasonStreamingProtocolUnsupported
	}
}
