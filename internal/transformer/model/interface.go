package model

import (
	"context"
	"net/http"
)

// Inbound is the inbound transformer interface
// Responsible for converting client requests to internal format
// and converting internal responses back to client format
type Inbound interface {
	// TransformRequest converts client request to internal format
	TransformRequest(ctx context.Context, body []byte) (*InternalLLMRequest, error)

	// TransformResponse converts internal response to client format
	TransformResponse(ctx context.Context, response *InternalLLMResponse) ([]byte, error)

	// TransformStream converts internal streaming response to client streaming format
	TransformStream(ctx context.Context, stream *InternalLLMResponse) ([]byte, error)

	// GetInternalResponse returns the aggregated complete response
	// For streaming: aggregates stored streaming chunks into complete response
	// For non-streaming: returns the stored complete response
	GetInternalResponse(ctx context.Context) (*InternalLLMResponse, error)
}

// Outbound is the outbound transformer interface
// Responsible for converting internal requests to upstream format
// and converting upstream responses to internal format
type Outbound interface {
	// TransformRequest converts internal request to upstream HTTP request
	TransformRequest(ctx context.Context, request *InternalLLMRequest, baseUrl, key string) (*http.Request, error)

	// TransformResponse converts upstream HTTP response to internal format
	TransformResponse(ctx context.Context, response *http.Response) (*InternalLLMResponse, error)

	// TransformStream converts upstream streaming data to internal format
	TransformStream(ctx context.Context, eventData []byte) (*InternalLLMResponse, error)
}

/*
Request Flow

Non-streaming:
client      -> inbound.TransformRequest(ctx, body)
            -> outbound.TransformRequest(ctx, request, baseUrl, key)
            -> http.Do(request)
            -> outbound.TransformResponse(ctx, response)
            -> inbound.TransformResponse(ctx, response)
                                                        -> client

Streaming:
client      -> inbound.TransformRequest(ctx, body)
            -> outbound.TransformRequest(ctx, request, baseUrl, key)
            -> http.Do(request)
            -> outbound.TransformStream(ctx, chunk)
            -> inbound.TransformStream(ctx, chunk)
                                                        -> client
*/
