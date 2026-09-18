package gateway

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/httplifecycle"
	"gpt-load/internal/protocol"
)

func feedChatStreamEvents(t *testing.T, observer *streamEventObserver, payloads ...string) {
	t.Helper()
	for _, payload := range payloads {
		if _, err := observer.classify(dialect.StreamEvent{Payload: []byte(payload)}, false); err != nil {
			t.Fatalf("classify(%s) error = %v", payload, err)
		}
	}
}

func TestStreamContentFilterWithoutAssistantContentIsError(t *testing.T) {
	for _, test := range []struct {
		name   string
		strict bool
	}{
		{name: "plain_stream"},
		{name: "buffered_stream", strict: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			observer := newStreamEventObserver(dialect.NewOpenAI(), nil, test.strict)
			feedChatStreamEvents(t, observer,
				`{"id":"chat_1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`{"id":"chat_1","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}`,
				`[DONE]`,
			)
			observation := observer.endObservation()
			if observation.EndReason != StreamEndContentFilter {
				t.Fatalf("EndReason = %v, want %v", observation.EndReason, StreamEndContentFilter)
			}
			if code := streamErrorCode(observation.EndReason); code != "upstream_content_filter" {
				t.Fatalf("streamErrorCode() = %q, want %q", code, "upstream_content_filter")
			}
			if observation.ErrorSummary != "upstream stream was blocked by content filter and contained no assistant content" {
				t.Fatalf("ErrorSummary = %q", observation.ErrorSummary)
			}
		})
	}
}

func TestStreamContentFilterWithAssistantContentStaysClean(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewOpenAI(), nil)
	feedChatStreamEvents(t, observer,
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"content":"partial answer"},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}`,
		`[DONE]`,
	)
	if observation := observer.endObservation(); observation.EndReason != StreamEndCleanEOF {
		t.Fatalf("EndReason = %v, want %v", observation.EndReason, StreamEndCleanEOF)
	}
}

func TestStreamContentFilterStopWithoutAssistantContentStaysClean(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewOpenAI(), nil)
	feedChatStreamEvents(t, observer,
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`[DONE]`,
	)
	if observation := observer.endObservation(); observation.EndReason != StreamEndCleanEOF {
		t.Fatalf("EndReason = %v, want %v", observation.EndReason, StreamEndCleanEOF)
	}
}

func TestStreamContentFilterToolCallPayloadStaysClean(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewOpenAI(), nil)
	feedChatStreamEvents(t, observer,
		`{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}`,
		`[DONE]`,
	)
	if observation := observer.endObservation(); observation.EndReason != StreamEndCleanEOF {
		t.Fatalf("EndReason = %v, want %v", observation.EndReason, StreamEndCleanEOF)
	}
}

func TestSSEObservationRejectsEventAboveImagesLimit(t *testing.T) {
	limit := execution.SSEEventLimit(protocol.OpenAIImages)
	buffer := newSSEEventObservationBuffer(
		limit,
		func(dialect.StreamEvent, bool) (bool, error) { return false, nil },
	)
	_, _, err := buffer.push(bytes.Repeat([]byte{'x'}, limit+1))
	if !errors.Is(err, errSSEEventTooLarge) {
		t.Fatalf("push() error = %v, want %v", err, errSSEEventTooLarge)
	}
}

func TestPrioritizeStreamObservationIdentifiesServerShutdown(t *testing.T) {
	requestContext, cancel := context.WithCancelCause(context.Background())
	cancel(httplifecycle.ErrServerShutdown)

	observation := prioritizeStreamObservation(
		requestContext,
		context.Canceled,
		StreamObservation{},
	)
	if observation.EndReason != StreamEndServerShutdown ||
		streamErrorCode(observation.EndReason) != "server_shutdown" ||
		observation.ErrorSummary != fixedErrorSummary("server_shutdown") {
		t.Fatalf("shutdown observation = %#v", observation)
	}
}
