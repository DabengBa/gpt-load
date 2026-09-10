package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"gpt-load/internal/dialect"
	"gpt-load/internal/usage"
)

type streamFailureKind uint8

const (
	streamFailureUpstreamRead streamFailureKind = iota + 1
	streamFailureProtocol
	streamFailureIdle
	streamFailureDownstreamWrite
	streamFailureClientCanceled
)

type streamFailure struct {
	kind streamFailureKind
	err  error
}

func (failure *streamFailure) Error() string { return failure.err.Error() }
func (failure *streamFailure) Unwrap() error { return failure.err }

type StreamEndReason uint8

const (
	StreamEndNone StreamEndReason = iota
	StreamEndCleanEOF
	StreamEndSSEError
	StreamEndUpstreamTerminated
	StreamEndUpstreamProtocolError
	StreamEndIdleTimeout
	StreamEndDownstreamWriteFailure
	StreamEndClientCanceled
	StreamEndServerShutdown
	StreamEndProviderIncomplete
)

type StreamObservation struct {
	EndReason    StreamEndReason
	ErrorSummary string
	ResponseID   string
}

type streamEventObserver struct {
	classifier          dialect.StreamEventClassifier
	terminalRequired    bool
	sawTerminal         bool
	terminalForwarded   bool
	terminalDisposition dialect.StreamEventDisposition
	eventCount          int
	firstProviderError  bool
	sawErrorEvent       bool
	firstSummary        string
	firstErrorPayload   []byte
	usage               *streamUsageCapture
	chatChoices         map[int]bool
	chatChoiceSeen      bool
	anthropicBlocks     map[int]bool
	anthropicBlockSeen  bool
	responseID          string
	strict              bool
}

// sseEventObservationBuffer frames arbitrary executor data chunks without
// changing their wire bytes. push returns the complete, successfully observed
// wire prefix; callers that use it may mark a returned terminal boundary
// forwarded only after that prefix has been written and flushed successfully.
type sseEventObservationBuffer struct {
	pending         []byte
	pendingTerminal bool
	maxEventBytes   int
	scanner         sseRewriteBoundaryScanner
	observe         func(dialect.StreamEvent, bool) (bool, error)
}

func newSSEEventObservationBuffer(
	maxEventBytes int,
	observe func(dialect.StreamEvent, bool) (bool, error),
) *sseEventObservationBuffer {
	return &sseEventObservationBuffer{
		maxEventBytes: normalizedSSEEventLimit(maxEventBytes),
		observe:       observe,
	}
}

func (buffer *sseEventObservationBuffer) push(chunk []byte) ([]byte, bool, error) {
	if buffer == nil || buffer.observe == nil {
		return nil, false, fmt.Errorf("SSE observation callback is required")
	}
	buffer.pending = append(buffer.pending, chunk...)
	wire := buffer.pending
	consumed := 0
	terminal := false
	for {
		waitingTerminal := buffer.pendingTerminal
		optionalLF, overflow := buffer.scanner.ConsumeOptionalLineFeed(
			buffer.pending,
			false,
			buffer.maxEventBytes,
		)
		if overflow {
			return nil, false, errSSEEventTooLarge
		}
		if waitingTerminal && !buffer.scanner.optionalLineFeed {
			buffer.pendingTerminal = false
			terminal = true
		}
		if optionalLF > 0 {
			buffer.discard(optionalLF)
			consumed += optionalLF
			continue
		}

		eventEnd, complete := buffer.scanner.Find(buffer.pending)
		if !complete {
			if len(buffer.pending) > buffer.maxEventBytes {
				return nil, false, errSSEEventTooLarge
			}
			return wire[:consumed], terminal, nil
		}
		if eventEnd > buffer.maxEventBytes {
			return nil, false, errSSEEventTooLarge
		}

		terminalEvent, err := buffer.observeEvent(buffer.pending[:eventEnd])
		if err != nil {
			return nil, false, err
		}
		buffer.discard(eventEnd)
		consumed += eventEnd
		buffer.scanner.AfterEvent(eventEnd, eventEnd)
		if terminalEvent && buffer.scanner.optionalLineFeed {
			buffer.pendingTerminal = true
		} else {
			terminal = terminal || terminalEvent
		}
	}
}

func (buffer *sseEventObservationBuffer) observeEvent(event []byte) (bool, error) {
	lines := splitSSEEventLines(event)
	dataValues := make([][]byte, 0, len(lines))
	var eventName []byte
	for index := range lines {
		if name, ok := parseSSEEventName(lines[index].content); ok {
			eventName = name
		}
		if lines[index].isData {
			dataValues = append(dataValues, lines[index].data)
		}
	}
	if len(dataValues) == 0 {
		return false, nil
	}
	payload := bytes.Join(dataValues, []byte{'\n'})
	if len(payload) == 0 {
		return false, nil
	}
	return buffer.observe(
		dialect.StreamEvent{Name: string(eventName), Payload: payload},
		bytes.Equal(eventName, []byte("error")) || isSSEErrorPayload(payload),
	)
}

func (buffer *sseEventObservationBuffer) finish() error {
	if buffer == nil {
		return fmt.Errorf("SSE observation buffer is required")
	}
	optionalLF, overflow := buffer.scanner.ConsumeOptionalLineFeed(
		buffer.pending,
		true,
		buffer.maxEventBytes,
	)
	if overflow {
		return errSSEEventTooLarge
	}
	if optionalLF > 0 {
		buffer.discard(optionalLF)
	}
	if buffer.pendingTerminal && !buffer.scanner.optionalLineFeed {
		buffer.pendingTerminal = false
	}
	if len(buffer.pending) > 0 {
		return errSSEEventIncomplete
	}
	buffer.scanner.Reset()
	return nil
}

func (buffer *sseEventObservationBuffer) discard(count int) {
	buffer.pending = buffer.pending[count:]
	if len(buffer.pending) == 0 {
		buffer.pending = nil
	}
}

func newStreamEventObserver(
	selected dialect.Dialect,
	capture *streamUsageCapture,
	strict ...bool,
) *streamEventObserver {
	observer := &streamEventObserver{usage: capture}
	if len(strict) > 0 {
		observer.strict = strict[0]
	}
	classifier, ok := selected.(dialect.StreamEventClassifier)
	if !ok {
		return observer
	}
	observer.classifier = classifier
	observer.terminalRequired = classifier.RequiresTerminalEvent()
	return observer
}

func safeClassifyStreamEvent(
	classifier dialect.StreamEventClassifier,
	event dialect.StreamEvent,
) (result dialect.StreamEventClassification, err error, panicked bool) {
	defer func() {
		if recover() != nil {
			result = dialect.StreamEventClassification{}
			err = nil
			panicked = true
		}
	}()
	result, err = classifier.ClassifyStreamEvent(dialect.StreamEvent{
		Name:    event.Name,
		Payload: bytes.Clone(event.Payload),
	})
	return result, err, false
}

func validStreamEventDisposition(value dialect.StreamEventDisposition) bool {
	return value >= dialect.StreamEventContinue &&
		value <= dialect.StreamEventFailed
}

func (observer *streamEventObserver) classify(
	event dialect.StreamEvent,
	genericProviderError bool,
) (bool, error) {
	if observer == nil {
		return genericProviderError, nil
	}
	if observer.sawTerminal {
		if observer.strict {
			return false, fmt.Errorf("%w: event received after stream terminal", ErrUpstreamProtocol)
		}
		return false, nil
	}
	if err := observer.observeProtocolState(event); err != nil {
		return false, err
	}

	classification := dialect.StreamEventClassification{
		Disposition: dialect.StreamEventContinue,
	}
	if observer.classifier != nil {
		var err error
		var panicked bool
		classification, err, panicked = safeClassifyStreamEvent(
			observer.classifier,
			event,
		)
		if panicked || err != nil ||
			!validStreamEventDisposition(classification.Disposition) {
			return false, fmt.Errorf(
				"%w: classify upstream SSE event",
				ErrUpstreamProtocol,
			)
		}
	}

	if classification.IsTerminal() {
		observer.sawTerminal = true
		observer.terminalDisposition = classification.Disposition
	}
	providerError := genericProviderError ||
		classification.IsProviderError()
	observer.eventCount++
	if observer.eventCount == 1 {
		observer.firstProviderError = providerError
	}
	return providerError, nil
}

func (observer *streamEventObserver) observeProtocolState(event dialect.StreamEvent) error {
	if observer == nil {
		return nil
	}
	if observer.classifier == nil {
		return nil
	}
	switch observer.classifier.(type) {
	case *dialect.OpenAI:
		if observer.strict {
			return observer.observeChatChoices(event)
		}
	case *dialect.Anthropic:
		if observer.strict {
			return observer.observeAnthropicBlocks(event)
		}
	case *dialect.OpenAIResponses:
		observer.captureResponsesID(event)
	}
	return nil
}

func (observer *streamEventObserver) observeChatChoices(event dialect.StreamEvent) error {
	if observer.chatChoices == nil {
		observer.chatChoices = make(map[int]bool)
	}
	if bytes.Equal(bytes.TrimSpace(event.Payload), []byte("[DONE]")) {
		if !observer.chatChoiceSeen {
			return fmt.Errorf("%w: Chat stream ended without a choice", ErrUpstreamProtocol)
		}
		for index, closed := range observer.chatChoices {
			if !closed {
				return fmt.Errorf("%w: Chat choice %d did not finish", ErrUpstreamProtocol, index)
			}
		}
		return nil
	}
	var envelope struct {
		Choices []struct {
			Index        *int            `json:"index"`
			FinishReason json.RawMessage `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(event.Payload, &envelope); err != nil {
		return nil
	}
	for position, choice := range envelope.Choices {
		index := position
		if choice.Index != nil {
			index = *choice.Index
		}
		observer.chatChoiceSeen = true
		closed := len(bytes.TrimSpace(choice.FinishReason)) > 0 && !bytes.Equal(bytes.TrimSpace(choice.FinishReason), []byte("null"))
		if closed {
			var finishReason string
			if err := json.Unmarshal(choice.FinishReason, &finishReason); err != nil || !validChatFinishReason(finishReason) {
				return fmt.Errorf("%w: invalid Chat finish_reason", ErrUpstreamProtocol)
			}
		}
		if previous, exists := observer.chatChoices[index]; exists && previous && !closed {
			return fmt.Errorf("%w: Chat choice %d reopened", ErrUpstreamProtocol, index)
		}
		if closed {
			observer.chatChoices[index] = true
		} else if _, exists := observer.chatChoices[index]; !exists {
			observer.chatChoices[index] = false
		}
	}
	return nil
}

func validChatFinishReason(value string) bool {
	switch value {
	case "stop", "length", "content_filter", "tool_calls", "function_call":
		return true
	default:
		return false
	}
}

func (observer *streamEventObserver) observeAnthropicBlocks(event dialect.StreamEvent) error {
	if observer.anthropicBlocks == nil {
		observer.anthropicBlocks = make(map[int]bool)
	}
	eventType, err := anthropicStreamEventType(event)
	if err != nil {
		return err
	}
	var envelope struct {
		Index *int `json:"index"`
	}
	if err := json.Unmarshal(event.Payload, &envelope); err != nil {
		return fmt.Errorf("%w: decode Anthropic stream event", ErrUpstreamProtocol)
	}
	index := 0
	if envelope.Index != nil {
		index = *envelope.Index
	}
	switch eventType {
	case "content_block_start":
		observer.anthropicBlockSeen = true
		if _, exists := observer.anthropicBlocks[index]; exists {
			return fmt.Errorf("%w: duplicate Anthropic content block start", ErrUpstreamProtocol)
		}
		observer.anthropicBlocks[index] = false
	case "content_block_stop":
		closed, exists := observer.anthropicBlocks[index]
		if !exists || closed {
			return fmt.Errorf("%w: invalid Anthropic content block stop", ErrUpstreamProtocol)
		}
		observer.anthropicBlocks[index] = true
	case "message_stop":
		for index, closed := range observer.anthropicBlocks {
			if !closed {
				return fmt.Errorf("%w: Anthropic content block %d did not stop", ErrUpstreamProtocol, index)
			}
		}
	}
	return nil
}

func anthropicStreamEventType(event dialect.StreamEvent) (string, error) {
	if event.Name != "" {
		var envelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(event.Payload, &envelope); err != nil {
			return "", fmt.Errorf("%w: decode Anthropic stream event", ErrUpstreamProtocol)
		}
		if envelope.Type != "" && envelope.Type != event.Name {
			return "", fmt.Errorf("%w: Anthropic event name conflicts with payload type", ErrUpstreamProtocol)
		}
		return event.Name, nil
	}
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(event.Payload, &envelope); err != nil {
		return "", fmt.Errorf("%w: decode Anthropic stream event", ErrUpstreamProtocol)
	}
	return envelope.Type, nil
}

func (observer *streamEventObserver) captureResponsesID(event dialect.StreamEvent) {
	var envelope struct {
		ID       string `json:"id"`
		Response struct {
			ID string `json:"id"`
		} `json:"response"`
	}
	if json.Unmarshal(event.Payload, &envelope) != nil {
		return
	}
	if envelope.Response.ID != "" {
		observer.responseID = envelope.Response.ID
	} else if envelope.ID != "" {
		observer.responseID = envelope.ID
	}
}
func (observer *streamEventObserver) firstEventWasProviderError() bool {
	return observer != nil && observer.firstProviderError
}

func (observer *streamEventObserver) observeError(payload []byte, summary string) {
	if observer == nil {
		return
	}
	observer.sawErrorEvent = true
	if observer.firstSummary == "" {
		observer.firstSummary = summary
		observer.firstErrorPayload = bytes.Clone(payload)
	}
}

func (observer *streamEventObserver) observeUsageEvent(
	event dialect.StreamEvent,
) {
	if bytes.Equal(bytes.TrimSpace(event.Payload), []byte("[DONE]")) {
		return
	}
	if observer != nil && observer.usage != nil {
		observer.usage.observeEvent(event)
	}
}

func (observer *streamEventObserver) finalizeUsage() usage.Result {
	if observer == nil || observer.usage == nil {
		return usage.Result{State: usage.StateMissing}
	}
	return observer.usage.finalize()
}

func (observer *streamEventObserver) markTerminalForwarded() bool {
	if observer == nil || !observer.sawTerminal {
		return false
	}
	observer.terminalForwarded = true
	return true
}

func (observer *streamEventObserver) endObservation() StreamObservation {
	if observer == nil {
		return StreamObservation{EndReason: StreamEndCleanEOF}
	}
	if observer.sawErrorEvent {
		return StreamObservation{
			EndReason:    StreamEndSSEError,
			ErrorSummary: observer.firstSummary,
			ResponseID:   observer.responseID,
		}
	}
	if observer.sawTerminal && observer.terminalDisposition == dialect.StreamEventFailed {
		return StreamObservation{
			EndReason:    StreamEndSSEError,
			ErrorSummary: fixedErrorSummary("upstream_sse_error"),
			ResponseID:   observer.responseID,
		}
	}
	if observer.sawTerminal &&
		observer.terminalDisposition == dialect.StreamEventIncomplete {
		return StreamObservation{
			EndReason:    StreamEndProviderIncomplete,
			ErrorSummary: fixedErrorSummary("upstream_response_incomplete"),
			ResponseID:   observer.responseID,
		}
	}
	return StreamObservation{EndReason: StreamEndCleanEOF, ResponseID: observer.responseID}
}

func (observer *streamEventObserver) validateEOF() error {
	if observer == nil || !observer.terminalRequired || observer.sawTerminal {
		return nil
	}
	return &streamFailure{
		kind: streamFailureProtocol,
		err: fmt.Errorf(
			"%w: stream ended before required terminal event",
			ErrUpstreamProtocol,
		),
	}
}

func observeStreamTermination(
	ctx context.Context,
	err error,
	events *streamEventObserver,
) StreamObservation {
	observation := events.endObservation()
	if events != nil && events.terminalForwarded {
		if errors.Is(err, ErrUpstreamProtocol) {
			result := prioritizeStreamObservation(nil, err, observation)
			result.ResponseID = observation.ResponseID
			return result
		}
		if errors.Is(err, context.Canceled) ||
			(ctx != nil && errors.Is(ctx.Err(), context.Canceled)) {
			return observation
		}
	}
	if events != nil && events.sawTerminal && events.terminalDisposition == dialect.StreamEventIncomplete &&
		(errors.Is(err, context.Canceled) || (ctx != nil && errors.Is(ctx.Err(), context.Canceled))) {
		return StreamObservation{EndReason: StreamEndProviderIncomplete, ErrorSummary: fixedErrorSummary("upstream_response_incomplete"), ResponseID: observation.ResponseID}
	}
	result := prioritizeStreamObservation(ctx, err, observation)
	result.ResponseID = observation.ResponseID
	return result
}

func prioritizeStreamObservation(
	ctx context.Context,
	err error,
	observation StreamObservation,
) StreamObservation {
	if isServerShutdown(ctx) {
		return streamTerminalObservation(StreamEndServerShutdown)
	}
	if (ctx != nil && ctx.Err() != nil) || errors.Is(err, context.Canceled) {
		return streamTerminalObservation(StreamEndClientCanceled)
	}

	var failure *streamFailure
	if errors.As(err, &failure) {
		switch failure.kind {
		case streamFailureClientCanceled:
			return streamTerminalObservation(StreamEndClientCanceled)
		case streamFailureDownstreamWrite:
			return streamTerminalObservation(StreamEndDownstreamWriteFailure)
		case streamFailureIdle:
			return streamTerminalObservation(StreamEndIdleTimeout)
		case streamFailureProtocol:
			return streamTerminalObservation(StreamEndUpstreamProtocolError)
		case streamFailureUpstreamRead:
			return streamTerminalObservation(StreamEndUpstreamTerminated)
		}
	}

	switch {
	case errors.Is(err, ErrUpstreamProtocol):
		return streamTerminalObservation(StreamEndUpstreamProtocolError)
	case errors.Is(err, errStreamIdleTimeout):
		return streamTerminalObservation(StreamEndIdleTimeout)
	case err != nil:
		return streamTerminalObservation(StreamEndUpstreamTerminated)
	case observation.EndReason != StreamEndNone:
		return observation
	default:
		return StreamObservation{EndReason: StreamEndCleanEOF}
	}
}

func streamTerminalObservation(reason StreamEndReason) StreamObservation {
	code := streamErrorCode(reason)
	return StreamObservation{
		EndReason:    reason,
		ErrorSummary: fixedErrorSummary(code),
	}
}

func streamTerminalObservationWithResponseID(reason StreamEndReason, responseID string) StreamObservation {
	observation := streamTerminalObservation(reason)
	observation.ResponseID = responseID
	return observation
}
func streamErrorCode(reason StreamEndReason) string {
	switch reason {
	case StreamEndCleanEOF, StreamEndNone:
		return ""
	case StreamEndSSEError:
		return "upstream_sse_error"
	case StreamEndUpstreamTerminated:
		return "upstream_stream_terminated"
	case StreamEndUpstreamProtocolError:
		return "upstream_protocol_error"
	case StreamEndIdleTimeout:
		return "upstream_stream_idle_timeout"
	case StreamEndDownstreamWriteFailure:
		return "downstream_write_failed"
	case StreamEndClientCanceled:
		return "client_canceled"
	case StreamEndServerShutdown:
		return "server_shutdown"
	case StreamEndProviderIncomplete:
		return "upstream_response_incomplete"
	default:
		return "upstream_stream_terminated"
	}
}
