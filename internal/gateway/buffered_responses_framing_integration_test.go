package gateway

import (
	"context"
	"errors"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	bifrostexecutor "gpt-load/internal/execution/bifrost"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/testutil/encryptiontest"
)

// bufferedResponsesFramingIncidentFixture is the sanitized incident shape: an
// event-only block whose remaining lines are a comment, immediately followed by
// its data-only block. It must be merged before the spool.
const bufferedResponsesFramingIncidentFixture = "event: response.created\n" +
	"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_framing\",\"status\":\"in_progress\",\"output\":[]}}\n" +
	"\n" +
	"event: response.function_call_arguments.delta\n" +
	": keep-alive\n" +
	"\n" +
	"data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"x\"}\n" +
	"\n" +
	"event: response.completed\n" +
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_framing\",\"status\":\"completed\",\"output\":[]}}\n" +
	"\n"

// bufferedResponsesFramingIncidentNormalized is the exact downstream wire for
// the incident fixture: only the redundant delimiter between the event-only
// block and its matching data block is removed.
const bufferedResponsesFramingIncidentNormalized = "event: response.created\n" +
	"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_framing\",\"status\":\"in_progress\",\"output\":[]}}\n" +
	"\n" +
	"event: response.function_call_arguments.delta\n" +
	": keep-alive\n" +
	"data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"x\"}\n" +
	"\n" +
	"event: response.completed\n" +
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_framing\",\"status\":\"completed\",\"output\":[]}}\n" +
	"\n"

// bufferedResponsesFramingAmbiguousFixture is an event-only block carrying an
// extra field, so the framing cannot be proven recoverable. The provider stream
// itself terminates cleanly, so only the framing normalizer can reject it.
const bufferedResponsesFramingAmbiguousFixture = "event: response.created\n" +
	"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_reject\",\"status\":\"in_progress\",\"output\":[]}}\n" +
	"\n" +
	"event: response.output_text.delta\n" +
	"id: 5\n" +
	"\n" +
	"data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n" +
	"\n" +
	"event: response.completed\n" +
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_reject\",\"status\":\"completed\",\"output\":[]}}\n" +
	"\n"

// bufferedResponsesFramingAmbiguousErrorEvent is the existing buffered failure
// envelope for a rejected framing that never reached a downstream commit, so no
// response ID is carried.
const bufferedResponsesFramingAmbiguousErrorEvent = "event: error\n" +
	"data: {\"type\":\"error\",\"error\":{\"type\":\"server_error\"," +
	"\"message\":\"The buffered upstream stream could not be completed.\"," +
	"\"code\":\"buffered_stream_failed\"}}\n" +
	"\n"

// bufferedResponsesFramingLegalFixture is a legal stream that already contains
// an upstream standalone heartbeat block; normalization must leave it intact.
const bufferedResponsesFramingLegalFixture = "event: response.created\n" +
	"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_legal\",\"status\":\"in_progress\",\"output\":[]}}\n" +
	"\n" +
	": keep-alive\n" +
	"\n" +
	"event: response.completed\n" +
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_legal\",\"status\":\"completed\",\"output\":[]}}\n" +
	"\n"

const bufferedResponsesFramingSuccessFixture = "event: response.created\n" +
	"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_ok\",\"status\":\"in_progress\",\"output\":[]}}\n" +
	"\n" +
	"event: response.completed\n" +
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ok\",\"status\":\"completed\",\"output\":[]}}\n" +
	"\n"

// bufferedResponsesFramingChatShape would be rejected if the Responses framing
// normalizer were applied to a non-Responses buffered stream: an event-only
// block followed by a data block whose JSON object has no unique type member.
const bufferedResponsesFramingChatShape = "event: chat.completion.chunk\n" +
	"\n" +
	"data: {\"id\":\"chat_1\",\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n" +
	"\n" +
	"data: [DONE]\n" +
	"\n"

func newConcreteTestExecutionForwarder(t *testing.T) *ExecutionForwarder {
	t.Helper()
	runtime, err := bifrostexecutor.NewRuntime(context.Background(), channel.NewRegistry())
	if err != nil {
		t.Fatalf("initialize execution runtime: %v", err)
	}
	t.Cleanup(runtime.Shutdown)
	return NewExecutionForwarder(runtime)
}

// newBufferedResponsesFramingRuntime builds a Responses gateway whose handler is
// returned so tests can install a capture factory. Only the forwarder varies
// between the capture and release/retry proofs.
func newBufferedResponsesFramingRuntime(
	t *testing.T,
	forwarder AttemptForwarder,
	retryBudget int,
	groups ...dialectGatewayGroup,
) (*gin.Engine, *Handler) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	keyService := encryptiontest.Service(t, "responses-framing-integration-master-key")
	configs := make([]state.GroupConfig, 0, len(groups))
	entries := make([]state.CredentialEntry, 0, len(groups))
	credentialConfigs := make([]state.CredentialConfig, 0, len(groups))
	for index, group := range groups {
		baseURL := testUpstreamBaseURL(group.upstreamURL, protocol.OpenAIResponses)
		_, params := testChannelConfig(t, protocol.OpenAIResponses, baseURL)
		models := group.models
		if len(models) == 0 {
			models = []state.ModelConfig{{ID: "gpt-4o"}}
		}
		configs = append(configs, state.GroupConfig{
			ConnectionType: "api_key", ID: group.id, Name: group.name,
			ChannelID: channel.OpenAI, Params: params, Models: models,
			Settings: group.settings, Enabled: true,
		})
		credentialID := uint(index + 1)
		apiKey := ""
		if len(group.apiKeys) > 0 {
			apiKey = group.apiKeys[0]
		}
		entries = append(entries, testCredentialEntry(t, keyService, credentialID, group.id, apiKey))
		credentialConfigs = append(credentialConfigs, testCredentialConfig(credentialID, group.id))
	}
	manager := state.NewManager()
	if _, err := manager.Publish(state.CompileInput{
		SystemSettings:  config.Settings{state.SettingRetryCount: retryBudget},
		ChannelRegistry: channel.NewRegistry(), Groups: configs,
		Credentials: credentialConfigs,
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: keyService.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	handler := NewHandler(
		manager, registry, keyService, forwarder,
		dialect.NewSet(dialect.NewOpenAIResponses()),
		health.NewStatsStore(), health.NewMutationCoordinator(),
		nil, nil, nil,
	)
	handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
	handler.newRequestID = func() (string, error) { return fixedRequestID, nil }
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	return engine, handler
}

func newBufferedResponsesFramingUpstream(t *testing.T, body func() string) *httptest.Server {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, body())
		writer.(http.Flusher).Flush()
	}))
	t.Cleanup(upstream.Close)
	return upstream
}

func performBufferedResponsesFramingRequest(t *testing.T, engine *gin.Engine, body string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer gl-client")
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	return recorder
}

func awaitObservedCaptureAttempt(
	t *testing.T,
	factory *captureTestFactory,
	attemptID string,
) *captureTestAttempt {
	t.Helper()
	var session *captureTestSession
	if !waitForCapture(t, func() bool {
		candidate, _ := factory.snapshot()
		if candidate == nil {
			return false
		}
		session = candidate
		return candidate.attemptByID(attemptID) != nil && candidate.isWaited()
	}) {
		t.Fatalf("capture attempt %q did not reach completion", attemptID)
	}
	return session.attemptByID(attemptID)
}

// countEventOnlyBlocks counts SSE blocks that carry an event: field but no
// data: line, which is exactly the shape that makes SDKs parse an empty JSON
// payload. Any line ending and blank-line delimiter is treated equivalently.
func countEventOnlyBlocks(body string) int {
	normalized := strings.ReplaceAll(body, "\r\n", "\n")
	normalized = strings.ReplaceAll(normalized, "\r", "\n")
	count := 0
	for _, block := range strings.Split(normalized, "\n\n") {
		hasEvent, hasData := false, false
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "event:"):
				hasEvent = true
			case strings.HasPrefix(line, "data:"):
				hasData = true
			}
		}
		if hasEvent && !hasData {
			count++
		}
	}
	return count
}

// R1: the actual gateway normalizes the incident shape before spool/release and
// completes the Responses terminal validation, while the raw upstream capture
// for the same attempt still holds the untouched malformed bytes.
func TestBufferedResponsesFramingNormalizesIncidentShapeBeforeRelease(t *testing.T) {
	upstream := newBufferedResponsesFramingUpstream(t, func() string {
		return bufferedResponsesFramingIncidentFixture
	})
	forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
	engine, handler := newBufferedResponsesFramingRuntime(
		t, forwarder, 1,
		dialectGatewayGroup{id: 1, name: "responses", upstreamURL: upstream.URL, apiKeys: []string{"sk-1"}},
	)
	factory := &captureTestFactory{}
	handler.captureFactory = factory

	recorder := performBufferedResponsesFramingRequest(
		t, engine, `{"model":"gpt-4o","input":"hello","stream":true,"store":false}`,
	)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %q", recorder.Code, recorder.Body.String())
	}
	want := bufferedStreamHeartbeat + bufferedResponsesFramingIncidentNormalized
	if got := recorder.Body.String(); got != want {
		t.Fatalf("downstream body mismatch:\n got = %q\nwant = %q", got, want)
	}
	if strings.Contains(recorder.Body.String(), "\n\n\ndata:") {
		t.Fatalf("downstream body still splits the event-only and data blocks: %q", recorder.Body.String())
	}
	if count := countEventOnlyBlocks(recorder.Body.String()); count != 0 {
		t.Fatalf("downstream body has %d event-only blocks, want 0: %q", count, recorder.Body.String())
	}
	if len(forwarder.results) != 1 {
		t.Fatalf("forward results = %d, want one", len(forwarder.results))
	}
	// The payload was released and the completed terminal passed validateEOF.
	if !forwarder.results[0].PayloadReleased || forwarder.results[0].Stream.EndReason != StreamEndCleanEOF {
		t.Fatalf("incident result = %#v, want released clean EOF", forwarder.results[0])
	}

	forwardAttempt := awaitObservedCaptureAttempt(t, factory, fixedRequestID+":1")
	if forwardAttempt.metadata.Sequence != 1 {
		t.Fatalf("captured attempt sequence = %d, want 1", forwardAttempt.metadata.Sequence)
	}
	if got := string(forwardAttempt.responseBodyValue()); got != bufferedResponsesFramingIncidentFixture {
		t.Fatalf("raw upstream capture mismatch:\n got = %q\nwant = %q", got, bufferedResponsesFramingIncidentFixture)
	}
}

// R2/R3: an unprovable event/data mismatch is rejected before release, the
// hidden spool bytes never reach the client, and the raw capture is untouched.
func TestBufferedResponsesFramingRejectsAmbiguousGapBeforeRelease(t *testing.T) {
	upstream := newBufferedResponsesFramingUpstream(t, func() string {
		return bufferedResponsesFramingAmbiguousFixture
	})
	forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
	engine, handler := newBufferedResponsesFramingRuntime(
		t, forwarder, 1,
		dialectGatewayGroup{id: 1, name: "responses", upstreamURL: upstream.URL, apiKeys: []string{"sk-1"}},
	)
	factory := &captureTestFactory{}
	handler.captureFactory = factory

	recorder := performBufferedResponsesFramingRequest(
		t, engine, `{"model":"gpt-4o","input":"hello","stream":true,"store":false}`,
	)
	want := bufferedStreamHeartbeat + bufferedResponsesFramingAmbiguousErrorEvent
	if got := recorder.Body.String(); got != want {
		t.Fatalf("failure response mismatch:\n got = %q\nwant = %q", got, want)
	}
	for _, leaked := range []string{
		"event: response.output_text.delta\nid: 5\n",
		"response.output_text.delta",
	} {
		if strings.Contains(recorder.Body.String(), leaked) {
			t.Fatalf("rejected framing leaked %q: %q", leaked, recorder.Body.String())
		}
	}
	if count := countEventOnlyBlocks(recorder.Body.String()); count != 0 {
		t.Fatalf("failure response has %d event-only blocks, want 0: %q", count, recorder.Body.String())
	}
	if len(forwarder.results) != 1 {
		t.Fatalf("forward results = %d, want one", len(forwarder.results))
	}
	first := forwarder.results[0]
	if first.PayloadReleased {
		t.Fatalf("rejected framing released the payload: %#v", first)
	}
	if !first.HTTPCommitted {
		t.Fatalf("heartbeat should have committed HTTP: %#v", first)
	}
	if !errors.Is(first.Err, ErrUpstreamProtocol) {
		t.Fatalf("rejected framing error = %v, want ErrUpstreamProtocol", first.Err)
	}

	forwardAttempt := awaitObservedCaptureAttempt(t, factory, fixedRequestID+":1")
	if got := string(forwardAttempt.responseBodyValue()); got != bufferedResponsesFramingAmbiguousFixture {
		t.Fatalf("raw upstream capture mismatch:\n got = %q\nwant = %q", got, bufferedResponsesFramingAmbiguousFixture)
	}
}

// R2/R3: when the attempt already committed accepted bytes to the spool, a later
// framing rejection still never releases that hidden spool and reports the
// existing protocol terminal.
func TestBufferedResponsesFramingRejectionHidesSpooledBytes(t *testing.T) {
	executor := fakeExecutionExecutor{stream: func(
		_ context.Context,
		_ execution.AttemptSpec,
		sink execution.StreamSink,
	) execution.StreamResult {
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			return framedStreamSinkFailure(err)
		}
		for index, chunk := range [][]byte{
			[]byte("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_hidden\"}}\n\n"),
			[]byte("event: response.output_text.delta\nid: 5\n\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n"),
		} {
			if err := sink(execution.StreamEvent{Sequence: uint64(index + 2), Kind: execution.StreamEventData, Data: chunk}); err != nil {
				return framedStreamSinkFailure(err)
			}
		}
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	input := responsesExecutionForwardInput()
	input.BufferedStream = true
	writer := httptest.NewRecorder()
	result := NewExecutionForwarder(executor).ForwardStream(context.Background(), input, writer)

	if !errors.Is(result.Err, ErrUpstreamProtocol) {
		t.Fatalf("err = %v, want ErrUpstreamProtocol", result.Err)
	}
	if result.PayloadReleased {
		t.Fatalf("rejected framing released payload: %#v", result)
	}
	if !result.HTTPCommitted {
		t.Fatalf("heartbeat did not commit HTTP: %#v", result)
	}
	if result.Stream.EndReason != StreamEndUpstreamProtocolError {
		t.Fatalf("end reason = %v, want %v", result.Stream.EndReason, StreamEndUpstreamProtocolError)
	}
	if result.Stream.ResponseID != "resp_hidden" {
		t.Fatalf("response ID = %q, want resp_hidden", result.Stream.ResponseID)
	}
	if result.BufferedPeakBytes <= 0 {
		t.Fatalf("expected hidden spooled bytes: %#v", result)
	}
	body := writer.Body.String()
	if body != bufferedStreamHeartbeat {
		t.Fatalf("client body = %q, want only the heartbeat before the handler failure envelope", body)
	}
	for _, leaked := range []string{"response.created", "output_text.delta", "resp_hidden"} {
		if strings.Contains(body, leaked) {
			t.Fatalf("hidden spool leaked %q to the client: %q", leaked, body)
		}
	}
}

func framedStreamSinkFailure(err error) execution.StreamResult {
	return execution.StreamResult{
		DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK,
		Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()},
	}
}

// R3: framing rejection follows the existing buffered retry rules. A
// replay-eligible request may switch candidates before release; an ineligible
// request and candidate exhaustion stay unreleased and return the existing
// buffered failure envelope.
func TestBufferedResponsesFramingRejectionUsesExistingRetryRules(t *testing.T) {
	t.Run("replay eligible switches candidate", func(t *testing.T) {
		var calls atomic.Int32
		upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "text/event-stream")
			if calls.Add(1) == 1 {
				_, _ = io.WriteString(writer, bufferedResponsesFramingAmbiguousFixture)
			} else {
				_, _ = io.WriteString(writer, bufferedResponsesFramingSuccessFixture)
			}
			writer.(http.Flusher).Flush()
		}))
		t.Cleanup(upstream.Close)

		forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
		engine, _ := newBufferedResponsesFramingRuntime(
			t, forwarder, 2,
			dialectGatewayGroup{id: 1, name: "responses-first", upstreamURL: upstream.URL, apiKeys: []string{"sk-1"}},
			dialectGatewayGroup{id: 2, name: "responses-second", upstreamURL: upstream.URL, apiKeys: []string{"sk-2"}},
		)
		recorder := performBufferedResponsesFramingRequest(
			t, engine, `{"model":"gpt-4o","input":"hello","stream":true,"store":false}`,
		)
		if calls.Load() != 2 {
			t.Fatalf("upstream calls = %d, want 2", calls.Load())
		}
		want := bufferedStreamHeartbeat + bufferedResponsesFramingSuccessFixture
		if got := recorder.Body.String(); got != want {
			t.Fatalf("downstream body mismatch:\n got = %q\nwant = %q", got, want)
		}
		if len(forwarder.results) != 2 {
			t.Fatalf("forward results = %d, want 2", len(forwarder.results))
		}
		first, second := forwarder.results[0], forwarder.results[1]
		if first.PayloadReleased || !first.HTTPCommitted ||
			!errors.Is(first.Err, ErrUpstreamProtocol) {
			t.Fatalf("first attempt = %#v, want committed unreleased protocol failure", first)
		}
		if !second.PayloadReleased || second.Stream.EndReason != StreamEndCleanEOF {
			t.Fatalf("second attempt = %#v, want released clean payload", second)
		}
		if forwarder.inputs[0].Group.ID == forwarder.inputs[1].Group.ID {
			t.Fatalf("candidate groups = %d/%d, want a switch",
				forwarder.inputs[0].Group.ID, forwarder.inputs[1].Group.ID)
		}
	})

	t.Run("replay ineligible does not switch", func(t *testing.T) {
		var calls atomic.Int32
		upstream := newBufferedResponsesFramingUpstream(t, func() string {
			calls.Add(1)
			return bufferedResponsesFramingAmbiguousFixture
		})
		forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
		engine, _ := newBufferedResponsesFramingRuntime(
			t, forwarder, 2,
			dialectGatewayGroup{id: 1, name: "responses", upstreamURL: upstream.URL, apiKeys: []string{"sk-1"}},
		)
		recorder := performBufferedResponsesFramingRequest(
			t, engine, `{"model":"gpt-4o","input":"hello","stream":true}`,
		)
		if calls.Load() != 1 {
			t.Fatalf("upstream calls = %d, want 1 for an ineligible request", calls.Load())
		}
		if !strings.Contains(recorder.Body.String(), "buffered_stream_failed") {
			t.Fatalf("downstream body = %q, want the buffered failure envelope", recorder.Body.String())
		}
		if len(forwarder.results) != 1 || forwarder.results[0].PayloadReleased {
			t.Fatalf("ineligible results = %#v, want one unreleased attempt", forwarder.results)
		}
	})

	t.Run("candidate exhaustion stays unreleased", func(t *testing.T) {
		var calls atomic.Int32
		upstream := newBufferedResponsesFramingUpstream(t, func() string {
			calls.Add(1)
			return bufferedResponsesFramingAmbiguousFixture
		})
		forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
		engine, _ := newBufferedResponsesFramingRuntime(
			t, forwarder, 2,
			dialectGatewayGroup{id: 1, name: "responses-first", upstreamURL: upstream.URL, apiKeys: []string{"sk-1"}},
			dialectGatewayGroup{id: 2, name: "responses-second", upstreamURL: upstream.URL, apiKeys: []string{"sk-2"}},
		)
		recorder := performBufferedResponsesFramingRequest(
			t, engine, `{"model":"gpt-4o","input":"hello","stream":true,"store":false}`,
		)
		if calls.Load() != 2 {
			t.Fatalf("upstream calls = %d, want the two-candidate budget", calls.Load())
		}
		if !strings.Contains(recorder.Body.String(), "buffered_stream_failed") {
			t.Fatalf("downstream body = %q, want the buffered failure envelope", recorder.Body.String())
		}
		for index, result := range forwarder.results {
			if result.PayloadReleased {
				t.Fatalf("exhausted candidate %d released the payload: %#v", index, result)
			}
		}
	})
}

// R2: legal Responses wire, including an upstream standalone heartbeat block, is
// forwarded unchanged with exactly one GPTL heartbeat prefix.
func TestBufferedResponsesFramingLeavesLegalStreamUnchanged(t *testing.T) {
	upstream := newBufferedResponsesFramingUpstream(t, func() string {
		return bufferedResponsesFramingLegalFixture
	})
	forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
	engine, _ := newBufferedResponsesFramingRuntime(
		t, forwarder, 1,
		dialectGatewayGroup{id: 1, name: "responses", upstreamURL: upstream.URL, apiKeys: []string{"sk-1"}},
	)
	recorder := performBufferedResponsesFramingRequest(
		t, engine, `{"model":"gpt-4o","input":"hello","stream":true,"store":false}`,
	)
	want := bufferedStreamHeartbeat + bufferedResponsesFramingLegalFixture
	if got := recorder.Body.String(); got != want {
		t.Fatalf("legal stream mismatch:\n got = %q\nwant = %q", got, want)
	}
	if count := strings.Count(recorder.Body.String(), bufferedStreamHeartbeat); count != 2 {
		t.Fatalf("heartbeat occurrences = %d, want GPTL heartbeat plus the upstream heartbeat", count)
	}
}

// R3 negative assertion: the framing normalizer is enabled only for the
// buffered OpenAI Responses wire. Chat, Anthropic, Images, Gemini and the live
// Responses path must never instantiate it.
func TestResponsesSSEFramingEnabledOnlyForBufferedResponses(t *testing.T) {
	for _, test := range []struct {
		name    string
		input   ForwardInput
		enabled bool
	}{
		{name: "buffered responses", input: ForwardInput{BufferedStream: true, ClientProtocol: protocol.OpenAIResponses}, enabled: true},
		{name: "live responses", input: ForwardInput{ClientProtocol: protocol.OpenAIResponses}},
		{name: "buffered chat", input: ForwardInput{BufferedStream: true, ClientProtocol: protocol.OpenAICompletions}},
		{name: "buffered anthropic", input: ForwardInput{BufferedStream: true, ClientProtocol: protocol.Anthropic}},
		{name: "buffered images", input: ForwardInput{BufferedStream: true, ClientProtocol: protocol.OpenAIImages}},
		{name: "buffered gemini", input: ForwardInput{BufferedStream: true, ClientProtocol: protocol.Gemini}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := responsesSSEFramingEnabled(test.input); got != test.enabled {
				t.Fatalf("responsesSSEFramingEnabled() = %v, want %v", got, test.enabled)
			}
		})
	}
}

// R3 negative assertion: the non-Responses buffered path must not apply the
// framing normalizer. The chosen Chat stream would be rejected if it were.
func TestBufferedResponsesFramingDoesNotTouchChatStream(t *testing.T) {
	upstream := newBufferedResponsesFramingUpstream(t, func() string {
		return bufferedResponsesFramingChatShape
	})
	engine, _ := newStreamingGatewayEngine(t,
		streamGatewayGroup{id: 1, name: "chat", upstreamURL: upstream.URL, apiKey: "sk-chat"},
	)
	recorder := performStreamingRequest(engine)
	want := bufferedStreamHeartbeat + bufferedResponsesFramingChatShape
	if got := recorder.Body.String(); got != want {
		t.Fatalf("Chat buffered stream was changed:\n got = %q\nwant = %q", got, want)
	}
}

// rejectingStreamWriter accepts headers and status but always fails the body
// write, so the framing remainder path can be classified without a real socket.
type rejectingStreamWriter struct {
	header http.Header
}

func (writer *rejectingStreamWriter) Header() http.Header {
	if writer.header == nil {
		writer.header = http.Header{}
	}
	return writer.header
}

func (*rejectingStreamWriter) WriteHeader(int) {}

func (*rejectingStreamWriter) Write([]byte) (int, error) {
	return 0, errors.New("downstream gone")
}

func (*rejectingStreamWriter) Flush() {}

// R2/R3: the framing remainder must flow through the same commit/write/flush
// path as the streaming sink, and its failures must stay classified.
func TestWriteResponsesFramingRemainderUsesSinkPaths(t *testing.T) {
	ready := &execution.StreamEvent{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"text/event-stream"}},
	}
	t.Run("empty remainder is a no-op", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		controller := newStreamWriteController(recorder, time.Second)
		readyCalls := 0
		committed, err := writeResponsesFramingRemainder(controller, ready, nil, false, func() { readyCalls++ })
		if err != nil || committed || readyCalls != 0 || recorder.Body.Len() != 0 {
			t.Fatalf("committed=%v err=%v readyCalls=%d body=%q", committed, err, readyCalls, recorder.Body.String())
		}
	})
	t.Run("uncommitted remainder commits first", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		controller := newStreamWriteController(recorder, time.Second)
		readyCalls := 0
		committed, err := writeResponsesFramingRemainder(controller, ready, []byte("\n"), false, func() { readyCalls++ })
		if err != nil || !committed || readyCalls != 1 {
			t.Fatalf("committed=%v err=%v readyCalls=%d", committed, err, readyCalls)
		}
		if got := recorder.Body.String(); got != "\n" || recorder.Code != http.StatusOK {
			t.Fatalf("status=%d body=%q, want 200 and %q", recorder.Code, got, "\n")
		}
	})
	t.Run("uncommitted without metadata is a protocol failure", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		controller := newStreamWriteController(recorder, time.Second)
		committed, err := writeResponsesFramingRemainder(controller, nil, []byte("\n"), false, nil)
		if committed || !errors.Is(err, ErrUpstreamProtocol) {
			t.Fatalf("committed=%v err=%v, want protocol failure", committed, err)
		}
	})
	t.Run("committed remainder writes through", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		controller := newStreamWriteController(recorder, time.Second)
		readyCalls := 0
		committed, err := writeResponsesFramingRemainder(controller, ready, []byte("\n"), true, func() { readyCalls++ })
		if err != nil || !committed || readyCalls != 0 {
			t.Fatalf("committed=%v err=%v readyCalls=%d", committed, err, readyCalls)
		}
		if got := recorder.Body.String(); got != "\n" {
			t.Fatalf("body = %q, want %q", got, "\n")
		}
	})
	t.Run("downstream write failure stays a write failure", func(t *testing.T) {
		controller := newStreamWriteController(&rejectingStreamWriter{}, time.Second)
		committed, err := writeResponsesFramingRemainder(controller, ready, []byte("\n"), true, nil)
		if !committed {
			t.Fatalf("committed = false, want true (already committed before the write)")
		}
		var failure *streamFailure
		if !errors.As(err, &failure) || failure.kind != streamFailureDownstreamWrite {
			t.Fatalf("err = %v, want downstream write failure", err)
		}
	})
}
