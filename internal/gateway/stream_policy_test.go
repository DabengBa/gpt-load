package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

func streamPolicyForwarder() *scriptedForwarder {
	return &scriptedForwarder{streamResults: []UpstreamResult{{
		StatusCode:     http.StatusOK,
		Header:         http.Header{"Content-Type": {"text/event-stream"}},
		RequestWritten: true,
		DispatchState:  execution.DispatchMaybeSent,
		Committed:      true,
	}}}
}

func streamPolicyDialectGatewayGroup(id uint, name, upstreamURL string, apiKeys ...string) dialectGatewayGroup {
	return dialectGatewayGroup{id: id, name: name, upstreamURL: upstreamURL, apiKeys: apiKeys}
}

// TestStreamPolicyMatrix pins the single policy entry point across the
// protocol/operation/stream matrix, including the reject reason contract.
func TestStreamPolicyMatrix(t *testing.T) {
	tests := []struct {
		name           string
		clientProtocol protocol.Protocol
		operation      execution.Operation
		stream         bool
		wantMode       streamDelivery
		wantReason     string
	}{
		{name: "OpenAI Completions generated stream", clientProtocol: protocol.OpenAICompletions, operation: execution.OperationChatCompletion, stream: true, wantMode: streamDeliveryBuffered},
		{name: "Anthropic Messages generated stream", clientProtocol: protocol.Anthropic, operation: execution.OperationChatCompletion, stream: true, wantMode: streamDeliveryBuffered},
		{name: "OpenAI Responses create stream", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesCreate, stream: true, wantMode: streamDeliveryBuffered},
		{name: "Gemini generated stream", clientProtocol: protocol.Gemini, operation: execution.OperationChatCompletion, stream: true, wantMode: streamDeliveryLiveException},
		{name: "OpenAI Images generate stream", clientProtocol: protocol.OpenAIImages, operation: execution.OperationImagesGenerate, stream: true, wantMode: streamDeliveryLiveException},
		{name: "OpenAI Images edit stream", clientProtocol: protocol.OpenAIImages, operation: execution.OperationImagesEdit, stream: true, wantMode: streamDeliveryLiveException},
		{name: "Responses retrieve stream rejected", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesRetrieve, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_operation_unsupported"},
		{name: "Responses passthrough stream rejected", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesPassthrough, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_operation_unsupported"},
		{name: "Responses input items stream rejected", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesInputItems, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_operation_unsupported"},
		{name: "Responses compact stream rejected", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesCompact, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_operation_unsupported"},
		{name: "Responses input tokens stream rejected", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesInputTokens, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_operation_unsupported"},
		{name: "Responses delete stream rejected", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesDelete, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_operation_unsupported"},
		{name: "Responses cancel stream rejected", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesCancel, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_operation_unsupported"},
		{name: "Gemini count tokens stream rejected", clientProtocol: protocol.Gemini, operation: execution.OperationCountTokens, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_operation_unsupported"},
		{name: "Embeddings stream protocol rejected", clientProtocol: protocol.OpenAIEmbeddings, operation: execution.OperationEmbeddingsCreate, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_protocol_unsupported"},
		{name: "Rerank stream protocol rejected", clientProtocol: protocol.Rerank, operation: execution.OperationRerank, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_protocol_unsupported"},
		{name: "Unknown protocol stream rejected", clientProtocol: protocol.Protocol("unknown-protocol"), operation: execution.OperationChatCompletion, stream: true, wantMode: streamDeliveryReject, wantReason: "streaming_protocol_unsupported"},
		{name: "Non streaming chat is not policy input", clientProtocol: protocol.OpenAICompletions, operation: execution.OperationChatCompletion, stream: false, wantMode: streamDeliveryNotStreaming},
		{name: "Non streaming responses create is not policy input", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesCreate, stream: false, wantMode: streamDeliveryNotStreaming},
		{name: "Non streaming responses retrieve is not policy input", clientProtocol: protocol.OpenAIResponses, operation: execution.OperationResponsesRetrieve, stream: false, wantMode: streamDeliveryNotStreaming},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mode, rejectReason := evaluateStreamDelivery(test.clientProtocol, test.operation, test.stream)
			if mode != test.wantMode {
				t.Fatalf("mode = %v, want %v", mode, test.wantMode)
			}
			gotReason := ""
			if rejectReason != nil {
				gotReason = rejectReason.Code
			}
			if gotReason != test.wantReason {
				t.Fatalf("reject reason = %q, want %q", gotReason, test.wantReason)
			}
		})
	}
}

// TestStreamingOperationUnsupportedReasonContract pins the HTTP contract for
// the two streaming rejection reasons that replaced the former
// protocol-specific rejection code.
func TestStreamingOperationUnsupportedReasonContract(t *testing.T) {
	tests := []struct {
		name string
		got  reason
		code string
	}{
		{name: "operation", got: reasonStreamingOperationUnsupported, code: "streaming_operation_unsupported"},
		{name: "protocol", got: reasonStreamingProtocolUnsupported, code: "streaming_protocol_unsupported"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.got.Status != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", test.got.Status)
			}
			if test.got.Code != test.code {
				t.Fatalf("code = %q, want %q", test.got.Code, test.code)
			}
		})
	}
}

// TestStreamPolicyForcesBufferedGenerationStreams pins the fixed delivery
// contract: the three supported generated HTTP/SSE streams must always enter
// the buffered release gate, regardless of former group/global configuration.
func TestStreamPolicyForcesBufferedGenerationStreams(t *testing.T) {
	tests := []struct {
		name     string
		protocol protocol.Protocol
		path     string
		body     string
		dialects dialect.Set
	}{
		{
			name:     "OpenAI Completions",
			protocol: protocol.OpenAICompletions,
			path:     "/v1/chat/completions",
			body:     `{"model":"public-model","stream":true,"messages":[{"role":"user","content":"ping"}]}`,
			dialects: dialect.NewSet(dialect.NewOpenAI()),
		},
		{
			name:     "Anthropic Messages",
			protocol: protocol.Anthropic,
			path:     "/v1/messages",
			body:     `{"model":"public-model","max_tokens":1,"stream":true,"messages":[{"role":"user","content":"ping"}]}`,
			dialects: dialect.NewSet(dialect.NewAnthropic()),
		},
		{
			name:     "OpenAI Responses Create",
			protocol: protocol.OpenAIResponses,
			path:     "/v1/responses",
			body:     `{"model":"public-model","input":"ping","stream":true,"store":false}`,
			dialects: dialect.NewSet(dialect.NewOpenAIResponses()),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := streamPolicyForwarder()
			engine, _ := newDialectGatewayEngineWithForwarder(
				t,
				test.protocol,
				"public-model",
				test.dialects,
				forwarder,
				streamPolicyDialectGatewayGroup(1, "policy", "https://provider.example/v1", "sk-provider"),
			)
			request := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			request.Header.Set("Authorization", "Bearer gl-client")
			recorder := httptest.NewRecorder()

			engine.ServeHTTP(recorder, request)

			if len(forwarder.streamInputs) != 1 {
				t.Fatalf(
					"stream attempts = %d, want one; status=%d body=%s",
					len(forwarder.streamInputs), recorder.Code, recorder.Body.String(),
				)
			}
			if !forwarder.streamInputs[0].BufferedStream {
				t.Fatalf(
					"BufferedStream = false, want the generated stream to be buffered; status=%d body=%s",
					recorder.Code, recorder.Body.String(),
				)
			}
		})
	}
}

// TestStreamPolicyKeepsLiveExceptionsUnbuffered pins the fixed live exception:
// Gemini and OpenAI Images streaming keep real-time pass-through.
func TestStreamPolicyKeepsLiveExceptionsUnbuffered(t *testing.T) {
	imageGroup := dialectGatewayGroup{
		id: 1, name: "images", channelID: channel.OpenAI,
		params: json.RawMessage(`{"base_url":"https://provider.example/v1"}`),
		models: []state.ModelConfig{{ID: "gpt-image-2", Alias: "public-image"}},
	}
	tests := []struct {
		name     string
		protocol protocol.Protocol
		path     string
		body     string
		dialects dialect.Set
		group    dialectGatewayGroup
	}{
		{
			name:     "Gemini",
			protocol: protocol.Gemini,
			path:     "/v1beta/models/public-model:streamGenerateContent",
			body:     `{"contents":[{"role":"user","parts":[{"text":"ping"}]}]}`,
			dialects: dialect.NewSet(dialect.NewGemini()),
			group:    streamPolicyDialectGatewayGroup(1, "gemini", "https://provider.example", "sk-gemini"),
		},
		{
			name:     "OpenAI Images",
			protocol: protocol.OpenAIImages,
			path:     "/v1/images/generations",
			body:     `{"model":"public-image","prompt":"draw","stream":true}`,
			dialects: dialect.NewSet(dialect.NewOpenAIImages()),
			group: dialectGatewayGroup{
				id: imageGroup.id, name: imageGroup.name, channelID: imageGroup.channelID,
				params: imageGroup.params, models: imageGroup.models, apiKeys: []string{"sk-image"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := streamPolicyForwarder()
			engine, _ := newDialectGatewayEngineWithForwarder(
				t, test.protocol, "public-model", test.dialects, forwarder, test.group,
			)
			request := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			request.Header.Set("Authorization", "Bearer gl-client")
			recorder := httptest.NewRecorder()

			engine.ServeHTTP(recorder, request)

			if len(forwarder.streamInputs) != 1 {
				t.Fatalf(
					"stream attempts = %d, want one; status=%d body=%s",
					len(forwarder.streamInputs), recorder.Code, recorder.Body.String(),
				)
			}
			if forwarder.streamInputs[0].BufferedStream {
				t.Fatalf(
					"BufferedStream = true, want live pass-through; status=%d body=%s",
					recorder.Code, recorder.Body.String(),
				)
			}
		})
	}
}

// TestStreamPolicyRejectsStreamingOperationBeforeDispatch pins that a stream
// request for a non-generating Responses operation is rejected before any
// provider attempt.
func TestStreamPolicyRejectsStreamingOperationBeforeDispatch(t *testing.T) {
	tests := []struct {
		name string
		path string
	}{
		{name: "retrieve", path: "/v1/responses/resp_policy_1?stream=true"},
		{name: "input items", path: "/v1/responses/resp_policy_1/input_items?stream=true"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := streamPolicyForwarder()
			engine, _ := newDialectGatewayEngineWithForwarder(
				t,
				protocol.OpenAIResponses,
				"public-model",
				dialect.NewSet(dialect.NewOpenAIResponses()),
				forwarder,
				streamPolicyDialectGatewayGroup(1, "policy", "https://provider.example/v1", "sk-provider"),
			)
			request := httptest.NewRequest(http.MethodGet, test.path, nil)
			request.Header.Set("Authorization", "Bearer gl-client")
			recorder := httptest.NewRecorder()

			engine.ServeHTTP(recorder, request)

			if recorder.Code != http.StatusBadRequest ||
				!strings.Contains(recorder.Body.String(), `"code":"streaming_operation_unsupported"`) {
				t.Fatalf("status/body = %d/%s, want 400 streaming_operation_unsupported", recorder.Code, recorder.Body.String())
			}
			if len(forwarder.streamInputs) != 0 || len(forwarder.inputs) != 0 {
				t.Fatalf("provider attempts = stream:%d unary:%d, want none", len(forwarder.streamInputs), len(forwarder.inputs))
			}
		})
	}
}

// TestStreamPolicyMalformedStreamRemainsInvalidProtocolRequest pins precedence:
// a malformed stream parameter is still a malformed dialect request and must
// win over the operation-level streaming rejection.
func TestStreamPolicyMalformedStreamRemainsInvalidProtocolRequest(t *testing.T) {
	forwarder := streamPolicyForwarder()
	engine, _ := newDialectGatewayEngineWithForwarder(
		t,
		protocol.OpenAIResponses,
		"public-model",
		dialect.NewSet(dialect.NewOpenAIResponses()),
		forwarder,
		streamPolicyDialectGatewayGroup(1, "policy", "https://provider.example/v1", "sk-provider"),
	)
	request := httptest.NewRequest(http.MethodGet, "/v1/responses/resp_policy_1?stream=maybe", nil)
	request.Header.Set("Authorization", "Bearer gl-client")
	recorder := httptest.NewRecorder()

	engine.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest ||
		!strings.Contains(recorder.Body.String(), `"code":"invalid_protocol_request"`) {
		t.Fatalf("status/body = %d/%s, want 400 invalid_protocol_request", recorder.Code, recorder.Body.String())
	}
	if len(forwarder.streamInputs) != 0 || len(forwarder.inputs) != 0 {
		t.Fatalf("provider attempts = stream:%d unary:%d, want none", len(forwarder.streamInputs), len(forwarder.inputs))
	}
}

func TestStreamPolicyRejectionRecordsParsedRequestMetadata(t *testing.T) {
	forwarder := streamPolicyForwarder()
	sink := &recordingRequestLogSink{}
	engine, handler, _, _ := newRequestLogHandlerTestRuntime(
		t, forwarder, &recordingAccessKeyRPMLimiter{}, sink, "sk-provider",
	)
	handler.dialects = dialect.NewSet(dialect.NewOpenAIResponses())

	request := httptest.NewRequest(http.MethodGet, "/v1/responses/resp_policy_1/input_items?stream=true", nil)
	request.Header.Set("Authorization", "Bearer gl-client")
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest ||
		!strings.Contains(recorder.Body.String(), `"code":"streaming_operation_unsupported"`) {
		t.Fatalf("status/body = %d/%s", recorder.Code, recorder.Body.String())
	}
	events := sink.snapshot()
	if len(events) != 1 {
		t.Fatalf("request log events = %#v, want one event", events)
	}
	event := events[0]
	if event.Operation != execution.OperationResponsesInputItems || !event.Stream ||
		event.StatusCode != http.StatusBadRequest || event.ErrorCode != "streaming_operation_unsupported" {
		t.Fatalf("request log event = %#v", event)
	}
}

// TestStreamPolicyWebsocketNeverEntersPolicy pins that an upgrade request to
// the Responses endpoint is short-circuited by the WebSocket path and never
// reaches the HTTP/SSE streaming delivery policy.
func TestStreamPolicyWebsocketNeverEntersPolicy(t *testing.T) {
	forwarder := streamPolicyForwarder()
	engine, _ := newDialectGatewayEngineWithForwarder(
		t,
		protocol.OpenAIResponses,
		"public-model",
		dialect.NewSet(dialect.NewOpenAIResponses()),
		forwarder,
		streamPolicyDialectGatewayGroup(1, "policy", "https://provider.example/v1", "sk-provider"),
	)
	request := httptest.NewRequest(http.MethodGet, "/v1/responses?stream=true", nil)
	request.Header.Set("Authorization", "Bearer gl-client")
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", "websocket")
	request.Header.Set("Sec-WebSocket-Version", "13")
	request.Header.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
	recorder := httptest.NewRecorder()

	engine.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusForbidden ||
		!strings.Contains(recorder.Body.String(), `"code":"websocket_disabled"`) {
		t.Fatalf("status/body = %d/%s, want 403 websocket_disabled", recorder.Code, recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "streaming_operation_unsupported") {
		t.Fatalf("websocket request entered the HTTP/SSE stream policy: %s", recorder.Body.String())
	}
}
