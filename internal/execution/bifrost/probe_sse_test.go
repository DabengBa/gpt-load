package bifrost

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/usage"
)

func responsesSSEEvent(name, data string) string {
	return "event: " + name + "\n" + "data: " + data + "\n\n"
}

func responsesSSECompletedData() string {
	return `{"type":"response.completed","response":{"id":"resp_sse","object":"response","status":"completed","model":"probe-upstream","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"4","annotations":[]}]}],"usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21}}}`
}

func responsesSSEHead() string {
	created := `{"type":"response.created","response":{"id":"resp_sse","object":"response","status":"in_progress","model":"probe-upstream"}}`
	inProgress := `{"type":"response.in_progress","response":{"id":"resp_sse","object":"response","status":"in_progress","model":"probe-upstream"}}`
	itemAdded := `{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}`
	partAdded := `{"type":"response.content_part.added","item_id":"msg_1","content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}`
	delta := `{"type":"response.output_text.delta","item_id":"msg_1","content_index":0,"output_index":0,"delta":"4"}`
	textDone := `{"type":"response.output_text.done","item_id":"msg_1","content_index":0,"output_index":0,"text":"4"}`
	itemDone := `{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"4","annotations":[]}]}}`
	return responsesSSEEvent("response.created", created) +
		responsesSSEEvent("response.in_progress", inProgress) +
		responsesSSEEvent("response.output_item.added", itemAdded) +
		responsesSSEEvent("response.content_part.added", partAdded) +
		responsesSSEEvent("response.output_text.delta", delta) +
		responsesSSEEvent("response.output_text.done", textDone) +
		responsesSSEEvent("response.output_item.done", itemDone)
}

func responsesSSEReadyTerminal() string {
	return responsesSSEHead() + responsesSSEEvent("response.completed", responsesSSECompletedData())
}

func responsesProbeCRLFFixture() string {
	return strings.ReplaceAll(responsesSSEReadyTerminal(), "\n", "\r\n")
}

// TestNativeOpenAIResponsesProbeAcceptsCompletedSSE proves an OpenAI Responses
// probe tolerates an upstream that ignores the non-streaming intent and
// returns HTTP 200 text/event-stream ending in response.completed: the SSE is
// normalized to the terminal response JSON before alias rewriting, usage and
// request IDs survive, and the raw capture observer still sees the untouched
// original SSE bytes.
func TestNativeOpenAIResponsesProbeAcceptsCompletedSSE(t *testing.T) {
	t.Parallel()

	fixtures := map[string]func() string{
		"lf":   responsesSSEReadyTerminal,
		"gzip": responsesSSEReadyTerminal,
		"gzip_json": func() string {
			var event struct {
				Response json.RawMessage `json:"response"`
			}
			_ = json.Unmarshal([]byte(responsesSSECompletedData()), &event)
			return string(event.Response)
		},
		"cr":   func() string { return strings.ReplaceAll(responsesSSEReadyTerminal(), "\n", "\r") },
		"crlf": responsesProbeCRLFFixture,
		"comment_keepalive": func() string {
			return ": keep-alive\n\n" + responsesSSEReadyTerminal()
		},
		"multiline_data": func() string {
			completed := responsesSSECompletedData()
			split := strings.Index(completed, `"response":`)
			return responsesSSEHead() +
				"event: response.completed\n" +
				"data: " + completed[:split] + "\n" +
				"data: " + completed[split:] + "\n\n"
		},
	}

	for name, build := range fixtures {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			fixture := build()
			wire := []byte(fixture)
			if strings.HasPrefix(name, "gzip") {
				var compressed bytes.Buffer
				writer := gzip.NewWriter(&compressed)
				_, _ = writer.Write(wire)
				_ = writer.Close()
				wire = compressed.Bytes()
			}
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				calls.Add(1)
				if request.URL.Path != "/v1/responses" {
					t.Errorf("probe path = %s, want /v1/responses", request.URL.Path)
				}
				body, _ := io.ReadAll(request.Body)
				var payload map[string]any
				if err := json.Unmarshal(body, &payload); err != nil {
					t.Errorf("decode probe request: %v", err)
				}
				if payload["input"] == nil || payload["max_output_tokens"] != float64(testProbeOutputTokens) {
					t.Errorf("probe controls = %#v", payload)
				}
				if _, hasMessages := payload["messages"]; hasMessages {
					t.Errorf("Responses probe must not carry Chat messages: %s", body)
				}
				writer.Header().Set("Content-Type", "text/event-stream")
				if name == "gzip_json" {
					writer.Header().Set("Content-Type", "application/json")
				}
				if strings.HasPrefix(name, "gzip") {
					writer.Header().Set("Content-Encoding", "gzip")
				}
				writer.Header().Set("X-Request-Id", "sse-probe-1")
				_, _ = writer.Write(wire)
			}))
			defer server.Close()

			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
			spec := utilitySpec(channel.OpenAI, protocol.OpenAIResponses, execution.OperationProbe, "", "", nil)
			spec.ClientModel = "probe-client"
			spec.UpstreamModel = "probe-upstream"

			observer := newWiringObserver()
			ctx := execution.WithHTTPObserver(context.Background(), observer)
			result := runtime.Execute(ctx, spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
			}
			if !result.ProbeAnswerPresent || result.ProbeResponseInvalid {
				t.Fatalf("probe evidence = %+v body=%s", result, result.Body)
			}
			if result.UpstreamProtocol != protocol.OpenAIResponses || result.UpstreamRequestID != "sse-probe-1" {
				t.Fatalf("protocol/request id = %s/%s", result.UpstreamProtocol, result.UpstreamRequestID)
			}
			if bytes.Contains(result.Body, []byte("event: ")) || bytes.Contains(result.Body, []byte("data: ")) {
				t.Fatalf("normalized body still contains SSE framing: %s", result.Body)
			}
			var normalized map[string]any
			if err := json.Unmarshal(result.Body, &normalized); err != nil {
				t.Fatalf("normalized body is not JSON: %v; body=%s", err, result.Body)
			}
			if normalized["status"] != "completed" {
				t.Fatalf("normalized status = %v, want completed", normalized["status"])
			}
			if normalized["model"] != "probe-client" {
				t.Fatalf("normalized model = %v, want alias-applied probe-client", normalized["model"])
			}
			if !bytes.Contains(result.Body, []byte(`"text":"4"`)) {
				t.Fatalf("normalized body lost generated text: %s", result.Body)
			}
			assertUsage(t, result.Usage, usage.Tokens{UncachedInput: 20, Output: 1})

			_, observedResponse, _, _, _, _ := observer.snapshot()
			if !bytes.Equal(observedResponse, wire) {
				t.Fatalf("raw capture was overwritten by normalization\n got %q\nwant %q", observedResponse, fixture)
			}
		})
	}
}

// TestNativeOpenAIResponsesProbeRejectsInvalidSSE proves only a fully
// terminated response.completed stream with valid framing counts as a passed
// probe: truncated, failed, malformed, post-terminal and response.incomplete
// streams all fail while keeping the real upstream HTTP status.
func TestNativeOpenAIResponsesProbeRejectsInvalidSSE(t *testing.T) {
	t.Parallel()

	created := `{"type":"response.created","response":{"id":"resp_sse","object":"response","status":"in_progress","model":"probe-upstream"}}`
	completed := `{"type":"response.completed","response":{"id":"resp_sse","object":"response","status":"completed","model":"probe-upstream","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"4","annotations":[]}]}],"usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21}}}`
	incomplete := `{"type":"response.incomplete","response":{"id":"resp_sse","object":"response","status":"incomplete","model":"probe-upstream","output":[{"id":"msg_1","type":"message","role":"assistant","status":"incomplete","content":[{"type":"output_text","text":"4","annotations":[]}]}],"incomplete_details":{"reason":"max_output_tokens"}}}`
	failed := `{"type":"response.failed","response":{"id":"resp_sse","object":"response","status":"failed","model":"probe-upstream","output":[]},"error":{"message":"upstream boom"}}`
	conflict := `{"type":"response.failed","response":{"id":"resp_sse","object":"response","status":"completed","model":"probe-upstream","output":[]}}`

	tests := []struct {
		name    string
		fixture string
	}{
		{name: "truncated_no_terminal", fixture: responsesSSEEvent("response.created", created)},
		{name: "failed_terminal", fixture: responsesSSEEvent("response.failed", failed)},
		{name: "malformed_json", fixture: "data: {not-json\n\n"},
		{name: "name_type_conflict", fixture: responsesSSEEvent("response.completed", conflict)},
		{name: "completed_event_error", fixture: responsesSSEEvent("response.completed", strings.Replace(completed, `"type":"response.completed"`, `"type":"response.completed","error":{"message":"failed"}`, 1))},
		{name: "completed_response_error", fixture: responsesSSEEvent("response.completed", strings.Replace(completed, `"status":"completed"`, `"status":"completed","error":{"message":"failed"}`, 1))},
		{name: "data_after_terminal", fixture: responsesSSEEvent("response.completed", completed) + responsesSSEEvent("response.output_text.delta", `{"type":"response.output_text.delta","delta":"5"}`)},
		{name: "incomplete_with_partial_text", fixture: responsesSSEEvent("response.incomplete", incomplete)},
		{
			name: "eof_half_frame",
			fixture: responsesSSEHead() +
				"event: response.completed\n" +
				`data: {"type":"response.completed","response":{"id":"resp_sse","object":"resp`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				writer.Header().Set("X-Request-Id", "sse-probe-bad-1")
				_, _ = io.WriteString(writer, test.fixture)
			}))
			defer server.Close()

			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
			spec := utilitySpec(channel.OpenAI, protocol.OpenAIResponses, execution.OperationProbe, "", "", nil)
			spec.ClientModel = "probe-client"
			spec.UpstreamModel = "probe-upstream"
			result := runtime.Execute(context.Background(), spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if result.Error == nil || result.Error.Kind != execution.ErrorKindProvider {
				t.Fatalf("probe accepted invalid SSE or misclassified: %+v", result)
			}
			if result.StatusCode != http.StatusOK {
				t.Fatalf("status = %d, want the real upstream 200", result.StatusCode)
			}
			if result.Error.OriginHint != execution.ErrorOriginUpstream {
				t.Fatalf("error origin = %q, want upstream", result.Error.OriginHint)
			}
			if result.ProbeAnswerPresent || result.UpstreamRequestID != "sse-probe-bad-1" {
				t.Fatalf("probe evidence/request id = %+v/%s", result, result.UpstreamRequestID)
			}
		})
	}
}

// TestNativeOpenAIResponsesProbeEmptyAnswerStaysNoAnswer proves a fully
// terminated response.completed stream that carries no generated text is a
// plain no-answer (structurally valid, ProbeAnswerPresent=false), never a
// probe success and never a protocol error.
func TestNativeOpenAIResponsesProbeEmptyAnswerStaysNoAnswer(t *testing.T) {
	t.Parallel()

	fixture := responsesSSEEvent("response.completed",
		`{"type":"response.completed","response":{"id":"resp_sse","object":"response","status":"completed","model":"probe-upstream","output":[]}}`)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("X-Request-Id", "sse-probe-empty-1")
		_, _ = io.WriteString(writer, fixture)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
	spec := utilitySpec(channel.OpenAI, protocol.OpenAIResponses, execution.OperationProbe, "", "", nil)
	spec.ClientModel = "probe-client"
	spec.UpstreamModel = "probe-upstream"
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if result.Error != nil || result.StatusCode != http.StatusOK {
		t.Fatalf("empty answer must stay a plain no-answer, not an error: %+v", result)
	}
	if result.ProbeAnswerPresent || result.ProbeResponseInvalid {
		t.Fatalf("empty answer flags = present:%v invalid:%v", result.ProbeAnswerPresent, result.ProbeResponseInvalid)
	}
}

// TestGatewayResponsesProbeAcceptsCompletedSSE proves the existing
// MultiProtocolGateway Responses probe also normalizes a 200 SSE stream into
// the terminal response JSON before its structural validation, while the raw
// capture observer keeps the original bytes.
func TestGatewayResponsesProbeAcceptsCompletedSSE(t *testing.T) {
	t.Parallel()

	fixture := responsesSSEReadyTerminal()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			t.Errorf("probe path = %s, want /v1/responses", request.URL.Path)
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("X-Request-Id", "sse-gateway-1")
		_, _ = io.WriteString(writer, fixture)
	}))
	defer server.Close()

	manager, spec := gatewayProbeForTest(t, channel.NewAPI, protocol.OpenAIResponses, server.URL)
	observer := newWiringObserver()
	ctx := execution.WithHTTPObserver(context.Background(), observer)
	result := manager.Execute(ctx, spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if result.Error != nil || result.StatusCode != http.StatusOK {
		t.Fatalf("gateway probe result = %+v", result)
	}
	if !result.ProbeAnswerPresent || result.ProbeResponseInvalid {
		t.Fatalf("probe evidence = %+v body=%s", result, result.Body)
	}
	if result.UpstreamProtocol != protocol.OpenAIResponses || result.UpstreamRequestID != "sse-gateway-1" {
		t.Fatalf("protocol/request id = %s/%s", result.UpstreamProtocol, result.UpstreamRequestID)
	}
	var normalized map[string]any
	if err := json.Unmarshal(result.Body, &normalized); err != nil {
		t.Fatalf("normalized body is not JSON: %v; body=%s", err, result.Body)
	}
	if normalized["object"] != "response" || normalized["status"] != "completed" || normalized["model"] != "probe-client" {
		t.Fatalf("normalized gateway response = %#v", normalized)
	}
	assertUsage(t, result.Usage, usage.Tokens{UncachedInput: 20, Output: 1})

	_, observedResponse, _, _, _, _ := observer.snapshot()
	if !bytes.Equal(observedResponse, []byte(fixture)) {
		t.Fatalf("raw capture was overwritten by normalization\n got %q\nwant %q", observedResponse, fixture)
	}
}
