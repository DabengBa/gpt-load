package bifrost

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/usage"
)

func TestOfficialOpenAIResponsesCreateNativePassthrough(t *testing.T) {
	t.Parallel()

	const responseBody = `{"id": "resp_1", "object":"response", "status":"completed", "model":"served-model", "output":[], "usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":4},"output_tokens":3,"total_tokens":13}, "vendor_response":{"precise":1.2300}}`
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/v1/responses" {
			t.Errorf("request target = %s %s", request.Method, request.URL.Path)
		}
		if request.URL.Query().Get("include") != "output_text" {
			t.Errorf("query = %q", request.URL.RawQuery)
		}
		if request.Header.Get("Authorization") != "Bearer "+testAPIKey {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		if request.Header.Get("X-Test-Header") != "forward-me" {
			t.Errorf("business header = %q", request.Header.Get("X-Test-Header"))
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Fatal(err)
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatal(err)
		}
		if payload["model"] != "upstream-model" || payload["stream"] != false {
			t.Errorf("request controls = %#v", payload)
		}
		for _, field := range []string{"provider", "fallbacks", "authorization", "api_key"} {
			if _, exists := payload[field]; exists {
				t.Errorf("control field %q reached upstream", field)
			}
		}
		if !bytes.Contains(body, []byte(`"precise":1.2300`)) {
			t.Errorf("unknown request field changed: %s", body)
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("X-Request-Id", "responses-create")
		_, _ = io.WriteString(writer, responseBody)
	}))
	defer server.Close()

	runtime := newOfficialOpenAITestRuntime(t, server.URL)
	spec := openAIResponsesSpec(execution.OperationResponsesCreate, http.MethodPost, "/v1/responses")
	spec.ClientModel = spec.UpstreamModel
	spec.Query.Set("include", "output_text")
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if result.StatusCode != http.StatusOK || result.UpstreamRequestID != "responses-create" || result.Model != "served-model" {
		t.Fatalf("result metadata = %+v", result)
	}
	if string(result.Body) != responseBody {
		t.Fatalf("response bytes changed:\n got: %s\nwant: %s", result.Body, responseBody)
	}
	assertUsage(t, result.Usage, usage.Tokens{UncachedInput: 6, CacheRead: 4, Output: 3})
}

func TestResponsesCreateMislabeledJSONContentType(t *testing.T) {
	const completed = `{"id":"resp_1","object":"response","status":"completed","model":"upstream-model","output":[]}`
	tests := []struct {
		name, body string
		status     int
		wantType   string
	}{
		{"completed", completed, 200, "application/json"},
		{"tool_call", `{"object":"response","status":"completed","output":[{"type":"shell_call","id":"shell_1","call_id":"call_1","action":{"commands":["pwd"]}}]}`, 200, "application/json"},
		{"malformed", `{"object":`, 200, "text/event-stream"},
		{"incomplete", `{"object":"response","status":"incomplete","output":[]}`, 200, "text/event-stream"},
		{"response_error", `{"object":"response","status":"completed","error":{"message":"failed"},"output":[]}`, 200, "text/event-stream"},
		{"invalid_message", `{"object":"response","status":"completed","output":[{"type":"message","content":null}]}`, 200, "text/event-stream"},
		{"invalid_shell_output", `{"object":"response","status":"completed","output":[{"type":"shell_call_output","id":"s1","call_id":"c1","output":[{"stdout":"","stderr":"","outcome":{"type":"exit"}}]}]}`, 200, "text/event-stream"},
		{"invalid_image_result", `{"object":"response","status":"completed","output":[{"type":"image_generation_call","id":"ig_1","status":"failed","result":{}}]}`, 200, "text/event-stream"},
		{"invalid_search_output", `{"object":"response","status":"completed","output":[{"type":"tool_search_output","id":"s","call_id":null,"status":"completed","execution":"server","tools":[null]}]}`, 200, "text/event-stream"},
		{"invalid_patch_output", `{"object":"response","status":"completed","output":[{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"completed","output":{}}]}`, 200, "text/event-stream"},
		{"invalid_function_output", `{"object":"response","status":"completed","output":[{"type":"function_call_output","id":"f","status":"completed","output":[null]}]}`, 200, "text/event-stream"},
		{"invalid_local_output", `{"object":"response","status":"completed","output":[{"type":"local_shell_call_output","id":"l","output":null}]}`, 200, "text/event-stream"},
		{"sse", responsesSSEReadyTerminal(), 200, "text/event-stream"},
		{"error_status", completed, 400, "text/event-stream"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(tt.status)
				_, _ = io.WriteString(w, tt.body)
			}))
			defer server.Close()
			runtime := newOfficialOpenAITestRuntime(t, server.URL)
			spec := openAIResponsesSpec(execution.OperationResponsesCreate, http.MethodPost, "/v1/responses")
			spec.ClientModel = spec.UpstreamModel
			result := runtime.Execute(context.Background(), spec)
			if result.Header.Get("Content-Type") != tt.wantType {
				t.Fatalf("content type = %q, want %q; error=%+v", result.Header.Get("Content-Type"), tt.wantType, result.Error)
			}
			if string(result.Body) != tt.body {
				t.Fatalf("body changed: %q", result.Body)
			}
		})
	}
}

func TestResponsesCreateMislabeledGzipJSON(t *testing.T) {
	const body = `{"object":"response","status":"completed","output":[],"usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21}}`
	for _, encoding := range []string{"gzip", ""} {
		t.Run("encoding="+encoding, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				if encoding != "" {
					w.Header().Set("Content-Encoding", encoding)
				}
				writer := gzip.NewWriter(w)
				_, _ = io.WriteString(writer, body)
				_ = writer.Close()
			}))
			defer server.Close()
			runtime := newOfficialOpenAITestRuntime(t, server.URL)
			spec := openAIResponsesSpec(execution.OperationResponsesCreate, http.MethodPost, "/v1/responses")
			spec.ClientModel = spec.UpstreamModel
			result := runtime.Execute(context.Background(), spec)
			if result.Error != nil || string(result.Body) != body || result.Header.Get("Content-Type") != "application/json" || result.Header.Get("Content-Encoding") != "" {
				t.Fatalf("result = %+v body=%q", result, result.Body)
			}
			assertUsage(t, result.Usage, usage.Tokens{UncachedInput: 20, Output: 1})
		})
	}
}

func TestResponsesInferredCompressionKeepsProtocolLimit(t *testing.T) {
	const completed = `{"object":"response","status":"completed","output":[]}`
	limit := execution.UnaryResponseBodyLimit(protocol.OpenAIResponses)
	var wire bytes.Buffer
	writer := gzip.NewWriter(&wire)
	_, _ = io.WriteString(writer, completed)
	padding := bytes.Repeat([]byte{' '}, 1<<20)
	for remaining := limit; remaining > 0; remaining -= int64(len(padding)) {
		if _, err := writer.Write(padding[:min(remaining, int64(len(padding)))]); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(wire.Bytes())
	}))
	defer server.Close()
	runtime := newRuntimeForTest(t, testRuntimeOptions{
		allowPrivateNetwork: true, openAIBaseURL: server.URL, maxUnaryResponseBodyBytes: limit + int64(len(completed)),
	})
	for _, operation := range []execution.Operation{execution.OperationResponsesCreate, execution.OperationProbe, execution.OperationModelTrace} {
		t.Run(string(operation), func(t *testing.T) {
			spec := openAIResponsesSpec(operation, http.MethodPost, "/v1/responses")
			if operation != execution.OperationResponsesCreate {
				spec = utilitySpec(channel.OpenAI, protocol.OpenAIResponses, operation, "", "", nil)
				spec.UpstreamModel = "upstream-model"
				spec.PromptText, spec.ProbeMaxOutputTokens = "test", testProbeOutputTokens
			}
			spec.ClientModel = spec.UpstreamModel
			result := runtime.Execute(context.Background(), spec)
			if !result.ResponseStarted || result.Error == nil || len(result.Body) != 0 {
				t.Fatalf("oversized compressed Responses: status=%d, error=%v, body length=%d", result.StatusCode, result.Error, len(result.Body))
			}
		})
	}
}

func TestOfficialOpenAIResponsesCreateNativeStream(t *testing.T) {
	t.Parallel()

	const rawStream = "event: response.output_text.delta\n" +
		"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\",\"vendor\":{\"precise\":1.2300}}\n\n" +
		"event: response.completed\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream\",\"model\":\"served-model\",\"usage\":{\"input_tokens\":8,\"input_tokens_details\":{\"cached_tokens\":3},\"output_tokens\":2,\"total_tokens\":10}}}\n\n"
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Fatal(err)
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatal(err)
		}
		if payload["model"] != "upstream-model" || payload["stream"] != true {
			t.Errorf("request controls = %#v", payload)
		}
		if _, exists := payload["stream_options"]; exists {
			t.Errorf("unexpected stream_options = %#v", payload["stream_options"])
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("X-Request-Id", "responses-stream")
		writer.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(writer, rawStream)
		writer.(http.Flusher).Flush()
	}))
	defer server.Close()

	runtime := newOfficialOpenAITestRuntime(t, server.URL)
	spec := openAIResponsesSpec(execution.OperationResponsesCreate, http.MethodPost, "/v1/responses")
	spec.ClientModel = spec.UpstreamModel
	var events []execution.StreamEvent
	result := runtime.ExecuteStream(
		context.Background(),
		spec,
		func(event execution.StreamEvent) error {
			events = append(events, event.Clone())
			return nil
		},
	)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if len(events) < 3 || events[0].Kind != execution.StreamEventReady || events[0].UpstreamRequestID != "responses-stream" {
		t.Fatalf("events = %+v", events)
	}
	var data bytes.Buffer
	var sawUsage bool
	for _, event := range events[1:] {
		if event.Kind == execution.StreamEventData {
			data.Write(event.Data)
		}
		if event.Kind == execution.StreamEventUsage {
			sawUsage = true
			assertUsage(t, event.Usage, usage.Tokens{UncachedInput: 5, CacheRead: 3, Output: 2})
		}
	}
	if data.String() != rawStream || !sawUsage {
		t.Fatalf("stream data/usage = %q/%t", data.String(), sawUsage)
	}
	assertUsage(t, result.Usage, usage.Tokens{UncachedInput: 5, CacheRead: 3, Output: 2})
}

func TestOfficialOpenAIResponsesLifecyclePreservesTargetAndStatus(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		operation execution.Operation
		method    string
		path      string
		status    int
		body      string
	}{
		{name: "retrieve", operation: execution.OperationResponsesRetrieve, method: http.MethodGet, path: "/v1/responses/resp_123", status: http.StatusOK, body: `{"id":"resp_123","object":"response","status":"completed"}`},
		{name: "delete", operation: execution.OperationResponsesDelete, method: http.MethodDelete, path: "/v1/responses/resp_123", status: http.StatusOK, body: `{"id":"resp_123","object":"response.deleted","deleted":true}`},
		{name: "cancel", operation: execution.OperationResponsesCancel, method: http.MethodPost, path: "/v1/responses/resp_123/cancel", status: http.StatusAccepted, body: `{"id":"resp_123","object":"response","status":"cancelling"}`},
		{name: "input items", operation: execution.OperationResponsesInputItems, method: http.MethodGet, path: "/v1/responses/resp_123/input_items", status: http.StatusPartialContent, body: `{"object":"list","data":[]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.Method != test.method || request.URL.Path != test.path {
					t.Errorf("request target = %s %s", request.Method, request.URL.Path)
				}
				if request.URL.Query().Get("after") != "item_1" {
					t.Errorf("query = %q", request.URL.RawQuery)
				}
				if request.Header.Get("Authorization") != "Bearer "+testAPIKey {
					t.Errorf("authorization = %q", request.Header.Get("Authorization"))
				}
				writer.Header().Set("Content-Type", "application/json")
				writer.Header().Set("X-Request-Id", "lifecycle-"+test.name)
				writer.WriteHeader(test.status)
				_, _ = io.WriteString(writer, test.body)
			}))
			defer server.Close()

			runtime := newOfficialOpenAITestRuntime(t, server.URL)
			spec := openAIResponsesSpec(test.operation, test.method, test.path)
			spec.ClientModel = ""
			spec.UpstreamModel = ""
			spec.Body = nil
			spec.Query.Set("after", "item_1")
			result := runtime.Execute(context.Background(), spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if result.StatusCode != test.status || string(result.Body) != test.body || result.Error != nil {
				t.Fatalf("result = %+v, body=%s", result, result.Body)
			}
		})
	}
}

func TestOfficialOpenAIResponsesUtilityOperationsPreserveNativeSemantics(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		operation execution.Operation
		path      string
		body      []byte
		response  string
	}{
		{
			name:      "compact",
			operation: execution.OperationResponsesCompact,
			path:      "/v1/responses/compact",
			body:      []byte(`{"model":"client-model","input":[{"role":"user","content":"hello"}],"provider":"injected","fallbacks":["other/model"],"vendor_extension":{"precise":1.2300}}`),
			response:  `{"id":"cmp_123","object":"response.compaction","output":[],"vendor":{"precise":1.2300}}`,
		},
		{
			name:      "input tokens",
			operation: execution.OperationResponsesInputTokens,
			path:      "/v1/responses/input_tokens",
			body:      []byte(`{"model":"client-model","input":"hello","provider":"injected","api_key":"injected","vendor_extension":{"precise":1.2300}}`),
			response:  `{"object":"response.input_tokens","input_tokens":7,"vendor":{"precise":1.2300}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				calls.Add(1)
				if request.Method != http.MethodPost || request.URL.Path != test.path || request.URL.Query().Get("trace") != "kept" {
					t.Errorf("target = %s %s?%s", request.Method, request.URL.Path, request.URL.RawQuery)
				}
				if request.Header.Get("Authorization") != "Bearer "+testAPIKey || request.Header.Get("Api-Key") != "" {
					t.Errorf("credential headers = authorization:%q api-key:%q", request.Header.Get("Authorization"), request.Header.Get("Api-Key"))
				}
				body, _ := io.ReadAll(request.Body)
				var payload map[string]any
				if err := json.Unmarshal(body, &payload); err != nil {
					t.Fatalf("decode request body: %v", err)
				}
				if payload["model"] != "upstream-model" || payload["provider"] != nil || payload["fallbacks"] != nil || payload["api_key"] != nil {
					t.Errorf("sanitized request = %#v", payload)
				}
				if _, ok := payload["vendor_extension"]; !ok {
					t.Errorf("vendor extension was dropped: %#v", payload)
				}
				writer.Header().Set("Content-Type", "application/json")
				writer.Header().Set("X-Request-Id", "utility-"+test.name)
				_, _ = io.WriteString(writer, test.response)
			}))
			defer server.Close()

			runtime := newOfficialOpenAITestRuntime(t, server.URL)
			spec := openAIResponsesSpec(test.operation, http.MethodPost, test.path)
			spec.Body = test.body
			spec.Query.Set("trace", "kept")
			result := runtime.Execute(context.Background(), spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if calls.Load() != 1 || result.StatusCode != http.StatusOK || result.Error != nil || string(result.Body) != test.response {
				t.Fatalf("calls/result = %d/%+v body=%s", calls.Load(), result, result.Body)
			}
		})
	}
}

func newOfficialOpenAITestRuntime(t *testing.T, baseURL string) *testRuntime {
	t.Helper()
	return newRuntimeForTest(t, testRuntimeOptions{
		allowPrivateNetwork: true,
		openAIBaseURL:       baseURL,
	})
}

func openAIResponsesSpec(operation execution.Operation, method, path string) execution.AttemptSpec {
	return freezeTestAttempt(execution.NewAttemptSpec(execution.AttemptSpec{
		RequestID:      "responses-request",
		AttemptID:      "responses-attempt",
		Sequence:       1,
		ChannelID:      string(channel.OpenAI),
		ClientProtocol: protocol.OpenAIResponses,
		Operation:      operation,
		ClientModel:    "client-model",
		UpstreamModel:  "upstream-model",
		Method:         method,
		Path:           path,
		Query:          make(map[string][]string),
		Header: http.Header{
			"Authorization": []string{"Bearer client-injected"},
			"Api-Key":       []string{"client-injected"},
			"X-Test-Header": []string{"forward-me"},
		},
		Body:         []byte(`{"model":"client-model","stream":true,"provider":"injected","fallbacks":["other/model"],"authorization":"body-injected","api_key":"body-injected","input":"hello","vendor_extension":{"precise":1.2300}}`),
		TargetConfig: json.RawMessage(`{}`),
		Credential:   execution.NewCredentialSnapshot(7, 3, 2, []byte(`{"api_key":"`+testAPIKey+`"}`)),
	}))
}
