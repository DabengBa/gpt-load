package bifrost

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/tidwall/gjson"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
)

// toolOutputRequestBody builds a Responses request whose only interesting part
// is the tool result at input[2].
func toolOutputRequestBody(t *testing.T, stream bool, output any) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model": "client-model",
		"input": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"type": "function_call", "call_id": "call_1", "name": "inspect", "arguments": "{}"},
			map[string]any{"type": "function_call_output", "call_id": "call_1", "output": output},
		},
		"max_output_tokens": 32,
		"stream":            stream,
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// strictToolOutputUpstream rejects any tool message whose content is not a JSON
// string, mirroring Chat Completions typing of tool content.
func strictToolOutputUpstream(t *testing.T, calls *atomic.Int64, captured *[]byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Error(err)
			return
		}
		*captured = body
		content := gjson.GetBytes(body, `messages.#(role=="tool").content`)
		if !content.Exists() {
			t.Errorf("upstream request has no tool message: %s", body)
		} else if content.Type != gjson.String {
			t.Errorf("tool content is not a string on the Chat wire: %s", content.Raw)
		}
		writer.Header().Set("Content-Type", "application/json")
		if request.URL.Query().Get("stream") == "true" {
			writer.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(writer, openAIChatFinalStream)
			return
		}
		_, _ = io.WriteString(writer, openAIChatFinalResponse)
	}))
}

func aheadToolOutputSpec(t *testing.T, runtime *testRuntime, server *httptest.Server, stream bool, body []byte) execution.AttemptSpec {
	t.Helper()
	spec := openAIResponsesSpec(execution.OperationResponsesCreate, http.MethodPost, "/v1/responses")
	spec.ChannelID = string(channel.OpenAICompatible)
	spec.TargetConfig = json.RawMessage(`{"base_url":"` + server.URL + `/v1"}`)
	if stream {
		if spec.Query == nil {
			spec.Query = make(map[string][]string)
		}
		spec.Query.Set("stream", "true")
	}
	spec = freezeTestAttempt(spec)
	spec.Body = body
	_ = runtime
	return spec
}

func executeAheadToolOutput(t *testing.T, stream bool, output any) ([]byte, int64) {
	t.Helper()
	var calls atomic.Int64
	var captured []byte
	server := strictToolOutputUpstream(t, &calls, &captured)
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := aheadToolOutputSpec(t, runtime, server, stream, toolOutputRequestBody(t, stream, output))
	if stream {
		result := runtime.ExecuteStream(context.Background(), spec, func(execution.StreamEvent) error { return nil })
		if err := result.Validate(); err != nil || result.Error != nil {
			t.Fatalf("stream result = %+v err=%v", result, result.Error)
		}
	} else {
		result := runtime.Execute(context.Background(), spec)
		if err := result.Validate(); err != nil || result.Error != nil {
			t.Fatalf("result = %+v err=%v body=%s", result, result.Error, result.Body)
		}
	}
	return captured, calls.Load()
}

func assertToolWireText(t *testing.T, upstream []byte, want string) {
	t.Helper()
	if got := gjson.GetBytes(upstream, `messages.#(role=="tool").content`).String(); got != want {
		t.Fatalf("tool content = %q, want %q (wire: %s)", got, want, upstream)
	}
	assistant := gjson.GetBytes(upstream, `messages.#(role=="assistant")`)
	if assistant.Get("tool_calls.0.id").String() != "call_1" ||
		assistant.Get("tool_calls.0.function.name").String() != "inspect" {
		t.Fatalf("assistant tool call id/name lost: %s", upstream)
	}
	if got := gjson.GetBytes(upstream, `messages.#(role=="tool").tool_call_id`).String(); got != "call_1" {
		t.Fatalf("tool_call_id = %q, want call_1", got)
	}
}

func TestAheadToolOutputTextBlocksFoldToString(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			upstream, calls := executeAheadToolOutput(t, stream, []any{
				map[string]any{"type": "input_text", "text": "first"},
				map[string]any{"type": "output_text", "text": "second"},
			})
			if calls != 1 {
				t.Fatalf("upstream calls = %d, want 1", calls)
			}
			assertToolWireText(t, upstream, "first\nsecond")
		})
	}
}

func TestAheadToolOutputEmptyArrayBecomesEmptyString(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			upstream, calls := executeAheadToolOutput(t, stream, []any{})
			if calls != 1 {
				t.Fatalf("upstream calls = %d, want 1", calls)
			}
			assertToolWireText(t, upstream, "")
		})
	}
}

func TestAheadToolOutputStringOutputUnchanged(t *testing.T) {
	t.Parallel()
	upstream, calls := executeAheadToolOutput(t, false, "plain string result")
	if calls != 1 {
		t.Fatalf("upstream calls = %d, want 1", calls)
	}
	assertToolWireText(t, upstream, "plain string result")
}

// executeAheadToolOutputRejection runs one lossy tool output and returns the
// dispatch state, error code and upstream call count. The original request body
// must survive untouched for a not-sent rejection.
func executeAheadToolOutputRejection(t *testing.T, stream bool, output any) (execution.DispatchState, string, int64) {
	t.Helper()
	var calls atomic.Int64
	var captured []byte
	server := strictToolOutputUpstream(t, &calls, &captured)
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	body := toolOutputRequestBody(t, stream, output)
	spec := aheadToolOutputSpec(t, runtime, server, stream, body)
	if stream {
		result := runtime.ExecuteStream(context.Background(), spec, func(execution.StreamEvent) error { return nil })
		if result.DispatchState != execution.DispatchNotSent || result.Error == nil {
			t.Fatalf("stream dispatch = %s error=%+v", result.DispatchState, result.Error)
		}
		return result.DispatchState, result.Error.Code, calls.Load()
	}
	result := runtime.Execute(context.Background(), spec)
	if result.DispatchState != execution.DispatchNotSent || result.Error == nil {
		t.Fatalf("dispatch = %s error=%+v", result.DispatchState, result.Error)
	}
	return result.DispatchState, result.Error.Code, calls.Load()
}

func TestAheadToolOutputLossyContentRejectsBeforeDispatch(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		output []any
	}{
		{"image block", []any{
			map[string]any{"type": "input_text", "text": "screenshot"},
			map[string]any{"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"},
		}},
		{"file block", []any{map[string]any{"type": "input_file", "file_id": "f_1"}}},
		{"nested audio block", []any{
			map[string]any{"type": "input_audio", "input_audio": map[string]any{"format": "wav", "data": "AA=="}},
		}},
		{"summary block", []any{map[string]any{"type": "summary_text", "text": "summary"}}},
		{"reasoning block", []any{map[string]any{"type": "reasoning_text", "text": "thought"}}},
		{"unknown block", []any{map[string]any{"type": "compaction", "text": "?"}}},
		{"text block without text", []any{map[string]any{"type": "input_text"}}},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", test.name, stream), func(t *testing.T) {
				dispatch, code, calls := executeAheadToolOutputRejection(t, stream, test.output)
				if dispatch != execution.DispatchNotSent || code != execution.ErrorCodeCriticalSemanticLoss || calls != 0 {
					t.Fatalf("lossy tool output dispatched: dispatch=%s code=%s calls=%d", dispatch, code, calls)
				}
			})
		}
	}
}

func TestAheadToolOutputRejectionLeavesRequestBodyUnchanged(t *testing.T) {
	t.Parallel()
	var calls atomic.Int64
	var captured []byte
	server := strictToolOutputUpstream(t, &calls, &captured)
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	body := toolOutputRequestBody(t, false, []any{
		map[string]any{"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"},
	})
	spec := aheadToolOutputSpec(t, runtime, server, false, body)
	original := bytes.Clone(spec.Body)
	result := runtime.Execute(context.Background(), spec)
	if result.DispatchState != execution.DispatchNotSent || result.Error == nil ||
		result.Error.Code != execution.ErrorCodeCriticalSemanticLoss {
		t.Fatalf("rejection result = %+v", result)
	}
	if !bytes.Equal(spec.Body, original) {
		t.Fatalf("spec body was mutated by rejection: %s", spec.Body)
	}
	if calls.Load() != 0 || captured != nil {
		t.Fatalf("request reached upstream: calls=%d captured=%s", calls.Load(), captured)
	}
}
