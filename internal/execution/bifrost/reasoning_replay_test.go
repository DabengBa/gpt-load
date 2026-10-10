package bifrost

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
)

func TestConvertedReasoningReplaysToNativeResponses(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "unary", true: "stream"}[stream], func(t *testing.T) {
			var nativeBody []byte
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/chat/v1/chat/completions":
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, "data: "+`{"id":"c1","object":"chat.completion.chunk","model":"served","choices":[{"index":0,"delta":{"reasoning_content":"plan"},"finish_reason":null}]}`+"\n\n"+
							"data: "+`{"id":"c1","object":"chat.completion.chunk","model":"served","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}`+"\n\n"+
							"data: "+`{"id":"c1","object":"chat.completion.chunk","model":"served","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\ndata: [DONE]\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"served","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"plan","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
					}
				case "/native/v1/responses":
					nativeBody, _ = io.ReadAll(r.Body)
					var request struct {
						Input []map[string]json.RawMessage `json:"input"`
					}
					if err := json.Unmarshal(nativeBody, &request); err != nil {
						t.Error(err)
					}
					valid, sawReasoning, sawCall, sawResult := true, false, false, false
					for _, item := range request.Input {
						switch string(item["type"]) {
						case `"reasoning"`:
							sawReasoning = true
							_, role := item["role"]
							valid = valid && !role && string(item["summary"]) == `[]`
							var content []struct{ Type, Text string }
							if err := json.Unmarshal(item["content"], &content); err != nil || len(content) != 1 || content[0].Type != "reasoning_text" || content[0].Text != "plan" {
								valid = false
							}
						case `"function_call"`:
							sawCall = string(item["call_id"]) == `"call_1"`
						case `"function_call_output"`:
							sawResult = string(item["call_id"]) == `"call_1"`
						}
					}
					if !valid || !sawReasoning || !sawCall || !sawResult {
						w.WriteHeader(http.StatusBadRequest)
						_, _ = io.WriteString(w, `{"error":{"type":"invalid_request_error","message":"invalid reasoning or tool pairing"}}`)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"r1","object":"response","status":"completed","model":"served","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
				default:
					t.Errorf("unexpected path: %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			spec := openAIResponsesSpec(execution.OperationResponsesCreate, http.MethodPost, "/v1/responses")
			spec.ChannelID = string(channel.OpenAICompatible)
			spec.TargetConfig, _ = json.Marshal(map[string]string{"base_url": server.URL + "/chat/v1"})
			spec.Body, _ = json.Marshal(map[string]any{"model": "worker", "input": "hello", "stream": stream, "store": false})
			spec = freezeTestAttempt(spec)
			var output []json.RawMessage
			if stream {
				result := runtime.ExecuteStream(t.Context(), spec, func(event execution.StreamEvent) error {
					for _, line := range strings.Split(string(event.Data), "\n") {
						if !strings.HasPrefix(line, "data:") {
							continue
						}
						var frame struct {
							Type     string          `json:"type"`
							Item     json.RawMessage `json:"item"`
							Response struct {
								Output []json.RawMessage `json:"output"`
							} `json:"response"`
						}
						if err := json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &frame); err != nil {
							continue
						}
						if frame.Type == "response.output_item.done" {
							output = append(output, frame.Item)
						}
					}
					return nil
				})
				if result.Error != nil {
					t.Fatalf("converted stream: %+v", result)
				}
			} else {
				result := runtime.Execute(t.Context(), spec)
				if result.Error != nil {
					t.Fatalf("converted unary: %+v", result)
				}
				var response struct {
					Output []json.RawMessage `json:"output"`
				}
				if err := json.Unmarshal(result.Body, &response); err != nil {
					t.Fatal(err)
				}
				output = response.Output
			}
			if len(output) != 2 {
				t.Fatalf("reasoning and tool output expected, got %s", output)
			}
			output = append(output, json.RawMessage(`{"type":"function_call_output","call_id":"call_1","output":"result"}`))
			// Keep the captured objects byte-for-byte; native passthrough must not
			// make this succeed merely by dropping the invalid reasoning item.
			spec.ChannelID = string(channel.OpenAI)
			spec.TargetConfig, _ = json.Marshal(map[string]string{"base_url": server.URL + "/native"})
			spec.Body, _ = json.Marshal(map[string]any{"model": "worker", "input": output, "stream": false, "store": false})
			spec = freezeTestAttempt(spec)
			result := runtime.Execute(t.Context(), spec)
			if result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("native replay rejected: result=%+v body=%s sent=%s", result, result.Body, nativeBody)
			}
		})
	}
}
