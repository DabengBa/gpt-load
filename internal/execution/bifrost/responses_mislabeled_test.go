package bifrost

import (
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestCompletedResponsesJSONAdditionalOutputValidation(t *testing.T) {
	tests := []struct {
		name, item string
		valid      bool
	}{
		{"shell_exit", `{"type":"shell_call_output","id":"sho_1","call_id":"call_1","output":[{"stdout":"ok","stderr":"","outcome":{"type":"exit","exit_code":0}}]}`, true},
		{"shell_timeout", `{"type":"shell_call_output","id":"sho_1","call_id":"call_1","output":[{"stdout":"","stderr":"","outcome":{"type":"timeout"}}]}`, true},
		{"shell_no_call_id", `{"type":"shell_call_output","id":"sho_1","output":[]}`, false},
		{"shell_null_output", `{"type":"shell_call_output","id":"sho_1","call_id":"c","output":null}`, false},
		{"shell_null_chunk", `{"type":"shell_call_output","id":"sho_1","call_id":"c","output":[null]}`, false},
		{"shell_missing_stdout", `{"type":"shell_call_output","id":"sho_1","call_id":"c","output":[{"stderr":"","outcome":{"type":"timeout"}}]}`, false},
		{"shell_invalid_stderr", `{"type":"shell_call_output","id":"sho_1","call_id":"c","output":[{"stdout":"","stderr":null,"outcome":{"type":"timeout"}}]}`, false},
		{"shell_invalid_outcome", `{"type":"shell_call_output","id":"sho_1","call_id":"c","output":[{"stdout":"","stderr":"","outcome":{"type":"other"}}]}`, false},
		{"shell_missing_exit_code", `{"type":"shell_call_output","id":"sho_1","call_id":"c","output":[{"stdout":"","stderr":"","outcome":{"type":"exit"}}]}`, false},
		{"shell_invalid_exit_code", `{"type":"shell_call_output","id":"sho_1","call_id":"c","output":[{"stdout":"","stderr":"","outcome":{"type":"exit","exit_code":null}}]}`, false},
		{"shell_fractional_exit_code", `{"type":"shell_call_output","id":"sho_1","call_id":"c","output":[{"stdout":"","stderr":"","outcome":{"type":"exit","exit_code":1.5}}]}`, false},
		{"image_null_failed", `{"type":"image_generation_call","id":"ig_1","status":"failed","result":null}`, true},
		{"image_null_completed", `{"type":"image_generation_call","id":"ig_1","status":"completed","result":null}`, true},
		{"image_string", `{"type":"image_generation_call","id":"ig_1","status":"completed","result":"base64"}`, true},
		{"image_missing_result", `{"type":"image_generation_call","id":"ig_1","status":"failed"}`, false},
		{"image_invalid_result", `{"type":"image_generation_call","id":"ig_1","status":"failed","result":{}}`, false},
		{"image_missing_status", `{"type":"image_generation_call","id":"ig_1","result":null}`, false},
		{"search_call_hosted", `{"type":"tool_search_call","id":"s","call_id":null,"status":"completed","execution":"server","arguments":{}}`, true},
		{"search_call_null_arguments", `{"type":"tool_search_call","id":"s","call_id":"c","status":"completed","execution":"client","arguments":null}`, true},
		{"search_missing_call_id", `{"type":"tool_search_call","id":"s","status":"completed","execution":"server","arguments":{}}`, false},
		{"search_invalid_call_id", `{"type":"tool_search_call","id":"s","call_id":{},"status":"completed","execution":"server","arguments":{}}`, false},
		{"search_missing_arguments", `{"type":"tool_search_call","id":"s","call_id":null,"status":"completed","execution":"server"}`, false},
		{"search_invalid_execution", `{"type":"tool_search_call","id":"s","call_id":null,"status":"completed","execution":"other","arguments":{}}`, false},
		{"search_output_hosted", `{"type":"tool_search_output","id":"s","call_id":null,"status":"completed","execution":"server","tools":[]}`, true},
		{"search_output_tool", `{"type":"tool_search_output","id":"s","call_id":"c","status":"completed","execution":"client","tools":[{"type":"function","name":"f","parameters":null,"strict":null}]}`, true},
		{"search_output_null_tools", `{"type":"tool_search_output","id":"s","call_id":null,"status":"completed","execution":"server","tools":null}`, false},
		{"search_output_invalid_tool", `{"type":"tool_search_output","id":"s","call_id":null,"status":"completed","execution":"server","tools":[null]}`, false},
		{"search_output_untyped_tool", `{"type":"tool_search_output","id":"s","call_id":null,"status":"completed","execution":"server","tools":[{}]}`, false},
		{"patch_output", `{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"completed","output":"ok"}`, true},
		{"patch_output_null", `{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"failed","output":null}`, true},
		{"patch_output_absent", `{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"completed"}`, true},
		{"patch_missing_call_id", `{"type":"apply_patch_call_output","id":"p","status":"completed"}`, false},
		{"patch_invalid_status", `{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"other"}`, false},
		{"patch_invalid_output", `{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"completed","output":{}}`, false},
		{"function_output", `{"type":"function_call_output","id":"f","status":"completed","output":"ok"}`, true},
		{"function_output_blocks", `{"type":"function_call_output","id":"f","call_id":"c","status":"completed","output":[{"type":"input_text","text":"ok"},{"type":"input_image","detail":"auto","image_url":null},{"type":"input_file","file_id":"f"}]}`, true},
		{"function_null_output", `{"type":"function_call_output","id":"f","status":"completed","output":null}`, false},
		{"function_null_call_id", `{"type":"function_call_output","id":"f","status":"completed","call_id":null,"output":"ok"}`, false},
		{"function_missing_status", `{"type":"function_call_output","id":"f","output":"ok"}`, false},
		{"function_bad_block", `{"type":"function_call_output","id":"f","status":"completed","output":[{"type":"input_text","text":null}]}`, false},
		{"function_bad_block_type", `{"type":"function_call_output","id":"f","status":"completed","output":[{"type":"output_text","text":"ok"}]}`, false},
		{"function_bad_image", `{"type":"function_call_output","id":"f","status":"completed","output":[{"type":"input_image","detail":"auto","image_url":7}]}`, false},
		{"local_output", `{"type":"local_shell_call_output","id":"l","output":"ok"}`, true},
		{"local_output_null_status", `{"type":"local_shell_call_output","id":"l","status":null,"output":"ok"}`, true},
		{"local_missing_id", `{"type":"local_shell_call_output","output":"ok"}`, false},
		{"local_invalid_output", `{"type":"local_shell_call_output","id":"l","output":{}}`, false},
		{"local_invalid_status", `{"type":"local_shell_call_output","id":"l","status":7,"output":"ok"}`, false},
		{"computer_batch", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[{"type":"screenshot"},{"type":"click","button":"left","x":1,"y":2},{"type":"type","text":"hi"}]}`, true},
		{"computer_batch_all_actions", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[{"id":"s","code":null}],"actions":[{"type":"double_click","x":1,"y":2,"keys":null},{"type":"drag","path":[{"x":0,"y":1}]},{"type":"keypress","keys":["ENTER"]},{"type":"move","x":1,"y":2},{"type":"scroll","x":1,"y":2,"scroll_x":3,"scroll_y":4},{"type":"wait"}]}`, true},
		{"computer_batch_null", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":null}`, false},
		{"computer_batch_invalid_element", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[null]}`, false},
		{"computer_batch_invalid_type", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[{"type":"other"}]}`, false},
		{"computer_batch_invalid_click", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[{"type":"click","button":"left","x":null,"y":2}]}`, false},
		{"computer_batch_invalid_drag", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[{"type":"drag","path":[{}]}]}`, false},
		{"computer_batch_invalid_keys", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[{"type":"keypress","keys":[1]}]}`, false},
		{"computer_batch_double_click_missing_keys", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[{"type":"double_click","x":1,"y":2}]}`, false},
		{"computer_batch_invalid_even_with_single", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"action":{"type":"screenshot"},"actions":[null]}`, false},
		{"computer_batch_missing_call_id", `{"type":"computer_call","id":"c","status":"completed","pending_safety_checks":[],"actions":[]}`, false},
		{"computer_batch_invalid_status", `{"type":"computer_call","id":"c","call_id":"call","status":"unknown","pending_safety_checks":[],"actions":[]}`, false},
		{"computer_batch_invalid_safety", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[null],"actions":[]}`, false},
		{"computer_output", `{"type":"computer_call_output","id":"c","call_id":"call","status":"completed","output":{"type":"computer_screenshot","file_id":"f"}}`, true},
		{"computer_output_invalid", `{"type":"computer_call_output","id":"c","call_id":"call","status":"completed","output":{"type":"other"}}`, false},
		{"computer_output_invalid_url", `{"type":"computer_call_output","id":"c","call_id":"call","status":"completed","output":{"type":"computer_screenshot","image_url":null}}`, false},
		{"computer_output_bad_safety", `{"type":"computer_call_output","id":"c","call_id":"call","status":"completed","output":{"type":"computer_screenshot"},"acknowledged_safety_checks":[{}]}`, false},
		{"custom_output", `{"type":"custom_tool_call_output","id":"c","call_id":"call","status":"completed","output":[{"type":"input_text","text":"ok"}]}`, true},
		{"custom_output_invalid", `{"type":"custom_tool_call_output","id":"c","call_id":"call","status":"completed","output":[null]}`, false},
		{"approval", `{"type":"mcp_approval_response","id":"a","approval_request_id":"r","approve":false,"reason":null}`, true},
		{"approval_invalid", `{"type":"mcp_approval_response","id":"a","approval_request_id":"r","approve":null}`, false},
		{"program", `{"type":"program","id":"p","call_id":"c","code":"x","fingerprint":"fp"}`, true},
		{"program_missing_fingerprint", `{"type":"program","id":"p","call_id":"c","code":"x"}`, false},
		{"program_output", `{"type":"program_output","id":"p","call_id":"c","status":"incomplete","result":"ok"}`, true},
		{"program_output_bad_status", `{"type":"program_output","id":"p","call_id":"c","status":"failed","result":"ok"}`, false},
		{"additional_tools", `{"type":"additional_tools","id":"a","role":"assistant","tools":[]}`, true},
		{"additional_tools_invalid_role", `{"type":"additional_tools","id":"a","role":"other","tools":[]}`, false},
		{"additional_tools_invalid", `{"type":"additional_tools","id":"a","role":"developer","tools":[null]}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"object":"response","status":"completed","output":[` + tt.item + `]}`
			if got := validCompletedResponsesJSON([]byte(body)); got != tt.valid {
				t.Fatalf("valid = %v, want %v", got, tt.valid)
			}
			incomplete := `{"object":"response","status":"incomplete","output":[` + tt.item + `]}`
			if validCompletedResponsesJSON([]byte(incomplete)) {
				t.Fatal("incomplete response accepted as completed")
			}
		})
	}
}

func TestResponsesCompletedAdditionalOutputsMislabeled(t *testing.T) {
	fixtures := map[string]string{
		"shell_output":     `{"type":"shell_call_output","id":"sho_1","call_id":"c1","output":[{"stdout":"ok","stderr":"","outcome":{"type":"exit","exit_code":0}}]}`,
		"image_null":       `{"type":"image_generation_call","id":"ig_1","status":"failed","result":null}`,
		"search_call":      `{"type":"tool_search_call","id":"s","call_id":null,"status":"completed","execution":"server","arguments":{}}`,
		"search_output":    `{"type":"tool_search_output","id":"s","call_id":null,"status":"completed","execution":"server","tools":[]}`,
		"patch_output":     `{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"completed"}`,
		"function_output":  `{"type":"function_call_output","id":"f","status":"completed","output":[{"type":"input_text","text":"ok"}]}`,
		"local_output":     `{"type":"local_shell_call_output","id":"l","output":"ok"}`,
		"computer_batch":   `{"type":"computer_call","id":"cu","call_id":"c","status":"completed","pending_safety_checks":[],"actions":[{"type":"screenshot"}]}`,
		"computer_output":  `{"type":"computer_call_output","id":"cu","call_id":"c","status":"completed","output":{"type":"computer_screenshot","file_id":"f"}}`,
		"custom_output":    `{"type":"custom_tool_call_output","id":"ct","call_id":"c","status":"completed","output":"ok"}`,
		"approval":         `{"type":"mcp_approval_response","id":"a","approval_request_id":"r","approve":true}`,
		"program":          `{"type":"program","id":"p","call_id":"c","code":"x","fingerprint":"fp"}`,
		"program_output":   `{"type":"program_output","id":"p","call_id":"c","status":"completed","result":"ok"}`,
		"additional_tools": `{"type":"additional_tools","id":"a","role":"assistant","tools":[]}`,
	}
	for name, item := range fixtures {
		body := `{"object":"response","status":"completed","model":"upstream-model","output":[` + item + `,{"type":"message","content":[{"type":"output_text","text":"4"}]}]}`
		for _, operation := range []execution.Operation{execution.OperationResponsesCreate, execution.OperationProbe, execution.OperationModelTrace} {
			for _, compressed := range []bool{false, true} {
				t.Run(name+"/"+string(operation)+map[bool]string{false: "/plain", true: "/gzip"}[compressed], func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
						w.Header().Set("Content-Type", "text/event-stream")
						if compressed {
							w.Header().Set("Content-Encoding", "gzip")
							writer := gzip.NewWriter(w)
							_, _ = io.WriteString(writer, body)
							_ = writer.Close()
						} else {
							_, _ = io.WriteString(w, body)
						}
					}))
					defer server.Close()
					runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
					var spec execution.AttemptSpec
					if operation == execution.OperationResponsesCreate {
						spec = openAIResponsesSpec(operation, http.MethodPost, "/v1/responses")
					} else {
						spec = utilitySpec(channel.OpenAI, protocol.OpenAIResponses, operation, "", "", nil)
						spec.PromptText, spec.ProbeMaxOutputTokens = "test", testProbeOutputTokens
					}
					spec.ClientModel, spec.UpstreamModel = "upstream-model", "upstream-model"
					result := runtime.Execute(t.Context(), spec)
					if result.Error != nil || result.StatusCode != http.StatusOK || result.Header.Get("Content-Type") != "application/json" || result.Header.Get("Content-Encoding") != "" || string(result.Body) != body {
						t.Fatalf("result = %+v body=%q", result, result.Body)
					}
					if operation == execution.OperationModelTrace && result.GeneratedText != "4" {
						t.Fatalf("generated text = %q", result.GeneratedText)
					}
				})
			}
		}
	}
}

func TestGatewayCompletedResponseToolValidation(t *testing.T) {
	tests := []struct {
		name, item string
		valid      bool
	}{
		{"shell", `{"type":"shell_call","id":"s1","call_id":"c1","action":{"commands":["pwd"]}}`, true},
		{"shell_empty_action", `{"type":"shell_call","id":"s1","call_id":"c1","action":{}}`, false},
		{"shell_invalid_commands", `{"type":"shell_call","id":"s1","call_id":"c1","action":{"commands":[1]}}`, false},
		{"patch", `{"type":"apply_patch_call","id":"p1","call_id":"c1","operation":{"type":"create_file","path":"a","diff":"+a"}}`, true},
		{"patch_invalid_path", `{"type":"apply_patch_call","id":"p1","call_id":"c1","operation":{"type":"delete_file","path":7}}`, false},
		{"patch_missing_diff", `{"type":"apply_patch_call","id":"p1","call_id":"c1","operation":{"type":"create_file","path":"a"}}`, false},
		{"search_call_hosted", `{"type":"tool_search_call","id":"s","call_id":null,"status":"completed","execution":"server","arguments":{}}`, true},
		{"search_output_hosted", `{"type":"tool_search_output","id":"s","call_id":null,"status":"completed","execution":"server","tools":[]}`, true},
		{"patch_output", `{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"completed"}`, true},
		{"function_output", `{"type":"function_call_output","id":"f","status":"completed","output":"ok"}`, true},
		{"local_output", `{"type":"local_shell_call_output","id":"l","output":"ok"}`, true},
		{"search_output_invalid", `{"type":"tool_search_output","id":"s","call_id":null,"status":"completed","execution":"server","tools":[null]}`, false},
		{"patch_output_invalid", `{"type":"apply_patch_call_output","id":"p","call_id":"c","status":"completed","output":{}}`, false},
		{"function_output_invalid", `{"type":"function_call_output","id":"f","status":"completed","output":[null]}`, false},
		{"local_output_invalid", `{"type":"local_shell_call_output","id":"l","output":null}`, false},
		{"computer_batch", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[{"type":"screenshot"}]}`, true},
		{"computer_output", `{"type":"computer_call_output","id":"c","call_id":"call","status":"completed","output":{"type":"computer_screenshot"}}`, true},
		{"custom_output", `{"type":"custom_tool_call_output","id":"c","call_id":"call","status":"completed","output":"ok"}`, true},
		{"approval", `{"type":"mcp_approval_response","id":"a","approval_request_id":"r","approve":true}`, true},
		{"program", `{"type":"program","id":"p","call_id":"c","code":"x","fingerprint":"fp"}`, true},
		{"program_output", `{"type":"program_output","id":"p","call_id":"c","status":"completed","result":"ok"}`, true},
		{"additional_tools", `{"type":"additional_tools","id":"a","role":"assistant","tools":[]}`, true},
		{"computer_batch_invalid", `{"type":"computer_call","id":"c","call_id":"call","status":"completed","pending_safety_checks":[],"actions":[null]}`, false},
		{"computer_output_invalid", `{"type":"computer_call_output","id":"c","call_id":"call","status":"completed","output":null}`, false},
		{"custom_output_invalid", `{"type":"custom_tool_call_output","id":"c","call_id":"call","status":"completed","output":{}}`, false},
		{"approval_invalid", `{"type":"mcp_approval_response","id":"a","approval_request_id":"r","approve":null}`, false},
		{"program_invalid", `{"type":"program","id":"p","call_id":"c","code":"x"}`, false},
		{"program_output_invalid", `{"type":"program_output","id":"p","call_id":"c","status":"completed","result":{}}`, false},
		{"additional_tools_invalid", `{"type":"additional_tools","id":"a","role":"assistant","tools":[null]}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"object":"response","status":"completed","output":[` + tt.item + `]}`
			result := execution.AttemptResult{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: []byte(body)}
			spec := execution.AttemptSpec{Operation: execution.OperationProbe, ClientProtocol: protocol.OpenAIResponses}
			normalizeGatewayProtocolProbeResult(spec, &result)
			if (result.Error == nil) != tt.valid {
				t.Fatalf("error = %+v, want valid %v", result.Error, tt.valid)
			}
		})
	}
}
