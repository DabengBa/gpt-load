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

func TestProbeAnswerPresentExtractsEachProtocolShape(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		clientProtocol protocol.Protocol
		body           string
		want           probeExtraction
	}{
		{
			name: "openai chat string content", clientProtocol: protocol.OpenAICompletions,
			body: `{"choices":[{"message":{"role":"assistant","content":"4"}}]}`, want: probeExtraction{present: true, valid: true, text: "4"},
		},
		{
			name: "openai chat block content", clientProtocol: protocol.OpenAICompletions,
			body: `{"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"4"}]}}]}`, want: probeExtraction{present: true, valid: true, text: "4"},
		},
		{
			name: "openai chat empty content", clientProtocol: protocol.OpenAICompletions,
			body: `{"choices":[{"message":{"role":"assistant","content":"   "}}]}`, want: probeExtraction{valid: true},
		},
		{
			name: "openai chat missing choices", clientProtocol: protocol.OpenAICompletions,
			body: `{"id":"chat_1"}`, want: probeExtraction{},
		},
		{
			name: "openai responses output text", clientProtocol: protocol.OpenAIResponses,
			body: `{"output":[{"type":"message","content":[{"type":"output_text","text":"4"}]}]}`, want: probeExtraction{present: true, valid: true, text: "4"},
		},
		{
			name: "openai responses only reasoning", clientProtocol: protocol.OpenAIResponses,
			body: `{"output":[{"type":"reasoning","content":[{"type":"summary_text","text":"thinking"}]}]}`, want: probeExtraction{valid: true},
		},
		{
			name: "openai responses missing output", clientProtocol: protocol.OpenAIResponses,
			body: `{"id":"resp_1","status":"completed"}`, want: probeExtraction{},
		},
		{
			name: "anthropic text content", clientProtocol: protocol.Anthropic,
			body: `{"content":[{"type":"text","text":"4"}]}`, want: probeExtraction{present: true, valid: true, text: "4"},
		},
		{
			name: "anthropic thinking only", clientProtocol: protocol.Anthropic,
			body: `{"content":[{"type":"thinking","thinking":"hmm"}]}`, want: probeExtraction{valid: true},
		},
		{
			name: "anthropic missing content", clientProtocol: protocol.Anthropic,
			body: `{"id":"msg_1"}`, want: probeExtraction{},
		},
		{
			name: "gemini candidate part", clientProtocol: protocol.Gemini,
			body: `{"candidates":[{"content":{"parts":[{"text":"4"}]}}]}`, want: probeExtraction{present: true, valid: true, text: "4"},
		},
		{
			name: "gemini empty candidate", clientProtocol: protocol.Gemini,
			body: `{"candidates":[]}`, want: probeExtraction{valid: true},
		},
		{
			name: "gemini thought-only part is not an answer", clientProtocol: protocol.Gemini,
			body: `{"candidates":[{"content":{"parts":[{"thought":true,"text":"internal reasoning"}]}}]}`, want: probeExtraction{valid: true},
		},
		{
			name: "gemini thought plus answer", clientProtocol: protocol.Gemini,
			body: `{"candidates":[{"content":{"parts":[{"thought":true,"text":"internal reasoning"},{"text":"4"}]}}]}`, want: probeExtraction{present: true, valid: true, text: "4"},
		},
		{
			name: "gemini missing candidates", clientProtocol: protocol.Gemini,
			body: `{"modelVersion":"probe-upstream"}`, want: probeExtraction{},
		},
		{
			name: "empty body", clientProtocol: protocol.OpenAICompletions,
			body: ``, want: probeExtraction{},
		},
		{
			name: "invalid json", clientProtocol: protocol.OpenAICompletions,
			body: `<html>gateway error</html>`, want: probeExtraction{},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got := probeAnswerPresent(test.clientProtocol, []byte(test.body), false)
			if got != test.want {
				t.Fatalf("probeAnswerPresent() = %#v, want %#v for %s", got, test.want, test.body)
			}
			if got.text != test.want.text {
				t.Fatalf("extracted text = %q, want %q for %s", got.text, test.want.text, test.body)
			}
		})
	}
}

func TestNewProbeRequestCarriesQuestionAndContractBudget(t *testing.T) {
	t.Parallel()

	compatible := newProbeRequest(execution.AttemptSpec{Operation: execution.OperationProbe}, "openai-compatible", channel.ProviderOpenAICompatible, "probe-upstream", 16)
	if compatible.Params == nil || compatible.Params.MaxCompletionTokens != nil {
		t.Fatalf("compatible probe must not send max_completion_tokens: %#v", compatible.Params)
	}
	if compatible.Params.ExtraParams["max_tokens"] != 16 {
		t.Fatalf("compatible max_tokens = %#v", compatible.Params.ExtraParams)
	}
	if len(compatible.Input) != 1 || compatible.Input[0].Content == nil ||
		compatible.Input[0].Content.ContentStr == nil || *compatible.Input[0].Content.ContentStr != probeQuestion {
		t.Fatalf("compatible probe content = %#v", compatible.Input)
	}

	gated := newProbeRequest(execution.AttemptSpec{Operation: execution.OperationProbe}, "openai", channel.ProviderOpenAI, "probe-upstream", 16)
	if gated.Params == nil || gated.Params.MaxCompletionTokens == nil || *gated.Params.MaxCompletionTokens != 16 {
		t.Fatalf("openai probe max_completion_tokens = %#v", gated.Params)
	}

	gateway := newProbeRequest(execution.AttemptSpec{Operation: execution.OperationProbe}, "gpt-load", channel.ProviderMultiProtocolGateway, "probe-upstream", 16)
	if gateway.Params == nil || gateway.Params.MaxCompletionTokens != nil {
		t.Fatalf("gateway probe must not send max_completion_tokens: %#v", gateway.Params)
	}
	if gateway.Params.ExtraParams["max_tokens"] != 16 {
		t.Fatalf("gateway max_tokens = %#v", gateway.Params.ExtraParams)
	}
}

func TestNativeResponsesProbeUsesInputAndMaxOutputTokens(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		if request.Method != http.MethodPost || request.URL.Path != "/v1/responses" {
			t.Errorf("probe target = %s %s", request.Method, request.URL.Path)
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read probe body: %v", err)
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode probe body: %v", err)
		}
		if payload["model"] != "probe-upstream" {
			t.Errorf("model = %#v", payload["model"])
		}
		if payload["max_output_tokens"] != float64(testProbeOutputTokens) {
			t.Errorf("max_output_tokens = %#v, want %d", payload["max_output_tokens"], testProbeOutputTokens)
		}
		for _, forbidden := range []string{"max_tokens", "max_completion_tokens", "messages"} {
			if _, exists := payload[forbidden]; exists {
				t.Errorf("Responses probe carried %s: %s", forbidden, body)
			}
		}
		if !strings.Contains(string(body), probeQuestion) {
			t.Errorf("probe question missing: %s", body)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"probe-upstream","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"4","annotations":[]}]}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}`)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, openAIBaseURL: server.URL})
	spec := utilitySpec(channel.OpenAI, protocol.OpenAIResponses, execution.OperationProbe, "", "", nil)
	spec.ClientModel, spec.UpstreamModel = "probe-client", "probe-upstream"
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK {
		t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
	}
	if result.ProbeAnswerPresent != true {
		t.Fatalf("Responses probe answer not observed: %+v body=%s", result, result.Body)
	}
}

func TestRawPassthroughProbeDoesNotUseOpenAIChatFallback(t *testing.T) {
	t.Parallel()

	got := probeAnswerPresent(
		protocol.Gemini,
		[]byte(`{"choices":[{"message":{"content":"4"}}]}`),
		true,
	)
	if got.present || got.valid {
		t.Fatalf("raw Gemini probe accepted OpenAI Chat shape: %#v", got)
	}
}

func TestProbeWithoutGeneratedTextDoesNotSetAnswerPresent(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chat_1","object":"chat.completion","created":1,"model":"probe-upstream","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.OpenAICompatible, protocol.OpenAICompletions, execution.OperationProbe, "", "", nil)
	spec.ClientModel, spec.UpstreamModel = "probe-client", "probe-upstream"
	target, err := json.Marshal(map[string]string{"base_url": server.URL})
	if err != nil {
		t.Fatalf("marshal target: %v", err)
	}
	spec.TargetConfig = target
	spec = freezeTestAttempt(spec)
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v", err)
	}
	if result.StatusCode != http.StatusOK || result.Error != nil {
		t.Fatalf("result = %+v", result)
	}
	if result.ProbeAnswerPresent {
		t.Fatalf("empty probe response reported an answer: %+v", result)
	}
	if result.ProbeResponseInvalid {
		t.Fatalf("legal empty probe response reported as invalid: %+v", result)
	}
}

func TestNormalizeProbeAttemptResultDistinguishesNoAnswerFromInvalidShape(t *testing.T) {
	t.Parallel()

	spec := execution.AttemptSpec{
		Operation: execution.OperationProbe, ClientProtocol: protocol.OpenAICompletions,
	}

	// A legal chat shape without generated text is a no-answer, not corruption.
	noAnswer := &execution.AttemptResult{
		DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
		StatusCode: http.StatusOK, Header: http.Header{},
		Body: []byte(`{"choices":[]}`),
	}
	normalizeProbeAttemptResult(spec, noAnswer, false)
	if noAnswer.ProbeAnswerPresent || noAnswer.ProbeResponseInvalid {
		t.Fatalf("legal empty response must stay a plain no-answer: %+v", noAnswer)
	}

	// Valid JSON without the selected protocol's shape is invalid, not empty.
	wrongShape := &execution.AttemptResult{
		DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
		StatusCode: http.StatusOK, Header: http.Header{},
		Body: []byte(`{"unexpected":"payload"}`),
	}
	normalizeProbeAttemptResult(spec, wrongShape, false)
	if wrongShape.ProbeAnswerPresent || !wrongShape.ProbeResponseInvalid {
		t.Fatalf("wrong protocol shape must be invalid, not empty: %+v", wrongShape)
	}

	// A non-JSON body cannot be parsed as any protocol shape.
	nonJSON := &execution.AttemptResult{
		DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
		StatusCode: http.StatusOK, Header: http.Header{},
		Body: []byte(`<html>gateway error</html>`),
	}
	normalizeProbeAttemptResult(spec, nonJSON, false)
	if nonJSON.ProbeAnswerPresent || !nonJSON.ProbeResponseInvalid {
		t.Fatalf("non-JSON body must be invalid, not empty: %+v", nonJSON)
	}

	// Error and non-2xx results never carry probe evidence.
	withError := &execution.AttemptResult{
		DispatchState: execution.DispatchMaybeSent, StatusCode: http.StatusServiceUnavailable,
		Header: http.Header{}, Body: []byte(`{"choices":[{"message":{"content":"4"}}]}`),
		Error: &execution.ErrorEvidence{Kind: execution.ErrorKindHTTP},
	}
	normalizeProbeAttemptResult(spec, withError, false)
	if withError.ProbeAnswerPresent || withError.ProbeResponseInvalid {
		t.Fatalf("error results must not carry probe evidence: %+v", withError)
	}
}

func TestNormalizeProbeAttemptResultDecodesContentEncoding(t *testing.T) {
	t.Parallel()

	var encoded bytes.Buffer
	writer := gzip.NewWriter(&encoded)
	if _, err := writer.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"4"}]}}]}`)); err != nil {
		t.Fatalf("gzip.Write() error = %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("gzip.Close() error = %v", err)
	}

	result := &execution.AttemptResult{
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusOK,
		Header:          http.Header{"Content-Encoding": []string{"gzip"}},
		Body:            encoded.Bytes(),
	}
	normalizeProbeAttemptResult(execution.AttemptSpec{
		Operation:      execution.OperationProbe,
		ClientProtocol: protocol.Gemini,
	}, result, true)
	if !result.ProbeAnswerPresent || result.ProbeResponseInvalid {
		t.Fatalf("gzip probe response evidence = %+v", result)
	}
}

func TestGeminiProbeIgnoresThoughtOnlyPart(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"internal reasoning"}]},"finishReason":"STOP"}]}`)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true, geminiBaseURL: server.URL + "/v1beta"})
	spec := utilitySpec(channel.Gemini, protocol.Gemini, execution.OperationProbe, "", "", nil)
	spec.ClientModel, spec.UpstreamModel = "probe-client", "probe-upstream"
	result := runtime.Execute(context.Background(), spec)
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v", err)
	}
	if result.StatusCode != http.StatusOK || result.Error != nil {
		t.Fatalf("result = %+v", result)
	}
	if result.ProbeAnswerPresent || result.ProbeResponseInvalid {
		t.Fatalf("thought-only Gemini probe must be a valid no-answer: %+v body=%s", result, result.Body)
	}
}

func TestMultiProtocolGatewayProbeSerializesMaxTokensToChatWire(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		if request.Method != http.MethodPost || request.URL.Path != "/v1/chat/completions" {
			t.Errorf("probe target = %s %s", request.Method, request.URL.Path)
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read probe body: %v", err)
			return
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode probe body: %v", err)
			return
		}
		// The contract budget must reach the gateway's /v1/chat/completions wire
		// as the legacy max_tokens field, never max_completion_tokens.
		if payload["max_tokens"] != float64(testProbeOutputTokens) {
			t.Errorf("gateway max_tokens = %#v, want %d", payload["max_tokens"], testProbeOutputTokens)
		}
		for _, forbidden := range []string{"max_completion_tokens", "input"} {
			if _, exists := payload[forbidden]; exists {
				t.Errorf("gateway probe carried %s: %s", forbidden, body)
			}
		}
		if !strings.Contains(string(body), probeQuestion) {
			t.Errorf("probe question missing: %s", body)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chat_1","object":"chat.completion","created":1,"model":"probe-upstream","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.GPTLoad, protocol.OpenAICompletions, execution.OperationProbe, "", "", nil)
	spec.TargetConfig = json.RawMessage(`{"base_url":"` + server.URL + `"}`)
	spec.ClientModel, spec.UpstreamModel = "probe-client", "probe-upstream"
	result := runtime.Execute(context.Background(), freezeTestAttempt(spec))
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK {
		t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
	}
	if !result.ProbeAnswerPresent || result.ProbeResponseInvalid {
		t.Fatalf("gateway probe answer not observed: %+v body=%s", result, result.Body)
	}
}

// TestModelTraceChallengeRunsOnTheProbeWireWithItsOwnPrompt proves the
// fingerprint challenge is wire-identical to a probe — same target, same token
// field, same budget contract — while carrying its own prompt and its own
// result field. A challenge must never inherit the probe question or the probe
// answer flags, otherwise attribution would silently drift from the bank the
// challenge came from.
func TestModelTraceChallengeRunsOnTheProbeWireWithItsOwnPrompt(t *testing.T) {
	t.Parallel()

	const challengePrompt = "Provide exactly 218 integers from 1 to 355 inclusive, one per position."
	const challengeAnswer = "17 42 113 250"
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		if request.Method != http.MethodPost || request.URL.Path != "/v1/chat/completions" {
			t.Errorf("challenge target = %s %s", request.Method, request.URL.Path)
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read challenge body: %v", err)
			return
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode challenge body: %v", err)
			return
		}
		if payload["model"] != "challenge-upstream" {
			t.Errorf("challenge model = %#v", payload["model"])
		}
		if payload["max_tokens"] != float64(testProbeOutputTokens) {
			t.Errorf("challenge max_tokens = %#v, want %d", payload["max_tokens"], testProbeOutputTokens)
		}
		if _, exists := payload["max_completion_tokens"]; exists {
			t.Errorf("challenge carried max_completion_tokens: %s", body)
		}
		if !strings.Contains(string(body), challengePrompt) {
			t.Errorf("challenge prompt missing from wire: %s", body)
		}
		if strings.Contains(string(body), probeQuestion) {
			t.Errorf("challenge sent the probe question instead of its own prompt: %s", body)
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("X-Request-Id", "challenge-request")
		_, _ = io.WriteString(writer, `{"id":"chat_1","object":"chat.completion","created":1,"model":"challenge-upstream","choices":[{"index":0,"message":{"role":"assistant","content":"`+challengeAnswer+`"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}`)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.GPTLoad, protocol.OpenAICompletions, execution.OperationModelTrace, "", "", nil)
	spec.TargetConfig = json.RawMessage(`{"base_url":"` + server.URL + `"}`)
	spec.ClientModel, spec.UpstreamModel = "challenge-client", "challenge-upstream"
	spec.PromptText = challengePrompt
	spec.ProbeMaxOutputTokens = testProbeOutputTokens
	result := runtime.Execute(context.Background(), freezeTestAttempt(spec))
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v; result=%+v", err, result)
	}
	if calls.Load() != 1 || result.Error != nil || result.StatusCode != http.StatusOK {
		t.Fatalf("calls/result = %d/%+v", calls.Load(), result)
	}
	if result.GeneratedText != challengeAnswer || result.GeneratedTextInvalid {
		t.Fatalf("challenge answer = %q/%t, want %q", result.GeneratedText, result.GeneratedTextInvalid, challengeAnswer)
	}
	if result.UpstreamRequestID != "challenge-request" {
		t.Fatalf("upstream request id = %q", result.UpstreamRequestID)
	}
	if result.ProbeAnswerPresent || result.ProbeResponseInvalid {
		t.Fatalf("challenge wrote probe judgement fields: %+v", result)
	}
	if result.Usage == nil || result.Usage.Normalized.State != usage.StateComplete {
		t.Fatalf("challenge usage = %+v, want the provider's own usage", result.Usage)
	}
}

// TestModelTraceChallengeRejectsEmptyPrompt proves the challenge contract still
// refuses a request without a prompt: an empty prompt would be sent as a
// meaningless model call, so it must fail before dispatch.
func TestModelTraceChallengeRejectsEmptyPrompt(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"choices":[{"index":0,"message":{"role":"assistant","content":"1"}}]}`)
	}))
	defer server.Close()

	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.GPTLoad, protocol.OpenAICompletions, execution.OperationModelTrace, "", "", nil)
	spec.TargetConfig = json.RawMessage(`{"base_url":"` + server.URL + `"}`)
	spec.ClientModel, spec.UpstreamModel = "challenge-client", "challenge-upstream"
	spec.PromptText = "   "
	spec.ProbeMaxOutputTokens = testProbeOutputTokens
	result := runtime.Execute(context.Background(), freezeTestAttempt(spec))
	if err := result.Validate(); err != nil {
		t.Fatalf("result validation: %v", err)
	}
	if result.DispatchState != execution.DispatchNotSent || result.Error == nil {
		t.Fatalf("empty challenge prompt = %+v, want a not-sent contract failure", result)
	}
	if calls.Load() != 0 {
		t.Fatalf("upstream calls = %d, want none for an empty challenge prompt", calls.Load())
	}
}
