package cpa

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/gateway"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/usage"
)

type usageStreamFixtureBridge struct {
	providerBridge
	chunks <-chan providerStreamChunk
}

func (bridge usageStreamFixtureBridge) ExecuteStream(context.Context, string, providerCredential, providerRequest) (*providerStreamResponse, error) {
	return &providerStreamResponse{Headers: http.Header{"Content-Type": {"text/event-stream"}}, Chunks: bridge.chunks}, nil
}

func TestCPAAnthropicStreamUsageBoundaries(t *testing.T) {
	const start = "event: message_start\r\ndata: " + `{"type":"message_start","message":{"id":"msg_1","model":"upstream","content":[],"usage":{"input_tokens":902,"output_tokens":0}}}` + "\r\n\r\n"
	const delta = "event: message_delta\r\ndata: " + `{"type":"message_delta","usage":{"input_tokens":100,"cache_read_input_tokens":900,"output_tokens":10},"delta":{"stop_reason":"end_turn"}}` + "\r\n\r\n"
	const stop = "event: message_stop\r\ndata: " + `{"type":"message_stop"}` + "\r\n\r\n"
	for _, test := range []struct {
		name   string
		native bool
		wire   string
		err    error
		want   usage.Tokens
		state  usage.State
	}{
		{name: "combined events and model alias", wire: start + delta + stop,
			want: usage.Tokens{UncachedInput: 100, CacheRead: 900, Output: 10}, state: usage.StateComplete},
		{name: "native initial usage stays authoritative", native: true,
			wire: start + strings.Replace(delta, `"input_tokens":100,`, "", 1) + stop,
			want: usage.Tokens{UncachedInput: 902, CacheRead: 900, Output: 10}, state: usage.StateComplete},
		{name: "read failure cannot bill the estimate", wire: start, err: io.ErrUnexpectedEOF, state: usage.StatePartial},
		{name: "cancellation cannot bill the estimate", wire: start, err: context.Canceled, state: usage.StatePartial},
	} {
		t.Run(test.name, func(t *testing.T) {
			adapter, _, _, keyService, row := newAdapterFixture(t, credentialJSON("access", "refresh", time.Now().Add(time.Hour)))
			spec := validSpec(t, row, keyService)
			kind := channel.ProviderCodex
			spec.ClientProtocol, spec.RouteMode = protocol.Anthropic, execution.RouteConverted
			spec.Operation, spec.Path = execution.OperationChatCompletion, "/v1/messages"
			spec.Body = []byte(`{"model":"client-model","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"hello"}]}`)
			if test.native {
				adapter, keyService, row = newClaudeAdapterFixture(t)
				spec = validClaudeSpec(t, row, keyService)
				kind = channel.ProviderClaude
			}
			spec.ClientModel = "client-model"
			chunks := make(chan providerStreamChunk, 2)
			chunks <- providerStreamChunk{Payload: []byte(test.wire)}
			if test.err != nil {
				chunks <- providerStreamChunk{Err: test.err}
			}
			close(chunks)
			adapter.providers[kind] = usageStreamFixtureBridge{providerBridge: adapter.providers[kind], chunks: chunks}
			input := gateway.ForwardInput{
				Dialect: dialect.NewAnthropic(), Group: state.GroupView{ID: row.GroupID},
				Request:   &dialect.ParsedRequest{Method: spec.Method, Path: spec.Path, Body: spec.Body},
				RequestID: spec.RequestID, AttemptID: spec.AttemptID, AttemptSequence: spec.Sequence,
				ClientProtocol: spec.ClientProtocol, Operation: spec.Operation, ChannelID: spec.ChannelID,
				RouteMode: spec.RouteMode, TargetConfig: spec.TargetConfig, Credential: spec.Credential,
				ExternalModel: spec.ClientModel, UpstreamModelID: spec.UpstreamModel, ObserveUsage: true,
			}
			downstream := httptest.NewRecorder()
			result := gateway.NewExecutionForwarder(adapter).ForwardStream(t.Context(), input, downstream)
			if result.Err != nil || (result.ExecutionError != nil) != (test.err != nil) {
				t.Fatalf("forward error = %v, execution error = %+v, want failure=%t", result.Err, result.ExecutionError, test.err != nil)
			}
			if result.Usage.Tokens != test.want || result.Usage.State != test.state ||
				result.Usage.Diagnostics.Has(usage.DiagnosticInvalidEventSequence) {
				t.Fatalf("usage = %+v, want %+v/%s; wire=%s", result.Usage, test.want, test.state, downstream.Body)
			}
			if !strings.Contains(downstream.Body.String(), `"model":"client-model"`) {
				t.Fatalf("client model alias lost: %s", downstream.Body)
			}
		})
	}
}
