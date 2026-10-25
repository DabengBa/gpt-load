package gateway

import (
	"context"
	"net/http"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/health"
)

func TestResponsesRecoveryRejectsLocalErrorAuthority(t *testing.T) {
	result := UpstreamResult{StatusCode: 400, DispatchState: execution.DispatchMaybeSent, ExecutionError: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, OriginHint: execution.ErrorOriginInternal, Code: "invalid_encrypted_content"}}
	if got := responsesRecoverySignal(result); got != "" {
		t.Fatalf("internal error authorized history cleanup: %s", got)
	}
}

func TestResponsesRecoveryBufferedIneligibleDoesNotReplay(t *testing.T) {
	calls := 0
	executor := fakeExecutionExecutor{stream: func(_ context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		calls++
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			t.Error(err)
		}
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindProvider, OriginHint: execution.ErrorOriginUpstream, Code: "invalid_encrypted_content", Summary: "encrypted content could not be verified"}}
	}}
	engine, _ := newBufferedResponsesFramingRuntime(t, NewExecutionForwarder(executor), 3, dialectGatewayGroup{id: 1, name: "only", upstreamURL: "https://unused.example", apiKeys: []string{"sk-first"}})
	response := performBufferedResponsesFramingRequest(t, engine, `{"model":"gpt-4o","stream":true,"store":true,"input":[{"type":"reasoning","encrypted_content":"opaque"}]}`)
	if calls != 1 {
		t.Fatalf("ineligible buffered attempts = %d, want 1", calls)
	}
	if got := response.Body.String(); got != bufferedStreamHeartbeat+bufferedResponsesFramingAmbiguousErrorEvent {
		t.Fatalf("ineligible terminal = %q", got)
	}
}

func TestResponsesRecoveryDoesNotOverrideSafetyOrSkipGroupRule(t *testing.T) {
	result := UpstreamResult{StatusCode: 200, DispatchState: execution.DispatchMaybeSent, Committed: true, HTTPCommitted: true, BufferedStream: true, ClientVisibleBytes: 14, ExecutionError: &execution.ErrorEvidence{Kind: execution.ErrorKindProvider, Code: "invalid_encrypted_content"}}
	for _, rule := range []health.RuleID{"safety.operation_replay_unsafe", "safety.downstream_write_failed", "custom.skip_group"} {
		decision := health.Decision{Retry: health.RetryNone, Effect: health.EffectSkipGroup, RuleID: rule}
		if got := responsesBufferedRecoveryDecision(result, decision, true); got != decision {
			t.Fatalf("rule %s overridden: %+v", rule, got)
		}
	}
}
