package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/accessquota"
	"gpt-load/internal/execution"
	"gpt-load/internal/pricing"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func TestHandlerBilledRetrySettlesAllFrozenAttemptCosts(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{StatusCode: 429, Header: make(http.Header), RequestWritten: true, ClassificationBody: []byte(`{"error":{"type":"rate_limit_error"}}`), Usage: usage.Result{State: usage.StatePartial, Tokens: usage.Tokens{UncachedInput: 1_000_000}}},
		{StatusCode: 200, Header: make(http.Header), RequestWritten: true, Usage: usage.Result{State: usage.StateComplete, Tokens: usage.Tokens{UncachedInput: 1_000_000}}},
	}}
	sink := &recordingRequestLogSink{}
	engine, handler, _, _ := newRequestLogHandlerTestRuntime(t, forwarder, &recordingAccessKeyRPMLimiter{}, sink, "sk-first", "sk-second")
	runtime := accessquota.NewRuntime()
	if err := runtime.Reconcile(map[uint][]accessquota.Rule{1: {{ID: 103, Revision: 1, Kind: accessquota.KindTotal, LimitNanoUSD: 10_000_000_000}}}); err != nil {
		t.Fatal(err)
	}

	handler.accessQuota = runtime
	provider := &mutableGatewayPriceTableProvider{table: mustGatewayPriceTable(t, 2_000_000_000, true)}
	handler.priceTables = provider
	forwarder.onCall = func(call int) {
		if call == 0 {
			provider.Publish(mustGatewayPriceTable(t, 3_000_000_000, true))
		}
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	events := sink.snapshot()
	if response.Code != 200 || len(events) != 1 || len(events[0].Attempts) != 2 {
		t.Fatalf("response=%d events=%+v", response.Code, events)
	}
	for i, want := range []int64{2_000_000_000, 3_000_000_000} {
		if got := events[0].Attempts[i].Pricing.EstimatedCostNanoUSD; got != want {
			t.Fatalf("attempt %d cost=%d want=%d", i+1, got, want)
		}
	}
	view := runtime.Snapshot(1, time.Now())
	if len(view.Rules) != 1 || view.Rules[0].UsedNanoUSD != 5_000_000_000 {
		t.Fatalf("quota = %+v, want 5000000000", view)
	}
}

func TestBufferedCancelAfterObservedBilledUsage(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	want := usage.Result{State: usage.StatePartial, Tokens: usage.Tokens{UncachedInput: 9, Output: 2}}
	executor := fakeExecutionExecutor{stream: func(ctx context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		_ = sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}})
		_ = sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventUsage, Usage: &execution.UsageEvidence{Normalized: want}})
		cancel()
		<-ctx.Done()
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: 200, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled, Summary: "canceled after observation"}}
	}}
	input := responsesExecutionForwardInput()
	input.BufferedStream = true
	r := NewExecutionForwarder(executor).ForwardStream(ctx, input, httptest.NewRecorder())
	if r.Usage != want || r.DispatchState != execution.DispatchMaybeSent || r.PayloadReleased {
		t.Fatalf("cancel result = %+v, want observed usage with no payload", r)
	}
}

func TestBilledUsageQuotaIncludesEarlierAttempt(t *testing.T) {
	recorder := &requestRecorder{
		attempts: []telemetry.Attempt{
			{Sequence: 1, Pricing: telemetry.PricingObservation{EstimatedCostNanoUSD: 17}},
			{Sequence: 2, Pricing: telemetry.PricingObservation{EstimatedCostNanoUSD: 31}},
		},
		usage: telemetry.UsageObservation{AttemptSequence: 2, Pricing: telemetry.PricingObservation{EstimatedCostNanoUSD: 31}},
	}
	if got := recorder.estimatedCostNanoUSD(); got != 48 {
		t.Fatalf("quota cost = %d, want 48 (terminal counted once)", got)
	}
}

type firstHeartbeatFailureWriter struct {
	header http.Header
	flush  bool
}

func (w *firstHeartbeatFailureWriter) Header() http.Header { return w.header }
func (*firstHeartbeatFailureWriter) WriteHeader(int)       {}
func (w *firstHeartbeatFailureWriter) Write(p []byte) (int, error) {
	if !w.flush {
		return 0, io.ErrClosedPipe
	}
	return len(p), nil
}
func (*firstHeartbeatFailureWriter) FlushError() error { return io.ErrClosedPipe }

func TestBufferedFirstHeartbeatAbortPreservesBilledUsage(t *testing.T) {
	for _, flush := range []bool{false, true} {
		t.Run(fmt.Sprintf("flush=%t", flush), func(t *testing.T) {
			want := usage.Result{State: usage.StatePartial, Tokens: usage.Tokens{UncachedInput: 9, Output: 2}}
			executor := fakeExecutionExecutor{stream: func(ctx context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
				_ = sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}})
				<-ctx.Done()
				return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: 200,
					Usage: &execution.UsageEvidence{Normalized: want}, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled, Summary: "observed billed work"}}
			}}
			input := responsesExecutionForwardInput()
			input.BufferedStream = true
			r := NewExecutionForwarder(executor).ForwardStream(t.Context(), input, &firstHeartbeatFailureWriter{header: make(http.Header), flush: flush})
			if r.Usage != want || r.DispatchState != execution.DispatchMaybeSent || !r.ResponseStarted || r.ExecutionError == nil || !r.HTTPCommitted || r.PayloadReleased || r.Stream.EndReason != StreamEndDownstreamWriteFailure {
				t.Fatalf("first-heartbeat abort = %+v, want retained executor evidence and downstream failure", r)
			}
		})
	}
}

func TestBilledAttemptWithoutPriceKeepsUnpricedReceipt(t *testing.T) {
	result := usage.Result{State: usage.StatePartial, Tokens: usage.Tokens{UncachedInput: 10, CacheWriteUnknown: 3}}
	got := quoteFrozenAttempt(frozenAttemptPricing{channelID: "openai", upstreamModel: "gpt-4o", applicable: true}, result, pricing.ModeStandard)
	if got.CostState != string(pricing.CostStateUnpriced) || got.EstimatedCostNanoUSD != 0 || got.ReceiptJSON == "" {
		t.Fatalf("unpriced usage = %+v, want zero cost and durable receipt", got)
	}
	var receipt pricing.Receipt
	if err := json.Unmarshal([]byte(got.ReceiptJSON), &receipt); err != nil {
		t.Fatal(err)
	}
	if err := pricing.ValidateReceipt(receipt); err != nil {
		t.Fatal(err)
	}
	if len(receipt.LineItems) != 2 || receipt.LineItems[0].Quantity != 10 || receipt.LineItems[1].Quantity != 3 || receipt.LineItems[0].State != pricing.ReceiptLineUnpriced || receipt.LineItems[1].State != pricing.ReceiptLineUnpriced {
		t.Fatalf("unpriced receipt = %+v", receipt)
	}
}

func TestBilledAttemptUsesOwnFrozenPrice(t *testing.T) {
	price := pricing.Price{NanoUSDPerMillion: 2_000_000, Set: true}
	table, err := pricing.NewTable([]pricing.Rule{{Identity: pricing.Identity{ChannelID: "openai", ModelID: "gpt-4o"}, Prices: pricing.Prices{Input: price}}})
	if err != nil {
		t.Fatal(err)
	}
	frozen := frozenAttemptPricing{channelID: "openai", upstreamModel: "gpt-4o", table: table, applicable: true}
	r := &requestRecorder{usageApplicable: true, attempts: []telemetry.Attempt{{Sequence: 1, GroupID: 1, ChannelID: "openai", CredentialID: 1, DispatchState: execution.DispatchMaybeSent, Usage: usage.Result{State: usage.StatePartial, Tokens: usage.Tokens{UncachedInput: 10}}}}, attemptPricing: []frozenAttemptPricing{frozen}}
	r.bindFailedAttemptUsage(0)
	if got := r.attempts[0].Pricing; got.EstimatedCostNanoUSD != 20 || got.ReceiptJSON == "" {
		t.Fatalf("attempt frozen price = %+v, want 20 and receipt", got)
	}
}

func TestHandlerRetainsBilledUsageOnCancellationAndTimeout(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, kind := range []execution.ErrorKind{execution.ErrorKindCanceled, execution.ErrorKindTimeout} {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, kind), func(t *testing.T) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				want := usage.Result{State: usage.StatePartial, Tokens: usage.Tokens{UncachedInput: 6, CacheRead: 4, Output: 2}}
				fail := func() execution.AttemptResult {
					if kind == execution.ErrorKindCanceled {
						cancel()
					}
					return execution.AttemptResult{DispatchState: execution.DispatchMaybeSent,
						Usage: &execution.UsageEvidence{Normalized: want},
						Error: &execution.ErrorEvidence{Kind: kind, Summary: "later upstream call failed"}}
				}
				executor := fakeExecutionExecutor{
					unary: func(context.Context, execution.AttemptSpec) execution.AttemptResult { return fail() },
					stream: func(context.Context, execution.AttemptSpec, execution.StreamSink) execution.StreamResult {
						r := fail()
						return execution.StreamResult{DispatchState: r.DispatchState, Usage: r.Usage, Error: r.Error}
					},
				}
				sink := &recordingRequestLogSink{}
				engine, _, _, _ := newRequestLogHandlerTestRuntime(t, NewExecutionForwarder(executor), &recordingAccessKeyRPMLimiter{}, sink, "sk-first")
				request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(fmt.Sprintf(`{"model":"gpt-4o","stream":%t}`, stream))).WithContext(ctx)
				request.Header.Set("Authorization", "Bearer gl-client")
				engine.ServeHTTP(httptest.NewRecorder(), request)
				events := sink.snapshot()
				if len(events) != 1 || events[0].Usage.Result != want || events[0].Usage.CredentialID != 1 || events[0].Usage.AttemptSequence != 1 {
					t.Fatalf("events = %+v, want billed usage=%+v on terminal attempt 1", events, want)
				}
			})
		}
	}
}
