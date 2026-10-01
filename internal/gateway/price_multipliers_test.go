package gateway

import (
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/accessquota"
	"gpt-load/internal/pricing"
	"gpt-load/internal/state"
	"gpt-load/internal/usage"
)

func TestHandlerFreezesBasePricesAndAccountsTheSameEstimate(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "ordinary"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			forwarder := &scriptedForwarder{results: []UpstreamResult{{
				StatusCode: http.StatusOK, Header: make(http.Header), RequestWritten: true,
				Body:  []byte(`{"ok":true}`),
				Usage: usage.Result{State: usage.StateComplete, Tokens: usage.Tokens{UncachedInput: 1, CacheWrite1H: 1_000_000, Output: 1}},
			}}}
			if stream {
				forwarder.results[0].Committed = true
				forwarder.results[0].Stream = StreamObservation{EndReason: StreamEndCleanEOF}
				forwarder.streamResults = forwarder.results
			}
			sink := &recordingRequestLogSink{}
			engine, handler, manager, _ := newRequestLogHandlerTestRuntime(
				t, forwarder, &recordingAccessKeyRPMLimiter{}, sink, "sk-first",
			)
			runtime := accessquota.NewRuntime()
			rules := []accessquota.Rule{{ID: 901, Revision: 1, Kind: accessquota.KindTotal, LimitNanoUSD: 10_000_000}}
			if err := runtime.Reconcile(map[uint][]accessquota.Rule{1: rules}); err != nil {
				t.Fatal(err)
			}
			handler.accessQuota = runtime
			table, err := pricing.NewTable([]pricing.Rule{{
				Identity: pricing.Identity{ChannelID: "openai", ModelID: "gpt-4o"},
				Prices: pricing.Prices{
					Input:      pricing.Price{NanoUSDPerMillion: 600_000, Set: true},
					Output:     pricing.Price{NanoUSDPerMillion: 600_000, Set: true},
					CacheWrite: pricing.Price{NanoUSDPerMillion: 5, Set: true},
				},
			}})
			if err != nil {
				t.Fatal(err)
			}
			provider := &mutableGatewayPriceTableProvider{table: table}
			handler.priceTables = provider
			input := gatewayAccessQuotaCompileInput(handler, rules)
			if _, err := manager.Publish(input); err != nil {
				t.Fatal(err)
			}
			forwarder.onCall = func(int) {
				provider.table = mustGatewayPriceTable(t, 9_000_000_000, true)
				if _, err := manager.Publish(input); err != nil {
					t.Fatal(err)
				}
			}
			forwarder.onStreamCall = func(index int, _ http.ResponseWriter) { forwarder.onCall(index) }
			body := `{"model":"gpt-4o"}`
			if stream {
				body = `{"model":"gpt-4o","stream":true}`
			}
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			request.Header.Set("Authorization", "Bearer gl-client")
			response := httptest.NewRecorder()
			engine.ServeHTTP(response, request)
			if response.Code != http.StatusOK {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
			events := sink.snapshot()
			if len(events) != 1 || events[0].Usage.Pricing.EstimatedCostNanoUSD != 10 {
				t.Fatalf("frozen base estimate = %#v, want rounded components 1+8+1", events)
			}
			var receipt struct {
				SchemaVersion    int                   `json:"schema_version"`
				BaseTotalNanoUSD *int64                `json:"base_total_nano_usd"`
				PriceMultipliers json.RawMessage       `json:"price_multipliers"`
				TotalNanoUSD     int64                 `json:"total_nano_usd"`
				LineItems        []pricing.ReceiptLine `json:"line_items"`
			}
			if err := json.Unmarshal([]byte(events[0].Usage.Pricing.ReceiptJSON), &receipt); err != nil {
				t.Fatal(err)
			}
			if receipt.SchemaVersion != 7 || len(receipt.PriceMultipliers) != 0 || receipt.BaseTotalNanoUSD != nil || receipt.TotalNanoUSD != 10 ||
				len(receipt.LineItems) != 3 || receipt.LineItems[1].Code != "cache_write_1h" ||
				receipt.LineItems[1].Multiplier != (pricing.Multiplier{Numerator: 8, Denominator: 5}) {
				t.Fatalf("frozen receipt = %#v", receipt)
			}
			view := runtime.Snapshot(1, time.Now())
			if len(view.Rules) != 1 || view.Rules[0].UsedNanoUSD != 10 {
				t.Fatalf("quota = %#v, want the same base estimate", view)
			}
		})
	}
}

func TestHandlerCrossGroupRetryUsesFinalUsageBasePrice(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{
			StatusCode: http.StatusTooManyRequests, Header: make(http.Header), RequestWritten: true,
			Body:               []byte(`{"error":{"type":"rate_limit_error"}}`),
			ClassificationBody: []byte(`{"error":{"type":"rate_limit_error"}}`),
			Usage:              usage.Result{State: usage.StateComplete, Tokens: usage.Tokens{UncachedInput: 1_000_000}},
		},
		{
			StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`), RequestWritten: true,
			Usage: usage.Result{State: usage.StateComplete, Tokens: usage.Tokens{UncachedInput: 1000}},
		},
	}}
	sink := &recordingRequestLogSink{}
	engine, handler, manager, _ := newRequestLogHandlerTestRuntime(
		t, forwarder, &recordingAccessKeyRPMLimiter{}, sink, "sk-first", "sk-second",
	)
	input := gatewayAccessQuotaCompileInput(handler, nil)
	second := input.Groups[0]
	second.ID, second.Name = 2, "second"
	input.Groups = append(input.Groups, second)
	input.Credentials = append(input.Credentials, state.CredentialConfig{
		ID: 2, GroupID: 2, Version: 1, IdentityGeneration: 2, Fingerprint: "credential-2",
	})
	if _, err := manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
	handler.priceTables = &mutableGatewayPriceTableProvider{table: mustGatewayPriceTable(t, 2_000_000_000, true)}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	events := sink.snapshot()
	if response.Code != http.StatusOK || len(events) != 1 || len(events[0].Attempts) != 2 ||
		events[0].Usage.GroupID != 2 || events[0].Usage.AttemptSequence != 2 || events[0].Usage.Pricing.EstimatedCostNanoUSD != 2_000_000 {
		t.Fatalf("retry response = %d; events = %#v", response.Code, events)
	}
}
