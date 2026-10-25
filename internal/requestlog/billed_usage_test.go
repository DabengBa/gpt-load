package requestlog

import (
	"encoding/json"
	"math"
	"testing"
	"time"

	"gpt-load/internal/platform/redact"
	"gpt-load/internal/pricing"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func billedObservation(t *testing.T, model string, tokens, rate int64, state usage.State) (usage.Result, telemetry.PricingObservation) {
	t.Helper()
	table, err := pricing.NewTable([]pricing.Rule{{Identity: pricing.Identity{ChannelID: "openai", ModelID: model}, Prices: pricing.Prices{Input: pricing.Price{NanoUSDPerMillion: pricing.NanoUSD(rate), Set: true}}}})
	if err != nil {
		t.Fatal(err)
	}
	result := usage.Result{State: state, Tokens: usage.Tokens{UncachedInput: tokens}}
	quote, receipt := table.QuoteForModeWithReceipt(pricing.Identity{ChannelID: "openai", ModelID: model}, result, pricing.ModeStandard)
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	return result, telemetry.PricingObservation{UpstreamModel: model, CostState: string(quote.State), PricingCompleteness: string(quote.Completeness), EstimatedCostNanoUSD: int64(quote.EstimatedCostNanoUSD), ReceiptJSON: string(encoded)}
}

func TestBilledEarlierAttemptRejectsMissingReceiptAndOverflow(t *testing.T) {
	for _, missing := range []bool{false, true} {
		t.Run(map[bool]string{true: "missing frozen receipt", false: "overflow"}[missing], func(t *testing.T) {
			event := channelScopedEvent(t, "00000000-0000-4000-8000-000000009902")
			event.Usage.Result, event.Usage.Pricing = billedObservation(t, event.UpstreamModel, 1, 1_000_000, usage.StateComplete)
			first := event.Attempts[0]
			first.Sequence = 2
			first.Usage, first.Pricing = billedObservation(t, first.UpstreamModel, 1, 1_000_000, usage.StatePartial)
			if missing {
				first.Pricing.ReceiptJSON = ""
			} else {
				first.Usage.Tokens.UncachedInput = math.MaxInt64
			}
			event.Attempts = append(event.Attempts, first)
			if _, err := mapEvent(redact.New(), event); err == nil {
				t.Fatal("accepted incomplete/overflowing earlier billed observation")
			}
		})
	}
}

// Real SQLite persistence/aggregation boundary; openRequestLogQueryDB owns its
// t.TempDir and cleanup. No global state or provider resources; budget 120s.
func TestBilledAttemptsPersistAndAggregateOnce(t *testing.T) {
	event := channelScopedEvent(t, "00000000-0000-4000-8000-000000009901")
	event.Usage.Result, event.Usage.Pricing = billedObservation(t, event.UpstreamModel, 20, 3_000_000, usage.StateComplete)
	terminal := event.Attempts[0]
	terminal.Sequence = 2
	terminal.Usage, terminal.Pricing = event.Usage.Result, event.Usage.Pricing
	event.Usage.AttemptSequence = 2
	first := event.Attempts[0]
	first.UpstreamModel = "earlier-model"
	first.CredentialID = 9
	first.StatusCode = 500
	first.Usage, first.Pricing = billedObservation(t, first.UpstreamModel, 10, 2_000_000, usage.StatePartial)
	event.Attempts = []telemetry.Attempt{first, terminal}
	row, err := mapEvent(redact.New(), event)
	if err != nil {
		t.Fatal(err)
	}
	if row.EstimatedCostNanoUSD != 80 || row.UncachedInputTokens != 30 || row.UsageState != string(usage.StatePartial) || row.CredentialID != terminal.CredentialID || row.UpstreamModel != terminal.UpstreamModel {
		t.Fatalf("request total/terminal attribution = %+v, want 80/30/partial with terminal identity", row)
	}
	db := openRequestLogQueryDB(t)
	service := newRequestLogTestService(db)
	for range 2 {
		if err := service.writer.WriteBatch(t.Context(), []models.RequestLog{row}); err != nil {
			t.Fatal(err)
		}
	}
	detail, err := service.Get(t.Context(), row.ID)
	if err != nil || len(detail.Attempts) != 2 {
		t.Fatalf("detail=%+v err=%v", detail, err)
	}
	for i, want := range []int64{20, 60} {
		if receipt := detail.Attempts[i].PricingReceipt; receipt == nil || int64(receipt.TotalNanoUSD) != want {
			t.Fatalf("attempt %d receipt=%+v, want %d", i+1, receipt, want)
		}
	}
	from := event.CompletedAt.Truncate(time.Hour)
	report, err := service.QueryUsage(t.Context(), UsageQuery{FromMS: from.UnixMilli(), ToMS: from.Add(time.Hour).UnixMilli(), Granularity: UsageGranularityHour})
	if err != nil || report.Summary.RequestCount != 1 || report.Summary.EstimatedCostNanoUSD != 80 || report.Summary.UncachedInputTokens != 30 {
		t.Fatalf("aggregate=%+v err=%v", report, err)
	}
}
