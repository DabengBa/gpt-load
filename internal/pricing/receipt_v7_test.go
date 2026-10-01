package pricing

import (
	"encoding/json"
	"fmt"
	"testing"

	"gpt-load/internal/usage"
)

func TestReceiptV7GenerationAndRoundTrip(t *testing.T) {
	identity := Identity{ChannelID: "anthropic", ModelID: "model"}
	table := mustTable(t, Rule{Identity: identity, Prices: Prices{Input: fixedPrice(600_000), Output: fixedPrice(600_000), CacheWrite: fixedPrice(5)}})
	quote, receipt := table.QuoteForModeWithReceipt(identity, usage.Result{State: usage.StateComplete, Tokens: usage.Tokens{UncachedInput: 1, Output: 1, CacheWrite1H: 1_000_000}}, ModeStandard)
	if receipt == nil || receipt.SchemaVersion != 7 {
		t.Fatalf("receipt = %#v, want schema 7", receipt)
	}
	if quote.EstimatedCostNanoUSD != 10 || receipt.TotalNanoUSD != 10 || receipt.LineItems[1].Multiplier != (Multiplier{Numerator: 8, Denominator: 5}) {
		t.Fatalf("quote = %#v, receipt = %#v; want rounded components 1+8+1", quote, receipt)
	}
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"price_multipliers", "base_total_nano_usd"} {
		if _, exists := fields[field]; exists {
			t.Fatalf("retired field %s in %s", field, data)
		}
	}
	var decoded Receipt
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReceipt(decoded); err != nil {
		t.Fatal(err)
	}
}

func TestReceiptV7RejectsOtherSchemas(t *testing.T) {
	for _, version := range []int{0, 1, 2, 3, 4, 5, 6, 8} {
		t.Run(fmt.Sprint(version), func(t *testing.T) {
			data := fmt.Sprintf(`{"schema_version":%d,"method":"unit_rate_sum","method_version":1,"currency":"USD","pricing_mode":"standard","rule":{"channel_id":"openai","model_id":"model"},"line_items":[],"total_nano_usd":0}`, version)
			var receipt Receipt
			if err := json.Unmarshal([]byte(data), &receipt); err == nil {
				t.Fatal("decoded unsupported schema")
			}
			if err := ValidateReceipt(Receipt{SchemaVersion: version, Method: ReceiptMethodUnitRateSum, MethodVersion: 1, Currency: "USD", PricingMode: ModeStandard, Rule: ReceiptRule{ChannelID: "openai", ModelID: "model"}}); err == nil {
				t.Fatal("validated unsupported schema")
			}
		})
	}
}

func TestReceiptV7RejectsRetiredAndUnknownFields(t *testing.T) {
	for _, field := range []string{`"price_multipliers":null`, `"price_multipliers":{"group":"1","access_key":"1"}`, `"base_total_nano_usd":null`, `"base_total_nano_usd":0`, `"unknown":1`, `"rule":{"scope_key":"provider:openai","model_id":"model"}`} {
		var receipt Receipt
		if err := json.Unmarshal([]byte(`{"schema_version":7,`+field+`}`), &receipt); err == nil {
			t.Fatalf("accepted %s", field)
		}
	}
}

func TestReceiptV7RejectsTampering(t *testing.T) {
	identity := Identity{ChannelID: "openai", ModelID: "model"}
	table := mustTable(t, Rule{Identity: identity, Prices: Prices{Input: fixedPrice(100)}})
	for _, test := range []struct {
		name   string
		mutate func(*Receipt)
	}{
		{"method", func(r *Receipt) { r.Method = "other" }},
		{"method version", func(r *Receipt) { r.MethodVersion = 2 }},
		{"currency", func(r *Receipt) { r.Currency = "EUR" }},
		{"negative threshold", func(r *Receipt) { r.ContextThresholdTokens = new(int64(-1)) }},
		{"total mismatch", func(r *Receipt) { r.TotalNanoUSD++ }},
		{"unknown code", func(r *Receipt) { r.LineItems[0].Code = "other" }},
		{"duplicate code", func(r *Receipt) { r.LineItems = append(r.LineItems, r.LineItems[0]) }},
		{"zero quantity", func(r *Receipt) { r.LineItems[0].Quantity = 0 }},
		{"zero numerator", func(r *Receipt) { r.LineItems[0].Multiplier.Numerator = 0 }},
		{"zero denominator", func(r *Receipt) { r.LineItems[0].Multiplier.Denominator = 0 }},
		{"missing rate", func(r *Receipt) { r.LineItems[0].RateNanoUSDPerMillion = nil }},
		{"missing amount", func(r *Receipt) { r.LineItems[0].AmountNanoUSD = nil }},
		{"amount mismatch", func(r *Receipt) { (*r.LineItems[0].AmountNanoUSD)++ }},
		{"unpriced with amount", func(r *Receipt) { r.LineItems[0].State = ReceiptLineUnpriced }},
		{"unknown state", func(r *Receipt) { r.LineItems[0].State = "other" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, receipt := table.QuoteWithReceipt(identity, usage.Result{State: usage.StateComplete, Tokens: usage.Tokens{UncachedInput: 1_000_000}})
			test.mutate(receipt)
			if err := ValidateReceipt(*receipt); err == nil {
				t.Fatal("accepted tampered receipt")
			}
		})
	}
}
