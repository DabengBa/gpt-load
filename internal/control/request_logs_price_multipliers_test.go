package control

import (
	"encoding/json"
	"testing"

	"gpt-load/internal/pricing"
)

func TestMapRequestLogReceiptIncludesOnlyFrozenBasePrice(t *testing.T) {
	var receipt pricing.Receipt
	if err := json.Unmarshal([]byte(`{
		"schema_version":7,"method":"unit_rate_sum","method_version":1,
		"currency":"USD","pricing_mode":"standard",
		"rule":{"channel_id":"openai","model_id":"gpt-4o"},
		"line_items":[{"code":"input","quantity":1000,"rate_nano_usd_per_million":2000000000,
		"multiplier":{"numerator":1,"denominator":1},"state":"priced","amount_nano_usd":2000000}],
		"total_nano_usd":2000000
	}`), &receipt); err != nil {
		t.Fatal(err)
	}
	response, err := mapRequestLogPricingReceipt(&receipt)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	var actual map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &actual); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"price_multipliers", "base_total_nano_usd"} {
		if _, present := actual[field]; present {
			t.Fatalf("receipt exposes retired field %s: %s", field, encoded)
		}
	}
	if response.SchemaVersion != 7 || response.Rule.ChannelID == nil || *response.Rule.ChannelID != "openai" ||
		response.TotalNanoUSD != "2000000" || len(response.LineItems) != 1 ||
		response.LineItems[0].AmountNanoUSD == nil || *response.LineItems[0].AmountNanoUSD != "2000000" {
		t.Fatalf("receipt response = %s", encoded)
	}
}
