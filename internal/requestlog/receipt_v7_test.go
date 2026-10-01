package requestlog

import (
	"encoding/json"
	"fmt"
	"testing"

	"gpt-load/internal/platform/redact"
	"gpt-load/internal/storage/models"
)

func TestPersistedReceiptRejectsRetiredShapes(t *testing.T) {
	tests := []struct {
		name string
		edit func(map[string]any)
	}{
		{"retired factors", func(receipt map[string]any) {
			receipt["price_multipliers"] = map[string]any{"group": "1", "access_key": "1"}
		}},
		{"retired base total", func(receipt map[string]any) { receipt["base_total_nano_usd"] = 0 }},
		{"retired scope identity", func(receipt map[string]any) {
			receipt["rule"].(map[string]any)["scope_key"] = "provider:openai"
		}},
	}
	for version := 1; version <= 6; version++ {
		tests = append(tests, struct {
			name string
			edit func(map[string]any)
		}{fmt.Sprintf("schema v%d", version), func(receipt map[string]any) { receipt["schema_version"] = version }})
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			event := channelScopedEvent(t, "00000000-0000-4000-8000-000000009002")
			var receipt map[string]any
			if err := json.Unmarshal([]byte(event.Usage.Pricing.ReceiptJSON), &receipt); err != nil {
				t.Fatal(err)
			}
			test.edit(receipt)
			encoded, err := json.Marshal(receipt)
			if err != nil {
				t.Fatal(err)
			}
			event.Usage.Pricing.ReceiptJSON = string(encoded)
			if _, err := mapEvent(redact.New(), event); err == nil {
				t.Fatal("new write accepted retired receipt shape")
			}
			if _, err := decodeAttemptPricingReceipt(models.RequestLogAttempt{
				ChannelID: "openai", UpstreamModel: event.UpstreamModel, PricingReceipt: encoded,
			}); err == nil {
				t.Fatal("persisted decoder accepted retired receipt shape")
			}
		})
	}
}
