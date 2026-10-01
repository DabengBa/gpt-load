package pricing

import (
	"strings"
	"testing"
)

func TestValidateReceiptV7IdentityAndMode(t *testing.T) {
	valid := Receipt{SchemaVersion: 7, Method: ReceiptMethodUnitRateSum, MethodVersion: 1, Currency: "USD", PricingMode: ModeStandard, Rule: ReceiptRule{ChannelID: "openai", ModelID: "model"}}
	if err := ValidateReceipt(valid); err != nil {
		t.Fatal(err)
	}
	for _, channelID := range []string{"", " openai", "openai ", "openai\n", strings.Repeat("c", 65)} {
		receipt := valid
		receipt.Rule.ChannelID = channelID
		if err := ValidateReceipt(receipt); err == nil {
			t.Fatalf("accepted channel ID %q", channelID)
		}
	}
	for _, modelID := range []string{"", " model", "model\n"} {
		receipt := valid
		receipt.Rule.ModelID = modelID
		if err := ValidateReceipt(receipt); err == nil {
			t.Fatalf("accepted model ID %q", modelID)
		}
	}
	for _, mode := range []Mode{"", "Invalid Mode"} {
		receipt := valid
		receipt.PricingMode = mode
		if err := ValidateReceipt(receipt); err == nil {
			t.Fatalf("accepted mode %q", mode)
		}
	}
}
