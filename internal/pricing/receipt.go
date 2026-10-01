package pricing

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// UnmarshalJSON accepts only the current schema and rejects unknown fields.
func (receipt *Receipt) UnmarshalJSON(data []byte) error {

	type receiptJSON Receipt
	var decoded receiptJSON
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	if decoded.SchemaVersion != 7 {
		return fmt.Errorf("unsupported pricing receipt schema version")
	}
	*receipt = Receipt(decoded)
	return nil
}

// ValidateReceipt verifies a persisted request-time receipt without consulting
// the mutable current pricing table.
func ValidateReceipt(receipt Receipt) error {
	if receipt.SchemaVersion != 7 ||
		receipt.Method != ReceiptMethodUnitRateSum || receipt.MethodVersion != 1 || receipt.Currency != "USD" {
		return fmt.Errorf("unsupported pricing receipt contract")
	}
	if !receipt.PricingMode.Valid() {
		return fmt.Errorf("invalid pricing receipt mode")
	}
	if err := validateIdentity(Identity{ChannelID: receipt.Rule.ChannelID, ModelID: receipt.Rule.ModelID}); err != nil {
		return fmt.Errorf("invalid pricing receipt rule: %w", err)
	}
	if receipt.ContextThresholdTokens != nil && *receipt.ContextThresholdTokens < 0 {
		return fmt.Errorf("invalid pricing receipt context threshold")
	}
	if receipt.TotalNanoUSD < 0 {
		return fmt.Errorf("invalid pricing receipt total")
	}

	total, err := validateReceiptLines(receipt)
	if err != nil {
		return err
	}
	if int64(total) != receipt.TotalNanoUSD {
		return fmt.Errorf("pricing receipt total mismatch")
	}
	return nil
}

func validateReceiptLines(receipt Receipt) (NanoUSD, error) {
	allowed := map[string]struct{}{
		"input": {}, "cache_read": {}, "cache_write_5m": {},
		"cache_write_1h": {}, "cache_write": {}, "output": {},
	}
	seen := make(map[string]struct{}, len(receipt.LineItems))
	total := NanoUSD(0)
	for _, line := range receipt.LineItems {
		if _, ok := allowed[line.Code]; !ok {
			return 0, fmt.Errorf("invalid pricing receipt line code %q", line.Code)
		}
		if _, exists := seen[line.Code]; exists {
			return 0, fmt.Errorf("duplicate pricing receipt line code %q", line.Code)
		}
		seen[line.Code] = struct{}{}
		if line.Quantity <= 0 || line.Multiplier.Numerator <= 0 ||
			line.Multiplier.Denominator <= 0 {
			return 0, fmt.Errorf("invalid pricing receipt line quantity or multiplier")
		}
		switch line.State {
		case ReceiptLinePriced:
			if line.RateNanoUSDPerMillion == nil || line.AmountNanoUSD == nil ||
				*line.RateNanoUSDPerMillion < 0 || *line.AmountNanoUSD < 0 {
				return 0, fmt.Errorf("invalid priced receipt line")
			}
			amount, ok := QuoteComponent(
				line.Quantity,
				NanoUSD(*line.RateNanoUSDPerMillion),
				line.Multiplier,
			)

			if !ok || int64(amount) != *line.AmountNanoUSD {
				return 0, fmt.Errorf("pricing receipt line amount mismatch")
			}
			total, ok = CheckedAddNanoUSD(total, amount)
			if !ok {
				return 0, fmt.Errorf("pricing receipt total overflows")
			}
		case ReceiptLineUnpriced:
			if line.RateNanoUSDPerMillion != nil || line.AmountNanoUSD != nil {
				return 0, fmt.Errorf("invalid unpriced receipt line")
			}
		default:
			return 0, fmt.Errorf("invalid pricing receipt line state")
		}
	}
	return total, nil
}
