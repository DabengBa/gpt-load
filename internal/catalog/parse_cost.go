package catalog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/big"
	"strconv"
	"strings"

	"gpt-load/internal/pricing"
)

func parseModePrices(raw json.RawMessage) (pricing.Prices, error) {
	fields, err := decodeCanonicalObject(raw, "mode cost", costCanonicalFields)
	if err != nil {
		return pricing.Prices{}, err
	}
	input, output, cacheRead, cacheWrite, err := decodePriceFields(fields)
	if err != nil {
		return pricing.Prices{}, err
	}
	return parsePrices(input, output, cacheRead, cacheWrite)
}

func parseCost(raw json.RawMessage) (*ModelCost, error) {
	fields, err := decodeCanonicalObject(raw, "cost", costCanonicalFields)
	if err != nil {
		return nil, err
	}
	input, output, cacheRead, cacheWrite, err := decodePriceFields(fields)
	if err != nil {
		return nil, err
	}
	prices, err := parsePrices(input, output, cacheRead, cacheWrite)
	if err != nil {
		return nil, err
	}

	var rawTiers []json.RawMessage
	if raw, exists := fields["tiers"]; exists && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if err := decodeOneJSON(raw, &rawTiers); err != nil {
			return nil, fmt.Errorf("decode cost tiers: %w", err)
		}
	}
	tiers := make([]pricing.ContextTier, 0, len(rawTiers))
	previousThreshold := int64(-1)
	for index, rawTier := range rawTiers {
		threshold, tierPrices, err := parseContextTier(rawTier)
		if err != nil {
			return nil, fmt.Errorf("tier %d: %w", index, err)
		}
		if threshold <= previousThreshold {
			return nil, fmt.Errorf("context tier thresholds must be strictly increasing")
		}
		if !hasSetPrice(tierPrices) {
			return nil, fmt.Errorf("tier %d must set at least one supported price", index)
		}
		tiers = append(tiers, pricing.ContextTier{InputThresholdTokens: threshold, Prices: tierPrices})
		previousThreshold = threshold
	}
	if !hasSetPrice(prices) && len(tiers) == 0 {
		return nil, nil
	}
	return &ModelCost{Prices: prices, ContextTiers: tiers}, nil
}

func parseContextTier(raw json.RawMessage) (int64, pricing.Prices, error) {
	fields, err := decodeCanonicalObject(raw, "context tier", tierCanonicalFields)
	if err != nil {
		return 0, pricing.Prices{}, err
	}
	tierRaw, exists := fields["tier"]
	if !exists || bytes.Equal(bytes.TrimSpace(tierRaw), []byte("null")) {
		return 0, pricing.Prices{}, fmt.Errorf("tier rule is required")
	}
	ruleFields, err := decodeCanonicalObject(tierRaw, "context tier rule", tierRuleCanonicalFields)
	if err != nil {
		return 0, pricing.Prices{}, err
	}
	tierType, err := decodeStringField(ruleFields, "type")
	if err != nil {
		return 0, pricing.Prices{}, err
	}
	if tierType != "context" {
		return 0, pricing.Prices{}, fmt.Errorf("unsupported type %q", tierType)
	}
	size, err := decodeNumberField(ruleFields, "size")
	if err != nil {
		return 0, pricing.Prices{}, err
	}
	if size == nil {
		return 0, pricing.Prices{}, fmt.Errorf("context size is required")
	}
	threshold, err := strconv.ParseInt(size.String(), 10, 64)
	if err != nil || threshold < 0 {
		return 0, pricing.Prices{}, fmt.Errorf("context size must be a non-negative integer")
	}
	input, output, cacheRead, cacheWrite, err := decodePriceFields(fields)
	if err != nil {
		return 0, pricing.Prices{}, err
	}
	prices, err := parsePrices(input, output, cacheRead, cacheWrite)
	if err != nil {
		return 0, pricing.Prices{}, err
	}
	return threshold, prices, nil
}

func decodePriceFields(fields map[string]json.RawMessage) (*json.Number, *json.Number, *json.Number, *json.Number, error) {
	input, err := decodeNumberField(fields, "input")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	output, err := decodeNumberField(fields, "output")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	cacheRead, err := decodeNumberField(fields, "cache_read")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	cacheWrite, err := decodeNumberField(fields, "cache_write")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return input, output, cacheRead, cacheWrite, nil
}

func parsePrices(input, output, cacheRead, cacheWrite *json.Number) (pricing.Prices, error) {
	values := [...]struct {
		name   string
		number *json.Number
		set    func(*pricing.Prices, pricing.Price)
	}{
		{name: "input", number: input, set: func(prices *pricing.Prices, price pricing.Price) { prices.Input = price }},
		{name: "output", number: output, set: func(prices *pricing.Prices, price pricing.Price) { prices.Output = price }},
		{name: "cache_read", number: cacheRead, set: func(prices *pricing.Prices, price pricing.Price) { prices.CacheRead = price }},
		{name: "cache_write", number: cacheWrite, set: func(prices *pricing.Prices, price pricing.Price) { prices.CacheWrite = price }},
	}
	var prices pricing.Prices
	for _, value := range values {
		if value.number == nil {
			continue
		}
		amount, err := parseJSONUSD(value.number.String())
		if err != nil {
			return pricing.Prices{}, fmt.Errorf("%s price: %w", value.name, err)
		}
		value.set(&prices, pricing.Price{NanoUSDPerMillion: amount, Set: true})
	}
	return prices, nil
}

// parseJSONUSD expands JSON exponent notation with string arithmetic before
// handing the plain fixed-point decimal to pricing.ParseUSD. No binary float is
// created. Long fixed-point values emitted by Models.dev are rounded to the
// nearest nano-USD only when they remain above zero after normalization.
func parseJSONUSD(value string) (pricing.NanoUSD, error) {
	if strings.HasPrefix(value, "-") {
		return 0, fmt.Errorf("USD price must be non-negative")
	}
	if normalized, ok := parseModelsDevFloatArtifact(value); ok {
		return normalized, nil
	}
	mantissa, exponentText, hasExponent := strings.Cut(value, "e")
	if !hasExponent {
		mantissa, exponentText, hasExponent = strings.Cut(value, "E")
	}
	if !hasExponent {
		return pricing.ParseUSD(value)
	}
	exponent, err := strconv.ParseInt(exponentText, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid USD exponent")
	}
	integer, fraction, hasFraction := strings.Cut(mantissa, ".")
	if integer == "" || (hasFraction && fraction == "") {
		return 0, fmt.Errorf("invalid USD decimal")
	}
	digits := integer + fraction
	leadingZeros := len(digits) - len(strings.TrimLeft(digits, "0"))
	if leadingZeros == len(digits) {
		return pricing.ParseUSD("0")
	}
	digits = digits[leadingZeros:]
	decimalPosition := int64(len(integer)-leadingZeros) + exponent
	digits = strings.TrimRight(digits, "0")
	if digits == "" {
		return pricing.ParseUSD("0")
	}

	var plain string
	switch {
	case decimalPosition <= 0:
		fractionalDigits := -decimalPosition + int64(len(digits))
		if fractionalDigits > 9 {
			return 0, fmt.Errorf("USD price has more than nine decimal places")
		}
		plain = "0." + strings.Repeat("0", int(-decimalPosition)) + digits
	case decimalPosition >= int64(len(digits)):
		integerDigits := decimalPosition
		if integerDigits > 32 {
			return 0, fmt.Errorf("USD price exceeds int64 nano USD")
		}
		plain = digits + strings.Repeat("0", int(decimalPosition)-len(digits))
	default:
		fractionalDigits := int64(len(digits)) - decimalPosition
		if fractionalDigits > 9 {
			return 0, fmt.Errorf("USD price has more than nine decimal places")
		}
		plain = digits[:decimalPosition] + "." + digits[decimalPosition:]
	}
	return pricing.ParseUSD(plain)
}

const modelsDevArtifactMinFractionDigits = 15

func parseModelsDevFloatArtifact(value string) (pricing.NanoUSD, bool) {
	_, fraction, hasFraction := strings.Cut(value, ".")
	if !hasFraction || strings.ContainsAny(fraction, "eE") ||
		len(fraction) < modelsDevArtifactMinFractionDigits {
		return 0, false
	}

	exact, ok := new(big.Rat).SetString(value)
	if !ok || exact.Sign() < 0 {
		return 0, false
	}
	scaled := new(big.Rat).Mul(exact, big.NewRat(1_000_000_000, 1))
	quotient, remainder := new(big.Int), new(big.Int)
	quotient.QuoRem(scaled.Num(), scaled.Denom(), remainder)
	if new(big.Int).Lsh(remainder, 1).Cmp(scaled.Denom()) >= 0 {
		quotient.Add(quotient, big.NewInt(1))
	}
	if quotient.Sign() == 0 && exact.Sign() != 0 || !quotient.IsInt64() {
		return 0, false
	}
	return pricing.NanoUSD(quotient.Int64()), true
}
