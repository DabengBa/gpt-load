package catalog

import (
	"fmt"
	"strings"
	"unicode"

	"gpt-load/internal/pricing"
)

func validateSnapshot(snapshot *Snapshot) error {
	if snapshot == nil {
		return fmt.Errorf("catalog snapshot is required")
	}
	for providerKey, provider := range snapshot.Providers {
		if providerKey != provider.ID {
			return fmt.Errorf("provider %q map key and id must match", providerKey)
		}
		if _, err := pricing.ProviderScopeKey(provider.ID); err != nil {
			return fmt.Errorf("provider %q: %w", providerKey, err)
		}
		if strings.TrimSpace(provider.Name) == "" {
			return fmt.Errorf("provider %q name is required", providerKey)
		}
		if provider.Models == nil {
			return fmt.Errorf("provider %q models map is required", providerKey)
		}
		for modelKey, model := range provider.Models {
			if modelKey != model.ID {
				return fmt.Errorf("provider %q model %q map key and id must match", providerKey, modelKey)
			}
			if err := validateModelID(model.ID); err != nil {
				return fmt.Errorf("provider %q model %q: %w", providerKey, modelKey, err)
			}
			if strings.TrimSpace(model.Name) == "" {
				return fmt.Errorf("provider %q model %q name is required", providerKey, modelKey)
			}
			if err := validateModelCost(model.Cost); err != nil {
				return fmt.Errorf("provider %q model %q: %w", providerKey, modelKey, err)
			}
		}
	}
	return nil
}

func validateModelID(id string) error {
	if len(id) == 0 || len(id) > maxModelIDBytes {
		return fmt.Errorf("model id must be 1 through %d bytes", maxModelIDBytes)
	}
	if strings.TrimSpace(id) != id {
		return fmt.Errorf("model id must not have surrounding whitespace")
	}
	for _, character := range id {
		if unicode.IsControl(character) {
			return fmt.Errorf("model id must not contain control characters")
		}
	}
	return nil
}

func validateModelCost(cost *ModelCost) error {
	if cost == nil {
		return nil
	}
	for _, price := range [...]pricing.Price{cost.Prices.Input, cost.Prices.Output, cost.Prices.CacheRead, cost.Prices.CacheWrite} {
		if price.NanoUSDPerMillion < 0 {
			return fmt.Errorf("price must be non-negative")
		}
	}
	previousThreshold := int64(-1)
	for _, tier := range cost.ContextTiers {
		if tier.InputThresholdTokens < 0 || tier.InputThresholdTokens <= previousThreshold {
			return fmt.Errorf("context tier thresholds must be non-negative and strictly increasing")
		}
		if !hasSetPrice(tier.Prices) {
			return fmt.Errorf("context tier must set at least one price")
		}
		for _, price := range [...]pricing.Price{tier.Prices.Input, tier.Prices.Output, tier.Prices.CacheRead, tier.Prices.CacheWrite} {
			if price.NanoUSDPerMillion < 0 {
				return fmt.Errorf("context tier price must be non-negative")
			}
		}
		previousThreshold = tier.InputThresholdTokens
	}
	for mode, prices := range cost.ModePrices {
		if !mode.Valid() || mode == pricing.ModeStandard {
			return fmt.Errorf("invalid non-standard pricing mode %q", mode)
		}
		if !hasSetPrice(prices) {
			return fmt.Errorf("pricing mode %q must set at least one price", mode)
		}
		for _, price := range [...]pricing.Price{prices.Input, prices.Output, prices.CacheRead, prices.CacheWrite} {
			if price.NanoUSDPerMillion < 0 {
				return fmt.Errorf("pricing mode %q price must be non-negative", mode)
			}
		}
	}
	return nil
}

func hasSetPrice(prices pricing.Prices) bool {
	return prices.Input.Set || prices.Output.Set || prices.CacheRead.Set || prices.CacheWrite.Set
}
