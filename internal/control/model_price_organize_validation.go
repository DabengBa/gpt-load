package control

import (
	"encoding/json"
	"fmt"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/pricing"
	"gpt-load/internal/storage/models"
)

func modelPriceUpdateAllNull(request ModelPriceUpdateRequest) bool {
	if len(request.ModeSchedules.schedules) > 0 {
		return false
	}
	if request.Input.nanoUSD != nil ||
		request.Output.nanoUSD != nil ||
		request.CacheRead.nanoUSD != nil ||
		request.CacheWrite.nanoUSD != nil {
		return false
	}
	for _, tier := range request.ContextTiers.tiers {
		if tier.Input.nanoUSD != nil ||
			tier.Output.nanoUSD != nil ||
			tier.CacheRead.nanoUSD != nil ||
			tier.CacheWrite.nanoUSD != nil {
			return false
		}
	}
	return true
}

func buildModePriceSchedulesJSON(
	schedules map[pricing.Mode]ModelPriceScheduleRequest,
) (models.JSON, error) {
	if len(schedules) == 0 {
		return nil, nil
	}
	encoded := make(map[string]models.ModePriceSchedule, len(schedules))
	for mode, schedule := range schedules {
		encoded[string(mode)] = models.ModePriceSchedule{
			Prices: models.PriceSlots{
				InputPriceNanoUSDPerMillionTokens:      cloneModelPriceValue(schedule.Prices.Input.nanoUSD),
				OutputPriceNanoUSDPerMillionTokens:     cloneModelPriceValue(schedule.Prices.Output.nanoUSD),
				CacheReadPriceNanoUSDPerMillionTokens:  cloneModelPriceValue(schedule.Prices.CacheRead.nanoUSD),
				CacheWritePriceNanoUSDPerMillionTokens: cloneModelPriceValue(schedule.Prices.CacheWrite.nanoUSD),
			},
			ContextPriceTiers: contextPriceTierModels(schedule.ContextTiers.tiers),
		}
	}
	raw, err := json.Marshal(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode mode price schedules: %w", app_errors.ErrInternalServer)
	}
	normalized, err := models.NormalizeModePriceSchedules(models.JSON(raw))
	if err != nil {
		return nil, fmt.Errorf("validate mode price schedules: %w", app_errors.ErrValidation)
	}
	return normalized, nil
}

func validateModeScheduleUpdate(
	row models.ModelPrice,
	requested map[pricing.Mode]ModelPriceScheduleRequest,
) error {
	rule, err := persistedPriceRule(row)
	if err != nil {
		return fmt.Errorf("decode persisted model price: %w", app_errors.ErrInternalServer)
	}
	// Ultrafast 允许在目录尚无价格时手工配置；其他模式保留原有增删约束。
	for mode := range rule.ModeSchedules {
		if _, exists := requested[mode]; !exists && mode != pricing.ModeUltrafast {
			return fmt.Errorf("model price mode schedule set cannot be changed: %w", app_errors.ErrValidation)
		}
	}
	for mode := range requested {
		if _, exists := rule.ModeSchedules[mode]; !exists && mode != pricing.ModeUltrafast {
			return fmt.Errorf("model price mode schedule set cannot be changed: %w", app_errors.ErrValidation)
		}
	}
	return nil
}

func cloneModelPriceValue(value *int64) *int64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

// buildContextPriceTiersJSON converts the submitted tier list to the
// persisted JSON shape, validating it through the single authoritative rule
// set (models.NormalizeContextPriceTiers) before any transaction opens. This
// keeps a malformed submission a plain validation error instead of a GORM
// hook failure surfacing through ParseDBError.
func buildContextPriceTiersJSON(tiers []ModelPriceContextTierRequest) (models.JSON, error) {
	if len(tiers) == 0 {
		return nil, nil
	}
	encoded := contextPriceTierModels(tiers)
	raw, err := json.Marshal(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode context price tiers: %w", app_errors.ErrInternalServer)
	}
	normalized, err := models.NormalizeContextPriceTiers(models.JSON(raw))
	if err != nil {
		return nil, fmt.Errorf("validate context price tiers: %w", app_errors.ErrValidation)
	}
	return normalized, nil
}

func contextPriceTierModels(tiers []ModelPriceContextTierRequest) []models.ContextPriceTier {
	encoded := make([]models.ContextPriceTier, 0, len(tiers))
	for _, tier := range tiers {
		encoded = append(encoded, models.ContextPriceTier{
			ThresholdTokens:                        tier.ThresholdTokens.tokens,
			InputPriceNanoUSDPerMillionTokens:      cloneModelPriceValue(tier.Input.nanoUSD),
			OutputPriceNanoUSDPerMillionTokens:     cloneModelPriceValue(tier.Output.nanoUSD),
			CacheReadPriceNanoUSDPerMillionTokens:  cloneModelPriceValue(tier.CacheRead.nanoUSD),
			CacheWritePriceNanoUSDPerMillionTokens: cloneModelPriceValue(tier.CacheWrite.nanoUSD),
		})
	}
	return encoded
}
