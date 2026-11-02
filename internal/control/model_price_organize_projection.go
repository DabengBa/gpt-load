package control

import (
	"gpt-load/internal/pricing"
	"gpt-load/internal/storage/models"
)

func modelPriceMethod(
	row models.ModelPrice,
	configured bool,
	matchedAutomaticPrice bool,
) *string {
	if !row.IsManual {
		if !configured || !matchedAutomaticPrice {
			return nil
		}
		method := "auto_sync"
		return &method
	}

	method := "user_set"
	if !configured {
		method = "user_marked_unpriced"
	}
	return &method
}

func modelPriceWireDecimal(value *int64) *string {
	if value == nil {
		return nil
	}
	formatted := pricing.FormatUSD(pricing.NanoUSD(*value))
	return &formatted
}

func modelPriceWireDecimalFromPrice(price pricing.Price) *string {
	if !price.Set {
		return nil
	}
	formatted := pricing.FormatUSD(price.NanoUSDPerMillion)
	return &formatted
}

func projectPriceSlots(prices pricing.Prices) PriceSlotsDTO {
	return PriceSlotsDTO{
		Input:      modelPriceWireDecimalFromPrice(prices.Input),
		Output:     modelPriceWireDecimalFromPrice(prices.Output),
		CacheRead:  modelPriceWireDecimalFromPrice(prices.CacheRead),
		CacheWrite: modelPriceWireDecimalFromPrice(prices.CacheWrite),
	}
}

func projectModeSchedules(schedules map[pricing.Mode]pricing.Schedule) map[pricing.Mode]ModelPriceScheduleDTO {
	result := make(map[pricing.Mode]ModelPriceScheduleDTO, len(schedules))
	for mode, schedule := range schedules {
		result[mode] = ModelPriceScheduleDTO{
			Prices:       projectPriceSlots(schedule.Prices),
			ContextTiers: projectContextPriceTiers(schedule.ContextTiers),
		}
	}
	return result
}

func projectContextPriceTiers(tiers []pricing.ContextTier) []ModelPriceContextTierDTO {
	result := make([]ModelPriceContextTierDTO, 0, len(tiers))
	for _, tier := range tiers {
		result = append(result, ModelPriceContextTierDTO{
			ThresholdTokens: tier.InputThresholdTokens,
			Prices:          projectPriceSlots(tier.Prices),
		})
	}
	return result
}
