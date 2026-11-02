package control

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"

	"gpt-load/internal/catalog"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/pricing"
	"gpt-load/internal/storage/models"
)

func (s *Service) applyCatalogSnapshot(ctx context.Context, snapshot *catalog.Snapshot) error {
	if snapshot == nil {
		return fmt.Errorf("catalog snapshot is required: %w", app_errors.ErrInternalServer)
	}
	err := s.writePriceConfig(ctx, func(tx *gorm.DB, _ *catalog.Snapshot) (*pricing.Table, error) {
		if err := reconcileCatalogAutomaticPrices(tx, snapshot); err != nil {
			return nil, err
		}
		if err := reconcileReferencedPrices(tx, snapshot); err != nil {
			return nil, err
		}
		if err := cleanupUnreferencedAutomaticPrices(tx); err != nil {
			return nil, err
		}
		return loadPriceTable(ctx, tx)
	})
	if err != nil {
		return err
	}
	s.catalogRuntime.Publish(snapshot)
	logMissingAutomaticPricePriorityProviders(snapshot)
	return nil
}

func logMissingAutomaticPricePriorityProviders(snapshot *catalog.Snapshot) {
	if snapshot == nil {
		return
	}
	priority := catalog.AutomaticPriceProviderPriority()
	missing := make([]string, 0, len(priority))
	for _, providerID := range priority {
		if _, exists := snapshot.Providers[providerID]; !exists {
			missing = append(missing, providerID)
		}
	}
	if len(missing) == 0 {
		return
	}
	utils.LogPlaneBestEffort(
		logrus.StandardLogger(),
		logrus.WarnLevel,
		utils.LogPlaneControl,
		logrus.Fields{
			"event":                "models_dev_catalog_price_priority_missing",
			"missing_provider_ids": missing,
		},
		"Catalog missing automatic price priority providers",
	)
}

func reconcileCatalogAutomaticPrices(tx *gorm.DB, snapshot *catalog.Snapshot) error {
	references, err := loadReferencedPrices(tx)
	if err != nil {
		return err
	}
	var rows []models.ModelPrice
	if err := tx.Where("is_manual = ?", false).Order("id ASC").Find(&rows).Error; err != nil {
		return fmt.Errorf("load automatic model prices: %w", app_errors.ParseDBError(err))
	}
	for _, row := range rows {
		identity, err := PriceIdentityForChannelModel(row.ChannelID, row.ModelID)
		if err != nil {
			return fmt.Errorf("validate automatic model price identity: %w", app_errors.ErrInternalServer)
		}
		if _, referenced := references[identity]; !referenced {
			continue
		}
		match, ok := resolveAutomaticPriceForIdentity(snapshot, identity)
		desired := models.ModelPrice{}
		if ok {
			desired, err = automaticCatalogValues(match.cost)
			if err != nil {
				return fmt.Errorf("normalize catalog price: %w", app_errors.ErrInternalServer)
			}
		}
		if catalogPriceValuesEqual(row, desired) {
			continue
		}
		if err := tx.Model(&models.ModelPrice{}).
			Where("id = ? AND is_manual = ?", row.ID, false).
			Updates(map[string]any{
				"input_price_nano_usd_per_million_tokens":       desired.InputPriceNanoUSDPerMillionTokens,
				"output_price_nano_usd_per_million_tokens":      desired.OutputPriceNanoUSDPerMillionTokens,
				"cache_read_price_nano_usd_per_million_tokens":  desired.CacheReadPriceNanoUSDPerMillionTokens,
				"cache_write_price_nano_usd_per_million_tokens": desired.CacheWritePriceNanoUSDPerMillionTokens,
				"context_price_tiers":                           desired.ContextPriceTiers,
				"mode_price_schedules":                          desired.ModePriceSchedules,
			}).Error; err != nil {
			return fmt.Errorf("update automatic model price: %w", app_errors.ParseDBError(err))
		}
	}
	return nil
}

func automaticCatalogValues(cost *catalog.ModelCost) (models.ModelPrice, error) {
	row := models.ModelPrice{}
	if cost == nil {
		return row, nil
	}
	row.InputPriceNanoUSDPerMillionTokens = priceStoragePointer(cost.Prices.Input)
	row.OutputPriceNanoUSDPerMillionTokens = priceStoragePointer(cost.Prices.Output)
	row.CacheReadPriceNanoUSDPerMillionTokens = priceStoragePointer(cost.Prices.CacheRead)
	row.CacheWritePriceNanoUSDPerMillionTokens = priceStoragePointer(cost.Prices.CacheWrite)
	if len(cost.ContextTiers) > 0 {
		tiers := storageContextTiers(cost.ContextTiers)
		encoded, err := json.Marshal(tiers)
		if err != nil {
			return models.ModelPrice{}, err
		}
		row.ContextPriceTiers, err = models.NormalizeContextPriceTiers(models.JSON(encoded))
		if err != nil {
			return models.ModelPrice{}, err
		}
	}
	if len(cost.ModePrices) > 0 {
		modeSchedules := make(map[string]models.ModePriceSchedule, len(cost.ModePrices))
		for mode, prices := range cost.ModePrices {
			modeSchedules[string(mode)] = models.ModePriceSchedule{
				Prices: storagePriceSlots(prices),
			}
		}
		encoded, err := json.Marshal(modeSchedules)
		if err != nil {
			return models.ModelPrice{}, err
		}
		row.ModePriceSchedules, err = models.NormalizeModePriceSchedules(models.JSON(encoded))
		if err != nil {
			return models.ModelPrice{}, err
		}
	}
	return row, nil
}

func storagePriceSlots(prices pricing.Prices) models.PriceSlots {
	return models.PriceSlots{
		InputPriceNanoUSDPerMillionTokens:      priceStoragePointer(prices.Input),
		OutputPriceNanoUSDPerMillionTokens:     priceStoragePointer(prices.Output),
		CacheReadPriceNanoUSDPerMillionTokens:  priceStoragePointer(prices.CacheRead),
		CacheWritePriceNanoUSDPerMillionTokens: priceStoragePointer(prices.CacheWrite),
	}
}

func storageContextTiers(tiers []pricing.ContextTier) []models.ContextPriceTier {
	result := make([]models.ContextPriceTier, 0, len(tiers))
	for _, tier := range tiers {
		result = append(result, models.ContextPriceTier{
			ThresholdTokens:                        tier.InputThresholdTokens,
			InputPriceNanoUSDPerMillionTokens:      priceStoragePointer(tier.Prices.Input),
			OutputPriceNanoUSDPerMillionTokens:     priceStoragePointer(tier.Prices.Output),
			CacheReadPriceNanoUSDPerMillionTokens:  priceStoragePointer(tier.Prices.CacheRead),
			CacheWritePriceNanoUSDPerMillionTokens: priceStoragePointer(tier.Prices.CacheWrite),
		})
	}
	return result
}

func catalogPriceValuesEqual(left, right models.ModelPrice) bool {
	return pricePointerEqual(left.InputPriceNanoUSDPerMillionTokens, right.InputPriceNanoUSDPerMillionTokens) &&
		pricePointerEqual(left.OutputPriceNanoUSDPerMillionTokens, right.OutputPriceNanoUSDPerMillionTokens) &&
		pricePointerEqual(left.CacheReadPriceNanoUSDPerMillionTokens, right.CacheReadPriceNanoUSDPerMillionTokens) &&
		pricePointerEqual(left.CacheWritePriceNanoUSDPerMillionTokens, right.CacheWritePriceNanoUSDPerMillionTokens) &&
		reflect.DeepEqual([]byte(left.ContextPriceTiers), []byte(right.ContextPriceTiers)) &&
		reflect.DeepEqual([]byte(left.ModePriceSchedules), []byte(right.ModePriceSchedules))
}

func pricePointerEqual(left, right *int64) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}

func cleanupUnreferencedAutomaticPrices(tx *gorm.DB) error {
	references, err := loadReferencedPrices(tx)
	if err != nil {
		return err
	}
	var rows []models.ModelPrice
	if err := tx.Where("is_manual = ?", false).Order("id ASC").Find(&rows).Error; err != nil {
		return fmt.Errorf("load automatic prices for cleanup: %w", app_errors.ParseDBError(err))
	}
	ids := make([]uint, 0)
	for _, row := range rows {
		identity, err := PriceIdentityForChannelModel(row.ChannelID, row.ModelID)
		if err != nil {
			return fmt.Errorf("validate automatic price cleanup identity: %w", app_errors.ErrInternalServer)
		}
		if _, referenced := references[identity]; !referenced {
			ids = append(ids, row.ID)
		}
	}
	if len(ids) == 0 {
		return nil
	}
	if err := tx.Where("is_manual = ? AND id IN ?", false, ids).Delete(&models.ModelPrice{}).Error; err != nil {
		return fmt.Errorf("remove unreferenced automatic prices: %w", app_errors.ParseDBError(err))
	}
	return nil
}
