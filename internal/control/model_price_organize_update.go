package control

import (
	"bytes"
	"context"
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/catalog"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/pricing"
	"gpt-load/internal/storage/models"
)

func (s *Service) UpdateModelPrice(
	ctx context.Context,
	id uint,
	request ModelPriceUpdateRequest,
) (ModelPriceDTO, error) {
	if id == 0 || uint64(id) > uint64(maxSafeInteger) {
		return ModelPriceDTO{}, app_errors.ErrBadRequest
	}
	if err := request.validate(); err != nil {
		return ModelPriceDTO{}, err
	}
	contextTiers, err := buildContextPriceTiersJSON(request.ContextTiers.tiers)
	if err != nil {
		return ModelPriceDTO{}, err
	}
	modeSchedules, err := buildModePriceSchedulesJSON(request.ModeSchedules.schedules)
	if err != nil {
		return ModelPriceDTO{}, err
	}
	var result ModelPriceDTO
	err = s.writePriceConfig(ctx, func(tx *gorm.DB, catalogSnapshot *catalog.Snapshot) (*pricing.Table, error) {
		var row models.ModelPrice
		if err := tx.First(&row, id).Error; err != nil {
			return nil, fmt.Errorf("load model price: %w", app_errors.ParseDBError(err))
		}
		if err := validateModeScheduleUpdate(row, request.ModeSchedules.schedules); err != nil {
			return nil, err
		}
		references, err := loadPriceReferenceSnapshot(tx)
		if err != nil {
			return nil, err
		}
		if modelPriceUpdateAllNull(request) && !request.ConfirmUnpriced {
			return nil, app_errors.NewAPIErrorWithData(
				app_errors.ErrModelPriceUnpricedConfirmationRequired,
				ModelPriceIDData{ID: id},
			)
		}

		desired := row
		desired.InputPriceNanoUSDPerMillionTokens = cloneModelPriceValue(request.Input.nanoUSD)
		desired.OutputPriceNanoUSDPerMillionTokens = cloneModelPriceValue(request.Output.nanoUSD)
		desired.CacheReadPriceNanoUSDPerMillionTokens = cloneModelPriceValue(request.CacheRead.nanoUSD)
		desired.CacheWritePriceNanoUSDPerMillionTokens = cloneModelPriceValue(request.CacheWrite.nanoUSD)
		desired.ContextPriceTiers = contextTiers
		desired.ModePriceSchedules = modeSchedules
		desired.IsManual = true
		if !modelPriceMutableValuesEqual(row, desired) {
			updatedAtMS, err := safeEpochMilliseconds(s.now())
			if err != nil {
				return nil, fmt.Errorf("timestamp model price update: %w", app_errors.ErrInternalServer)
			}
			if err := tx.Model(&models.ModelPrice{}).
				Where("id = ?", id).
				Updates(map[string]any{
					"input_price_nano_usd_per_million_tokens":       desired.InputPriceNanoUSDPerMillionTokens,
					"output_price_nano_usd_per_million_tokens":      desired.OutputPriceNanoUSDPerMillionTokens,
					"cache_read_price_nano_usd_per_million_tokens":  desired.CacheReadPriceNanoUSDPerMillionTokens,
					"cache_write_price_nano_usd_per_million_tokens": desired.CacheWritePriceNanoUSDPerMillionTokens,
					"context_price_tiers":                           desired.ContextPriceTiers,
					"mode_price_schedules":                          desired.ModePriceSchedules,
					"is_manual":                                     true,
					"updated_at_ms":                                 updatedAtMS,
				}).Error; err != nil {
				return nil, fmt.Errorf("update model price: %w", app_errors.ParseDBError(err))
			}
			desired.UpdatedAtMS = updatedAtMS
			row = desired
		}

		table, err := loadPriceTable(ctx, tx)
		if err != nil {
			return nil, err
		}
		record, err := projectModelPriceRow(row, references, catalogSnapshot)
		if err != nil {
			return nil, err
		}
		result = record.dto
		return table, nil
	})
	if err != nil {
		return ModelPriceDTO{}, err
	}
	return result, nil
}

func (s *Service) ResetModelPrice(
	ctx context.Context,
	id uint,
) (ModelPriceDTO, error) {
	if id == 0 || uint64(id) > uint64(maxSafeInteger) {
		return ModelPriceDTO{}, app_errors.ErrBadRequest
	}

	var result ModelPriceDTO
	err := s.writePriceConfig(ctx, func(tx *gorm.DB, catalogSnapshot *catalog.Snapshot) (*pricing.Table, error) {
		var row models.ModelPrice
		if err := tx.First(&row, id).Error; err != nil {
			return nil, fmt.Errorf("load model price: %w", app_errors.ParseDBError(err))
		}
		references, err := loadPriceReferenceSnapshot(tx)
		if err != nil {
			return nil, err
		}
		identity, err := PriceIdentityForChannelModel(row.ChannelID, row.ModelID)
		if err != nil {
			return nil, fmt.Errorf("validate model price identity: %w", app_errors.ErrInternalServer)
		}
		desired, err := resetModelPriceValues(identity, catalogSnapshot)
		if err != nil {
			return nil, fmt.Errorf("normalize catalog model price: %w", app_errors.ErrInternalServer)
		}
		if !modelPriceMutableValuesEqual(row, desired) {
			updatedAtMS, err := safeEpochMilliseconds(s.now())
			if err != nil {
				return nil, fmt.Errorf("timestamp model price reset: %w", app_errors.ErrInternalServer)
			}
			if err := tx.Model(&models.ModelPrice{}).
				Where("id = ?", id).
				Updates(map[string]any{
					"input_price_nano_usd_per_million_tokens":       desired.InputPriceNanoUSDPerMillionTokens,
					"output_price_nano_usd_per_million_tokens":      desired.OutputPriceNanoUSDPerMillionTokens,
					"cache_read_price_nano_usd_per_million_tokens":  desired.CacheReadPriceNanoUSDPerMillionTokens,
					"cache_write_price_nano_usd_per_million_tokens": desired.CacheWritePriceNanoUSDPerMillionTokens,
					"context_price_tiers":                           desired.ContextPriceTiers,
					"mode_price_schedules":                          desired.ModePriceSchedules,
					"is_manual":                                     false,
					"updated_at_ms":                                 updatedAtMS,
				}).Error; err != nil {
				return nil, fmt.Errorf("reset model price: %w", app_errors.ParseDBError(err))
			}
			row.InputPriceNanoUSDPerMillionTokens = desired.InputPriceNanoUSDPerMillionTokens
			row.OutputPriceNanoUSDPerMillionTokens = desired.OutputPriceNanoUSDPerMillionTokens
			row.CacheReadPriceNanoUSDPerMillionTokens = desired.CacheReadPriceNanoUSDPerMillionTokens
			row.CacheWritePriceNanoUSDPerMillionTokens = desired.CacheWritePriceNanoUSDPerMillionTokens
			row.ContextPriceTiers = desired.ContextPriceTiers
			row.ModePriceSchedules = desired.ModePriceSchedules
			row.IsManual = false
			row.UpdatedAtMS = updatedAtMS
		}

		table, err := loadPriceTable(ctx, tx)
		if err != nil {
			return nil, err
		}
		record, err := projectModelPriceRow(row, references, catalogSnapshot)
		if err != nil {
			return nil, err
		}
		result = record.dto
		return table, nil
	})
	if err != nil {
		return ModelPriceDTO{}, err
	}
	return result, nil
}

func resetModelPriceValues(identity pricing.Identity, snapshot *catalog.Snapshot) (models.ModelPrice, error) {
	match, ok := resolveAutomaticPriceForIdentity(snapshot, identity)
	if !ok {
		return models.ModelPrice{}, nil
	}
	return automaticCatalogValues(match.cost)
}

func (s *Service) DeleteModelPrice(ctx context.Context, id uint) error {
	if id == 0 || uint64(id) > uint64(maxSafeInteger) {
		return app_errors.ErrBadRequest
	}
	return s.writePriceConfig(ctx, func(tx *gorm.DB, _ *catalog.Snapshot) (*pricing.Table, error) {
		var row models.ModelPrice
		if err := tx.First(&row, id).Error; err != nil {
			return nil, fmt.Errorf("load model price: %w", app_errors.ParseDBError(err))
		}
		identity, err := PriceIdentityForChannelModel(row.ChannelID, row.ModelID)
		if err != nil {
			return nil, fmt.Errorf("validate model price identity: %w", app_errors.ErrInternalServer)
		}
		references, err := loadPriceReferenceSnapshot(tx)
		if err != nil {
			return nil, err
		}
		reference := references.references[identity]
		if reference.referenceCount > 0 {
			return nil, app_errors.NewAPIErrorWithData(
				app_errors.ErrModelPriceReferenced,
				ModelPriceReferenceData{
					ID: id, ReferenceCount: reference.referenceCount,
					ReferenceGroupCount: reference.referenceGroupCount(),
				},
			)
		}
		if !row.IsManual {
			return nil, app_errors.NewAPIErrorWithData(
				app_errors.ErrModelPriceAutomaticDeleteForbidden,
				ModelPriceIDData{ID: id},
			)
		}
		if err := tx.Where("id = ?", id).Delete(&models.ModelPrice{}).Error; err != nil {
			return nil, fmt.Errorf("delete model price: %w", app_errors.ParseDBError(err))
		}
		return loadPriceTable(ctx, tx)
	})
}

func modelPriceMutableValuesEqual(left, right models.ModelPrice) bool {
	leftTiers, leftErr := models.NormalizeContextPriceTiers(left.ContextPriceTiers)
	rightTiers, rightErr := models.NormalizeContextPriceTiers(right.ContextPriceTiers)
	if leftErr != nil || rightErr != nil {
		return false
	}
	leftSchedules, leftScheduleErr := models.NormalizeModePriceSchedules(left.ModePriceSchedules)
	rightSchedules, rightScheduleErr := models.NormalizeModePriceSchedules(right.ModePriceSchedules)
	if leftScheduleErr != nil || rightScheduleErr != nil {
		return false
	}
	return pricePointerEqual(left.InputPriceNanoUSDPerMillionTokens, right.InputPriceNanoUSDPerMillionTokens) &&
		pricePointerEqual(left.OutputPriceNanoUSDPerMillionTokens, right.OutputPriceNanoUSDPerMillionTokens) &&
		pricePointerEqual(left.CacheReadPriceNanoUSDPerMillionTokens, right.CacheReadPriceNanoUSDPerMillionTokens) &&
		pricePointerEqual(left.CacheWritePriceNanoUSDPerMillionTokens, right.CacheWritePriceNanoUSDPerMillionTokens) &&
		bytes.Equal(leftTiers, rightTiers) &&
		bytes.Equal(leftSchedules, rightSchedules) &&
		left.IsManual == right.IsManual
}
