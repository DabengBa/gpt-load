package control

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/catalog"
	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/pricing"
	"gpt-load/internal/storage/models"
)

// writePriceConfig runs one model-price write under the ordinary write path:
// serialized lock → operation-recovery barrier → catalog snapshot load →
// transaction → price-table publication. Unlike writeGroupConfig there is no
// inline runtime recovery: a failed publish leaves no persisted divergence
// (the table is rebuilt from DB by the next price/config write or reconcile).
func (s *Service) writePriceConfig(
	ctx context.Context,
	mutate func(tx *gorm.DB, catalogSnapshot *catalog.Snapshot) (*pricing.Table, error),
) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.enforceOperationRecoveryBarrierLocked(ctx, 0); err != nil {
		return err
	}
	var catalogSnapshot *catalog.Snapshot
	if s.catalogRuntime != nil {
		catalogSnapshot = s.catalogRuntime.Load()
	}
	var table *pricing.Table
	err := s.withControlTransaction(ctx, func(tx *gorm.DB) error {
		loaded, err := mutate(tx, catalogSnapshot)
		if err != nil {
			return err
		}
		table = loaded
		return nil
	})
	if err != nil {
		return err
	}
	s.priceRuntime.Publish(table)
	return nil
}

func (s *Service) ListModelPrices(
	ctx context.Context,
	query ModelPriceListQuery,
) (ModelPriceListResponse, error) {
	if err := validateModelPriceListQuery(query); err != nil {
		return ModelPriceListResponse{}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}

	s.writeMu.RLock()
	defer s.writeMu.RUnlock()
	var catalogSnapshot *catalog.Snapshot
	if s.catalogRuntime != nil {
		catalogSnapshot = s.catalogRuntime.Load()
	}
	var rows []models.ModelPrice
	var references priceReferenceSnapshot
	if err := s.withReadSnapshot(ctx, func(tx *gorm.DB) error {
		var err error
		references, err = loadPriceReferenceSnapshot(tx)
		if err != nil {
			return err
		}
		if err := tx.Order("id ASC").Find(&rows).Error; err != nil {
			return fmt.Errorf("load model prices: %w", app_errors.ParseDBError(err))
		}
		return nil
	}); err != nil {
		return ModelPriceListResponse{}, err
	}

	records := make([]modelPriceListRecord, 0, len(rows))
	for _, row := range rows {
		record, err := projectModelPriceRow(row, references, catalogSnapshot)
		if err != nil {
			return ModelPriceListResponse{}, err
		}
		if !modelPriceRecordMatches(record, query) {
			continue
		}
		records = append(records, record)
	}
	sortModelPriceRecords(records)

	totalItems := int64(len(records))
	return ModelPriceListResponse{
		Items: modelPricePageItems(records, query.Page, query.PageSize),
		Pagination: ModelPricePaginationDTO{
			Page: query.Page, PageSize: query.PageSize, TotalItems: totalItems,
			TotalPages: modelPriceTotalPages(totalItems, query.PageSize),
		},
	}, nil
}

func validateModelPriceListQuery(query ModelPriceListQuery) error {
	switch query.Usage {
	case ModelPriceUsageInUse, ModelPriceUsageUnreferenced, ModelPriceUsageAll:
	default:
		return app_errors.ErrValidation
	}
	switch query.Status {
	case ModelPriceStatusPending, ModelPriceStatusConfigured, ModelPriceStatusAll:
	default:
		return app_errors.ErrValidation
	}
	if query.Page <= 0 || query.Page > maxSafeInteger ||
		query.PageSize <= 0 || query.PageSize > maxModelPriceListPageSize ||
		len([]rune(query.Search)) > maxModelPriceSearchRunes {
		return app_errors.ErrValidation
	}
	return nil
}

func projectModelPriceRow(
	row models.ModelPrice,
	references priceReferenceSnapshot,
	catalogSnapshot *catalog.Snapshot,
) (modelPriceListRecord, error) {
	if row.ID == 0 || uint64(row.ID) > uint64(maxSafeInteger) || validateSafeMilliseconds(row.UpdatedAtMS) != nil {
		return modelPriceListRecord{}, fmt.Errorf("invalid persisted model price wire identity: %w", app_errors.ErrInternalServer)
	}
	rule, err := persistedPriceRule(row)
	if err != nil {
		return modelPriceListRecord{}, fmt.Errorf("decode persisted model price: %w", app_errors.ErrInternalServer)
	}
	if _, err := pricing.NewTable([]pricing.Rule{rule}); err != nil {
		return modelPriceListRecord{}, fmt.Errorf("validate persisted model price: %w", app_errors.ErrInternalServer)
	}
	identity := rule.Identity
	descriptor, ok := modelPriceChannelRegistry.Get(channel.ID(identity.ChannelID))
	if !ok || strings.TrimSpace(descriptor.Name) == "" {
		return modelPriceListRecord{}, fmt.Errorf("resolve model price channel: %w", app_errors.ErrInternalServer)
	}
	reference := references.references[identity]
	status := resolvePricingStatus(&row)
	prices := PriceSlotsDTO{
		Input:      modelPriceWireDecimal(row.InputPriceNanoUSDPerMillionTokens),
		Output:     modelPriceWireDecimal(row.OutputPriceNanoUSDPerMillionTokens),
		CacheRead:  modelPriceWireDecimal(row.CacheReadPriceNanoUSDPerMillionTokens),
		CacheWrite: modelPriceWireDecimal(row.CacheWritePriceNanoUSDPerMillionTokens),
	}
	configured := modelPriceHasConfiguredValue(row)
	var matchedProviderID *string
	var matchSource *ModelPriceMatchSource
	matchedAutomaticPrice := false
	if !row.IsManual {
		match, matched := resolveAutomaticPriceForIdentity(catalogSnapshot, identity)
		matchedAutomaticPrice = matched
		if configured && matchedAutomaticPrice {
			providerID := match.providerID
			matchedProviderID = &providerID
			source := match.source
			matchSource = &source
		} else {
			matchedAutomaticPrice = false
		}
	}
	dto := ModelPriceDTO{
		ID: row.ID, ChannelID: row.ChannelID, ChannelName: descriptor.Name,
		ChannelMark: descriptor.Mark, ChannelIcon: descriptor.Icon, ModelID: row.ModelID,
		Prices:              prices,
		ModeSchedules:       projectModeSchedules(rule.ModeSchedules),
		PricingStatus:       status,
		Method:              modelPriceMethod(row, configured, matchedAutomaticPrice),
		MatchedProviderID:   matchedProviderID,
		MatchSource:         matchSource,
		Referenced:          reference.referenceCount > 0,
		ReferenceCount:      reference.referenceCount,
		ReferenceGroupCount: reference.referenceGroupCount(),
		ContextTiers:        projectContextPriceTiers(rule.ContextTiers),
		UpdatedAtMS:         row.UpdatedAtMS,
		CanReset:            row.IsManual,
		CanDelete:           row.IsManual && reference.referenceCount == 0,
	}
	return modelPriceListRecord{dto: dto}, nil
}

func modelPriceRecordMatches(record modelPriceListRecord, query ModelPriceListQuery) bool {
	switch query.Usage {
	case ModelPriceUsageInUse:
		if !record.dto.Referenced {
			return false
		}
	case ModelPriceUsageUnreferenced:
		if record.dto.Referenced {
			return false
		}
	}
	if query.Status != ModelPriceStatusAll && string(record.dto.PricingStatus) != string(query.Status) {
		return false
	}
	return query.Search == "" ||
		collectionContainsFold(record.dto.ModelID, query.Search) ||
		collectionContainsFold(record.dto.ChannelID, query.Search)
}

func sortModelPriceRecords(records []modelPriceListRecord) {
	sort.Slice(records, func(leftIndex, rightIndex int) bool {
		left, right := records[leftIndex], records[rightIndex]
		if left.dto.PricingStatus != right.dto.PricingStatus {
			return left.dto.PricingStatus == PricingStatusPending
		}
		if left.dto.Referenced != right.dto.Referenced {
			return left.dto.Referenced
		}
		if left.dto.ModelID != right.dto.ModelID {
			return left.dto.ModelID < right.dto.ModelID
		}
		if left.dto.ChannelID != right.dto.ChannelID {
			return left.dto.ChannelID < right.dto.ChannelID
		}
		return left.dto.ID < right.dto.ID
	})
}

func modelPriceTotalPages(totalItems, pageSize int64) int64 {
	if totalItems == 0 || pageSize <= 0 {
		return 0
	}
	pages := totalItems / pageSize
	if totalItems%pageSize != 0 {
		pages++
	}
	return pages
}

func modelPricePageItems(
	records []modelPriceListRecord,
	page, pageSize int64,
) []ModelPriceDTO {
	items := []ModelPriceDTO{}
	itemCount := int64(len(records))
	if page <= 0 || pageSize <= 0 || page-1 > itemCount/pageSize {
		return items
	}
	offset := (page - 1) * pageSize
	if offset >= itemCount {
		return items
	}
	end := offset + pageSize
	if end < offset || end > itemCount {
		end = itemCount
	}
	items = make([]ModelPriceDTO, 0, end-offset)
	for _, record := range records[offset:end] {
		items = append(items, record.dto)
	}
	return items
}
