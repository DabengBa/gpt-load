package control

import (
	"gpt-load/internal/pricing"
)

type PriceSlotsDTO struct {
	Input      *string `json:"input"`
	Output     *string `json:"output"`
	CacheRead  *string `json:"cache_read"`
	CacheWrite *string `json:"cache_write"`
}

type ModelPriceDTO struct {
	ID                  uint                                   `json:"id"`
	ChannelID           string                                 `json:"channel_id"`
	ChannelName         string                                 `json:"channel_name"`
	ChannelMark         string                                 `json:"channel_mark"`
	ChannelIcon         string                                 `json:"channel_icon"`
	ModelID             string                                 `json:"model_id"`
	Prices              PriceSlotsDTO                          `json:"prices"`
	ModeSchedules       map[pricing.Mode]ModelPriceScheduleDTO `json:"mode_schedules"`
	PricingStatus       PricingStatus                          `json:"pricing_status"`
	Method              *string                                `json:"method"`
	MatchedProviderID   *string                                `json:"matched_provider_id"`
	MatchSource         *ModelPriceMatchSource                 `json:"match_source"`
	Referenced          bool                                   `json:"referenced"`
	ReferenceCount      int                                    `json:"reference_count"`
	ReferenceGroupCount int                                    `json:"reference_group_count"`
	ContextTiers        []ModelPriceContextTierDTO             `json:"context_tiers"`
	UpdatedAtMS         int64                                  `json:"updated_at_ms"`
	CanReset            bool                                   `json:"can_reset"`
	CanDelete           bool                                   `json:"can_delete"`
}

type ModelPriceScheduleDTO struct {
	Prices       PriceSlotsDTO              `json:"prices"`
	ContextTiers []ModelPriceContextTierDTO `json:"context_tiers"`
}

// ModelPriceContextTierDTO is one compiled input-quantity price tier. Unlike
// the request shape, prices nest under "prices" to mirror the top-level DTO.
type ModelPriceContextTierDTO struct {
	ThresholdTokens int64         `json:"threshold_tokens"`
	Prices          PriceSlotsDTO `json:"prices"`
}

type ModelPricePaginationDTO struct {
	Page       int64 `json:"page"`
	PageSize   int64 `json:"page_size"`
	TotalItems int64 `json:"total_items"`
	TotalPages int64 `json:"total_pages"`
}

type ModelPriceListResponse struct {
	Items      []ModelPriceDTO         `json:"items"`
	Pagination ModelPricePaginationDTO `json:"pagination"`
}

type modelPriceListRecord struct {
	dto ModelPriceDTO
}

type ModelPriceIDData struct {
	ID uint `json:"id"`
}

type ModelPriceReferenceData struct {
	ID                  uint `json:"id"`
	ReferenceCount      int  `json:"reference_count"`
	ReferenceGroupCount int  `json:"reference_group_count"`
}
