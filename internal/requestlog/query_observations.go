package requestlog

import (
	"context"
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/health"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/storage/models"
)

func (service *Service) loadAccessKeyRefs(ctx context.Context, records []Record) error {
	return loadAccessKeyRefs(ctx, service.db, records)
}

func (service *Service) loadFinalExecutionObservations(
	ctx context.Context,
	records []Record,
) error {
	if len(records) == 0 {
		return nil
	}
	requestIDs := make([]string, 0, len(records))
	recordIndexes := make(map[string]int, len(records))
	for index, record := range records {
		requestIDs = append(requestIDs, record.RequestID)
		recordIndexes[record.RequestID] = index
	}

	var attempts []models.RequestLogAttempt
	if err := service.db.WithContext(ctx).
		Select(
			"request_id", "sequence", "group_id", "channel_id", "credential_id", "route_mode",
			"upstream_protocol", "upstream_model", "pricing_receipt", "feedback_status", "feedback_reason",
			"provider_first_response_ms", "provider_tokens_per_second", "duration_ms",
		).
		Where("request_id IN ?", requestIDs).
		Order("request_id ASC").
		Order("sequence DESC").
		Find(&attempts).Error; err != nil {
		return fmt.Errorf("query request log final execution observations: %w", err)
	}
	executionResolved := make(map[string]struct{}, len(records))
	feedbackResolved := make(map[string]struct{}, len(records))
	pricingResolved := make(map[string]struct{}, len(records))
	for _, attempt := range attempts {
		index, ok := recordIndexes[attempt.RequestID]
		if !ok {
			continue
		}
		if _, ok := feedbackResolved[attempt.RequestID]; !ok {
			feedbackResolved[attempt.RequestID] = struct{}{}
			feedback, err := decodeAttemptFeedback(attempt)
			if err != nil {
				return fmt.Errorf("query request log final execution observations: %w", err)
			}
			records[index].FinalAttemptFeedback = feedback
			records[index].FinalAttemptDurationMs = new(attempt.DurationMs)
		}
		record := records[index]
		if record.GroupID != attempt.GroupID ||
			record.ChannelID != channel.ID(attempt.ChannelID) ||
			record.CredentialID != attempt.CredentialID {
			continue
		}
		if _, ok := executionResolved[attempt.RequestID]; !ok {
			executionResolved[attempt.RequestID] = struct{}{}
			mode := channel.RouteMode(attempt.RouteMode)
			if mode != "" && !mode.Valid() {
				return fmt.Errorf("query request log final execution observations: invalid route mode")
			}
			records[index].RouteMode = mode
			upstreamProtocol := protocol.Protocol(attempt.UpstreamProtocol)
			if upstreamProtocol != "" && !upstreamProtocol.Valid() {
				return fmt.Errorf("query request log final execution observations: invalid upstream protocol")
			}
			records[index].UpstreamProtocol = upstreamProtocol
		}
		if _, ok := pricingResolved[attempt.RequestID]; ok {
			continue
		}
		receipt, err := decodeAttemptPricingReceipt(attempt)
		if err != nil {
			return fmt.Errorf("query request log final execution observations: %w", err)
		}
		if receipt != nil {
			pricingResolved[attempt.RequestID] = struct{}{}
			records[index].PricingMode = pricingModeForReceipt(receipt)
			records[index].ContextThresholdTokens = contextThresholdForReceipt(receipt)
		}
	}
	return nil
}

func finalRouteMode(record Record, attempts []Attempt) channel.RouteMode {
	for index := len(attempts) - 1; index >= 0; index-- {
		attempt := attempts[index]
		if attempt.GroupID == record.GroupID &&
			attempt.ChannelID == record.ChannelID &&
			attempt.CredentialID == record.CredentialID {
			return attempt.RouteMode
		}

	}
	return ""
}

func finalAttemptFeedback(attempts []Attempt) health.Feedback {
	if len(attempts) == 0 {
		return health.Feedback{}
	}
	return attempts[len(attempts)-1].Feedback
}

func finalUpstreamProtocol(record Record, attempts []Attempt) protocol.Protocol {
	for index := len(attempts) - 1; index >= 0; index-- {
		attempt := attempts[index]
		if attempt.GroupID == record.GroupID &&
			attempt.ChannelID == record.ChannelID &&
			attempt.CredentialID == record.CredentialID {
			return attempt.UpstreamProtocol
		}
	}
	return ""
}

func finalPricingReceipt(record Record, attempts []Attempt) *pricing.Receipt {
	for index := len(attempts) - 1; index >= 0; index-- {
		attempt := attempts[index]
		if attempt.GroupID == record.GroupID &&
			attempt.ChannelID == record.ChannelID &&
			attempt.CredentialID == record.CredentialID &&
			attempt.PricingReceipt != nil {
			return attempt.PricingReceipt
		}
	}
	return nil
}

func pricingModeForReceipt(receipt *pricing.Receipt) pricing.Mode {
	if receipt == nil {
		return ""
	}

	return receipt.PricingMode
}

func contextThresholdForReceipt(receipt *pricing.Receipt) *int64 {
	if receipt == nil || receipt.ContextThresholdTokens == nil {
		return nil
	}
	value := *receipt.ContextThresholdTokens
	return &value
}

func loadAccessKeyRefs(ctx context.Context, db *gorm.DB, records []Record) error {
	if len(records) == 0 {
		return nil
	}

	ids := make([]uint, 0, len(records))
	seen := make(map[uint]struct{}, len(records))
	for _, record := range records {
		if record.AccessKey.ID == 0 {
			continue
		}
		if _, ok := seen[record.AccessKey.ID]; ok {
			continue
		}
		seen[record.AccessKey.ID] = struct{}{}
		ids = append(ids, record.AccessKey.ID)
	}
	if len(ids) == 0 {
		return nil
	}

	var accessKeys []struct {
		ID   uint
		Name string
	}
	if err := db.WithContext(ctx).
		Model(&models.AccessKey{}).
		Select("id", "name").
		Where("id IN ?", ids).
		Find(&accessKeys).Error; err != nil {
		return fmt.Errorf("query request log access keys: %w", err)
	}
	refs := make(map[uint]AccessKeyRef, len(accessKeys))
	for _, accessKey := range accessKeys {
		name := accessKey.Name
		refs[accessKey.ID] = AccessKeyRef{
			ID:      accessKey.ID,
			Name:    &name,
			Deleted: false,
		}
	}
	for index := range records {
		if ref, ok := refs[records[index].AccessKey.ID]; ok {
			records[index].AccessKey = ref
		}
	}
	return nil
}
