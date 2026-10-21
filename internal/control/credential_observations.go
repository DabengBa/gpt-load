package control

import (
	"context"
	"gpt-load/internal/requestlog"
	"time"
)

const credentialDailyUsageWindow = 24 * time.Hour

func (s *Service) enrichCredentialActivities(
	ctx context.Context,
	items []CredentialItemResponse,
) {
	if s == nil || s.credentialActivity == nil || len(items) == 0 {
		return
	}
	credentialIDs := make([]uint, 0, len(items))
	for index := range items {
		if items[index].CredentialID != 0 {
			credentialIDs = append(credentialIDs, items[index].CredentialID)
		}
	}
	if len(credentialIDs) == 0 {
		return
	}
	now := s.now().UTC()
	fromMS := now.Add(-credentialDailyUsageWindow).UnixMilli()
	toMS := now.UnixMilli()
	if fromMS < 0 || toMS <= fromMS {
		return
	}
	activities, err := s.credentialActivity.QueryCredentialActivity(
		ctx,
		requestlog.CredentialActivityQuery{
			CredentialIDs: credentialIDs,
			FromMS:        fromMS,
			ToMS:          toMS,
		},
	)
	if err != nil {
		return
	}
	for index := range items {
		activity, exists := activities[items[index].CredentialID]
		if !exists {
			continue
		}
		items[index].LastFailureSummary = activity.LastFailureSummary
	}
}
