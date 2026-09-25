package control

import (
	"sort"

	"gpt-load/internal/state"
)

const (
	defaultAccessKeyCollectionPage     int64 = 1
	defaultAccessKeyCollectionPageSize int64 = 20
)

type AccessKeyCollectionQuery struct {
	Query    string
	Status   *state.AccessKeyStatus
	Page     int64
	PageSize int64
}

func queryAccessKeyCollectionRecords(
	records []accessKeyCollectionRecord,
	query AccessKeyCollectionQuery,
) AccessKeyCollectionResponse {
	summary := summarizeAccessKeyCollectionRecords(records)
	filtered := make([]accessKeyCollectionRecord, 0, len(records))
	for _, record := range records {
		if matchesAccessKeyCollectionQuery(record, query) {
			filtered = append(filtered, record)
		}
	}
	sortAccessKeyCollectionRecords(filtered)

	totalItems := int64(len(filtered))
	return AccessKeyCollectionResponse{
		Summary: summary,
		Items: collectionPageItems(
			filtered, query.Page, query.PageSize,
			func(record accessKeyCollectionRecord) AccessKeyCollectionItem {
				return record.AccessKeyCollectionItem
			},
		),
		Pagination: AccessKeyCollectionPagination{
			Page:       query.Page,
			PageSize:   query.PageSize,
			TotalItems: totalItems,
			TotalPages: collectionTotalPages(totalItems, query.PageSize),
		},
	}
}

func normalizeAccessKeyCollectionQuery(query AccessKeyCollectionQuery) AccessKeyCollectionQuery {
	if query.Page <= 0 {
		query.Page = defaultAccessKeyCollectionPage
	}
	if query.PageSize <= 0 {
		query.PageSize = defaultAccessKeyCollectionPageSize
	}
	return query
}

func summarizeAccessKeyCollectionRecords(
	records []accessKeyCollectionRecord,
) AccessKeyCollectionSummary {
	summary := AccessKeyCollectionSummary{Total: int64(len(records))}
	for _, record := range records {
		switch record.Status {
		case state.AccessKeyStatusActive:
			summary.Active++
		case state.AccessKeyStatusDisabled:
			summary.Disabled++
		}
	}
	return summary
}

func matchesAccessKeyCollectionQuery(
	record accessKeyCollectionRecord,
	query AccessKeyCollectionQuery,
) bool {
	if query.Status != nil && record.Status != *query.Status {
		return false
	}
	return query.Query == "" ||
		collectionContainsFold(record.Name, query.Query) ||
		collectionContainsFold(record.MaskedKey, query.Query)
}

func sortAccessKeyCollectionRecords(records []accessKeyCollectionRecord) {
	sort.Slice(records, func(leftIndex, rightIndex int) bool {
		left, right := records[leftIndex], records[rightIndex]
		if left.UpdatedAtMS != right.UpdatedAtMS {
			return left.UpdatedAtMS > right.UpdatedAtMS
		}
		return left.ID > right.ID
	})
}
