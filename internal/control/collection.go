package control

import (
	"net/url"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	app_errors "gpt-load/internal/platform/errors"
)

// Shared primitives for collection read modules. Each resource keeps its own
// record capture, match/sort/summary declarations; this module owns the
// mechanics that were previously copied per resource: the strict query-string
// contract, positive-int parsing, pagination math, and Unicode-fold matching.
// The read itself must stay inside a withReadSnapshot transaction (ADR-0001).

// parseCollectionQueryValues enforces the shared collection query contract:
// only allowlisted keys, each with exactly one value.
func parseCollectionQueryValues(
	rawQuery string,
	forceQuery bool,
	allowed ...string,
) (url.Values, *app_errors.APIError) {
	if forceQuery && rawQuery == "" {
		return nil, app_errors.ErrBadRequest
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return nil, app_errors.ErrBadRequest
	}
	for key, entries := range values {
		allowedKey := false
		for _, name := range allowed {
			if key == name {
				allowedKey = true
				break
			}
		}
		if !allowedKey || len(entries) != 1 {
			return nil, app_errors.ErrBadRequest
		}
	}
	return values, nil
}

// collectionQueryText reads a bounded, trimmed text parameter.
func collectionQueryText(values url.Values, key string, maxRunes int) (string, bool) {
	entries, exists := values[key]
	if !exists {
		return "", true
	}
	text := strings.TrimSpace(entries[0])
	if utf8.RuneCountInString(text) > maxRunes {
		return "", false
	}
	return text, true
}

// parseCollectionPositiveInt64 parses a strict positive integer: digits only,
// no leading zero, value > 0.
func parseCollectionPositiveInt64(value string) (int64, bool) {
	if value == "" || value[0] == '0' {
		return 0, false
	}
	for index := range len(value) {
		if value[index] < '0' || value[index] > '9' {
			return 0, false
		}
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil || parsed <= 0 {
		return 0, false
	}
	return parsed, true
}

func collectionTotalPages(totalItems, pageSize int64) int64 {
	if totalItems == 0 || pageSize <= 0 {
		return 0
	}
	pages := totalItems / pageSize
	if totalItems%pageSize != 0 {
		pages++
	}
	return pages
}

// collectionPageItems returns the [offset,end) page of records projected to
// items. Out-of-range pages yield an empty slice, matching the per-resource
// copies this replaces.
func collectionPageItems[R, I any](records []R, page, pageSize int64, item func(R) I) []I {
	if page <= 0 || pageSize <= 0 {
		return []I{}
	}
	itemCount := int64(len(records))
	if page-1 > itemCount/pageSize {
		return []I{}
	}
	offset := (page - 1) * pageSize
	if offset >= itemCount {
		return []I{}
	}
	end := offset + pageSize
	if end < offset || end > itemCount {
		end = itemCount
	}
	items := make([]I, end-offset)
	for index, record := range records[offset:end] {
		items[index] = item(record)
	}
	return items
}

func collectionContainsFold(value, query string) bool {
	return strings.Contains(collectionFold(value), collectionFold(query))
}

func collectionFold(value string) string {
	var folded strings.Builder
	folded.Grow(len(value))
	for _, runeValue := range value {
		folded.WriteRune(collectionFoldRune(runeValue))
	}
	return folded.String()
}

func collectionFoldRune(value rune) rune {
	folded := value
	for candidate := unicode.SimpleFold(value); candidate != value; candidate = unicode.SimpleFold(candidate) {
		if candidate < folded {
			folded = candidate
		}
	}
	return folded
}
