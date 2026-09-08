// Route entry semantics shared by snapshot compilation and the management
// plane. See docs/design/model-route-entries.md §2-§3.

package state

import (
	"fmt"
	"regexp"
	"strings"
)

var entryIDPattern = regexp.MustCompile(`^e[0-9a-f]{12}$`)

type EntryCircuitBreaker struct {
	BlacklistThreshold *int `json:"blacklist_threshold,omitempty"`
	CooldownSeconds    *int `json:"cooldown_seconds,omitempty"`
}

func cloneEntryCircuitBreaker(value *EntryCircuitBreaker) *EntryCircuitBreaker {
	if value == nil {
		return nil
	}
	result := &EntryCircuitBreaker{}
	if value.BlacklistThreshold != nil {
		v := *value.BlacklistThreshold
		result.BlacklistThreshold = &v
	}
	if value.CooldownSeconds != nil {
		v := *value.CooldownSeconds
		result.CooldownSeconds = &v
	}
	return result
}

// ExternalModelName returns the client-facing model name of a route entry:
// the alias when it is set, otherwise the upstream model ID (design V4).
// All external name derivation must go through this helper so alias and
// upstream naming stay consistent across the gateway, catalog, and API.
func ExternalModelName(upstreamID, alias string) string {
	if name := strings.TrimSpace(alias); name != "" {
		return name
	}
	return strings.TrimSpace(upstreamID)
}

// ValidateModelRouteEntries enforces the route entry validation rules of the
// model-route-entries design (§3) on one group's entries:
//
//	V1 the (external name, upstream model) pairs are unique;
//	V2 the entries of every external name keep a positive total weight;
//	V3 upstream model IDs are non-empty;
//	V4 external names derive through ExternalModelName.
//
// weight defaults to 1 and priority to 1 when unset; weight 0 keeps an entry
// but excludes it from traffic splitting; negative weights, weights above
// MaxWeight, and priorities below 1 are rejected. The subject labels the
// owning group so errors can locate the offending group and model.
func ValidateModelRouteEntries(subject string, models []ModelConfig) error {
	type routeEntryKey struct {
		external string
		upstream string
	}
	seenEntries := make(map[routeEntryKey]struct{}, len(models))
	weightSums := make(map[string]int, len(models))
	externalOrder := make([]string, 0, len(models))
	for index, model := range models {
		upstream := strings.TrimSpace(model.ID)
		if upstream == "" {
			return fmt.Errorf("%s model entry %d: model id is required", subject, index)
		}
		if model.Weight != nil && (*model.Weight < 0 || *model.Weight > MaxWeight) {
			return fmt.Errorf("%s model %q: weight must be between 0 and %d", subject, upstream, MaxWeight)
		}
		if model.Priority != nil && *model.Priority < 1 {
			return fmt.Errorf("%s model %q: priority must be at least 1", subject, upstream)
		}
		if model.EntryID != "" && !entryIDPattern.MatchString(model.EntryID) {
			return fmt.Errorf("%s model %q: entry_id has invalid format", subject, upstream)
		}
		if model.CircuitBreaker != nil {
			if model.CircuitBreaker.BlacklistThreshold != nil && *model.CircuitBreaker.BlacklistThreshold < 1 {
				return fmt.Errorf("%s model %q: blacklist_threshold must be at least 1", subject, upstream)
			}
			if model.CircuitBreaker.CooldownSeconds != nil && *model.CircuitBreaker.CooldownSeconds < 0 {
				return fmt.Errorf("%s model %q: cooldown_seconds must not be negative", subject, upstream)
			}
		}
		external := ExternalModelName(upstream, model.Alias)
		entry := routeEntryKey{external: external, upstream: upstream}
		if _, duplicate := seenEntries[entry]; duplicate {
			return fmt.Errorf(
				"%s has duplicate route entry for external model %q and upstream model %q",
				subject, external, upstream,
			)
		}
		seenEntries[entry] = struct{}{}
		if model.EntryID != "" {
			for _, other := range models[:index] {
				if other.EntryID == model.EntryID {
					return fmt.Errorf("%s has duplicate entry_id %q", subject, model.EntryID)
				}
			}
		}
		weight := 1
		if model.Weight != nil {
			weight = *model.Weight
		}
		if _, exists := weightSums[external]; !exists {
			externalOrder = append(externalOrder, external)
		}
		weightSums[external] += weight
	}
	// Report V2 failures in entry order so repeated saves stay deterministic.
	for _, external := range externalOrder {
		if weightSums[external] <= 0 {
			return fmt.Errorf(
				"%s external model %q entry weights must sum to a positive value",
				subject, external,
			)
		}
	}
	return nil
}
