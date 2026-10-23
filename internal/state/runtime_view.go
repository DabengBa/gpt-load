package state

import (
	"sort"
	"time"
)

type CredentialRuntimeView struct {
	ID                 uint
	GroupID            uint
	Version            uint64
	IdentityGeneration uint64
	CooldownUntil      time.Time
	Blacklisted        bool
	BlacklistReleaseAt time.Time
	FailureCount       int
	BillingStreak      int
}

type CredentialRuntimeState string

const (
	CredentialRuntimeAvailable   CredentialRuntimeState = "available"
	CredentialRuntimeBlacklisted CredentialRuntimeState = "blacklisted"
	CredentialRuntimeCooldown    CredentialRuntimeState = "cooldown"
)

func (view CredentialRuntimeView) RuntimeState(now time.Time) CredentialRuntimeState {
	if view.Blacklisted {
		return CredentialRuntimeBlacklisted
	}
	if view.CooldownUntil.After(now) {
		return CredentialRuntimeCooldown
	}
	return CredentialRuntimeAvailable
}

func runtimeView(entry *CredentialEntry) CredentialRuntimeView {
	return CredentialRuntimeView{
		ID:                 entry.ID,
		GroupID:            entry.GroupID,
		Version:            entry.Version,
		IdentityGeneration: entry.IdentityGeneration,
		CooldownUntil:      entry.CooldownUntil,
		Blacklisted:        entry.Blacklisted,
		BlacklistReleaseAt: entry.BlacklistReleaseAt,
		FailureCount:       entry.FailureCount,
		BillingStreak:      entry.BillingStreak,
	}
}

func sortRuntimeViews(views []CredentialRuntimeView) {
	sort.Slice(views, func(i, j int) bool {
		if views[i].GroupID != views[j].GroupID {
			return views[i].GroupID < views[j].GroupID
		}
		return views[i].ID < views[j].ID
	})
}

func (r *CredentialRegistry) Snapshot() []CredentialRuntimeView {
	r.mu.RLock()
	views := make([]CredentialRuntimeView, 0, len(r.credentialGroups))
	for _, bucket := range r.buckets {
		for _, entry := range bucket {
			views = append(views, runtimeView(entry))
		}
	}
	r.mu.RUnlock()
	sortRuntimeViews(views)
	return views
}
