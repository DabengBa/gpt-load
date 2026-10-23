package gateway

import (
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

var reasonScheduleCapacity = reason{Status: 503, Code: "schedule_capacity_exhausted", Message: "All eligible route entries are at capacity."}

func capacityKey(selection scheduler.Selection) state.RouteEntryKey {
	return state.RouteEntryKey{GroupID: selection.GroupID, EntryID: selection.EntryID}
}

// A fixed continuation identity alone never grants affinity overflow. Match
// the actual resolved binding and recheck its live identity and health.
func (h *Handler) acquireCapacity(snapshot *state.ConfigSnapshot, selection scheduler.Selection, ref state.CredentialRef, affinity requestAffinity) (*state.CapacityReservation, state.CapacityAdmission) {
	overflow := affinity.boundTarget.Valid() && affinity.preferredCredentialID == ref.ID &&
		affinity.preferredIdentityGeneration == ref.IdentityGeneration &&
		affinityTargetForSelection(selection, ref) == affinity.boundTarget
	if overflow {
		current, exists := h.registry.CredentialRef(ref.ID)
		_, active := h.registry.ActiveEncryptedCredentialDataIfMatch(ref)
		overflow = exists && active && sameWebsocketIdentity(current, ref)
		candidates := h.registry.CollectCredentialCandidates([]uint{selection.GroupID}, nil, h.now())
		overflow = overflow && len(candidates) == 1 && candidates[0].ID == ref.ID && candidates[0].IdentityGeneration == ref.IdentityGeneration
		if source, ok := h.registry.(state.EntryRuntimeSource); ok {
			entry, _ := source.EntryRuntime(capacityKey(selection), h.now())
			overflow = overflow && entry.RuntimeState(h.now()) == state.EntryRuntimeAvailable
		}
	}
	return h.manager.AcquireCapacityForSnapshot(snapshot, capacityKey(selection), overflow)
}

func recordCapacityBlocked(recorder *requestRecorder) {
	if recorder != nil {
		recorder.capacityBlocked = true
	}
	if recorder != nil && len(recorder.attempts) > 0 {
		recorder.outcome.errorSummary += " [schedule_capacity_exhausted: remaining eligible route entries are at capacity]"
	}
}
