package accessquota

import (
	"fmt"
	"slices"
	"time"
)

// This file owns the read side of the runtime: live decisions and views, and the
// durable snapshot round trip that checkpoint persistence drives.

// Check reports the admission decision for one key without opening a window.
func (runtime *Runtime) Check(accessKeyID uint, now time.Time) Decision {
	entry := runtime.lockEntry(accessKeyID)
	if entry == nil {
		return allowedDecision()
	}
	defer entry.mu.Unlock()
	return decisionLocked(entry, now.UnixMilli())
}

// Snapshot renders one key's rules and the aggregate decision at now.
func (runtime *Runtime) Snapshot(accessKeyID uint, now time.Time) View {
	view := View{ObservedAtMS: now.UnixMilli(), Allowed: true, Recoverable: true, Rules: []RuleView{}}
	entry := runtime.lockEntry(accessKeyID)
	if entry == nil {
		return view
	}
	defer entry.mu.Unlock()
	view.Rules = make([]RuleView, 0, len(entry.rules))
	for _, rule := range entry.rules {
		view.Rules = append(view.Rules, ruleViewLocked(rule, view.ObservedAtMS))
	}
	decision := decisionLocked(entry, view.ObservedAtMS)
	view.Allowed = decision.Allowed
	view.Recoverable = decision.Recoverable
	view.NextAvailableAtMS = cloneInt64(decision.NextAvailableAtMS)
	return view
}

// DirtySnapshots collects the states awaiting persistence, oldest access key ID
// first, and stops after limit states unless limit is zero.
func (runtime *Runtime) DirtySnapshots(limit int) []RestoredState {
	if runtime == nil || limit == 0 {
		return nil
	}
	runtime.mu.RLock()
	ids := make([]uint, 0, len(runtime.entries))
	for id := range runtime.entries {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	runtime.mu.RUnlock()

	result := make([]RestoredState, 0)
	for _, id := range ids {
		entry := runtime.lockEntry(id)
		if entry == nil {
			continue
		}
		for _, rule := range entry.rules {
			if !rule.dirty {
				continue
			}
			result = append(result, restoredState(id, rule))
			if limit > 0 && len(result) >= limit {
				entry.mu.Unlock()
				return result
			}
		}
		entry.mu.Unlock()
	}
	return result
}

// Ack clears the dirty flag only for the snapshot version that was persisted.
func (runtime *Runtime) Ack(accessKeyID, ruleID uint, revision, snapshotVersion uint64) {
	entry := runtime.lockEntry(accessKeyID)
	if entry == nil {
		return
	}
	defer entry.mu.Unlock()
	rule := entry.byID[ruleID]
	if rule == nil || rule.definition.Revision != revision || rule.snapshotVersion != snapshotVersion {
		return
	}
	rule.dirty = false
}

func decisionLocked(entry *accessKeyEntry, nowMS int64) Decision {
	decision := allowedDecision()
	var next int64
	for _, rule := range entry.rules {
		view := ruleViewLocked(rule, nowMS)
		if view.Status != RuleStatusExhausted {
			continue
		}
		decision.Allowed = false
		decision.BlockingRules = append(decision.BlockingRules, view)
		if rule.definition.Kind == KindTotal {
			decision.Recoverable = false
			decision.NextAvailableAtMS = nil
			continue
		}
		if decision.Recoverable && view.WindowEndsAtMS != nil && *view.WindowEndsAtMS > next {
			next = *view.WindowEndsAtMS
		}
	}
	if !decision.Allowed && decision.Recoverable && next > 0 {
		decision.NextAvailableAtMS = &next
	}
	return decision
}

func allowedDecision() Decision {
	return Decision{Allowed: true, Recoverable: true, BlockingRules: []RuleView{}}
}

func ruleViewLocked(rule *runtimeRule, nowMS int64) RuleView {
	view := RuleView{
		Rule: rule.definition, RemainingNanoUSD: rule.definition.LimitNanoUSD,
		Status: RuleStatusAvailable, WindowGeneration: rule.windowGeneration,
	}
	if rule.definition.Kind == KindPeriodic && periodicInactive(rule, nowMS) {
		view.Status = RuleStatusInactive
		return view
	}
	view.UsedNanoUSD = rule.usedNanoUSD
	view.RemainingNanoUSD = remaining(rule.definition.LimitNanoUSD, rule.usedNanoUSD)
	view.WindowStartedAtMS = cloneInt64(rule.windowStartedAtMS)
	view.WindowEndsAtMS = cloneInt64(rule.windowEndsAtMS)
	if rule.usedNanoUSD >= rule.definition.LimitNanoUSD {
		view.Status = RuleStatusExhausted
	}
	return view
}

func periodicInactive(rule *runtimeRule, nowMS int64) bool {
	return rule.windowStartedAtMS == nil || rule.windowEndsAtMS == nil || nowMS >= *rule.windowEndsAtMS
}

func remaining(limit, used int64) int64 {
	if used >= limit {
		return 0
	}
	return limit - used
}

func restoredState(accessKeyID uint, rule *runtimeRule) RestoredState {
	return RestoredState{
		AccessKeyID: accessKeyID, RuleID: rule.definition.ID, RuleRevision: rule.definition.Revision,
		UsedNanoUSD: rule.usedNanoUSD, WindowStartedAtMS: cloneInt64(rule.windowStartedAtMS),
		WindowEndsAtMS: cloneInt64(rule.windowEndsAtMS), WindowGeneration: rule.windowGeneration,
		SnapshotVersion: rule.snapshotVersion,
	}
}

func cloneRestoredState(state RestoredState) RestoredState {
	state.WindowStartedAtMS = cloneInt64(state.WindowStartedAtMS)
	state.WindowEndsAtMS = cloneInt64(state.WindowEndsAtMS)
	return state
}

func cloneInt64(value *int64) *int64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func entriesFromRestore(
	definitions map[uint][]Rule,
	restored map[uint]RestoredState,
) (map[uint]*accessKeyEntry, error) {
	remaining := make(map[uint]RestoredState, len(restored))
	for id, state := range restored {
		remaining[id] = state
	}
	entries := make(map[uint]*accessKeyEntry, len(definitions))
	for accessKeyID, rules := range definitions {
		entry := newAccessKeyEntry(rules)
		for _, rule := range entry.rules {
			state, exists := remaining[rule.definition.ID]
			if !exists {
				return nil, fmt.Errorf("restore access quota rule %d for key %d: state is missing", rule.definition.ID, accessKeyID)
			}
			if state.AccessKeyID != accessKeyID || state.RuleRevision != rule.definition.Revision {
				return nil, fmt.Errorf("restore access quota rule %d: state identity or revision mismatch", rule.definition.ID)
			}
			if err := applyRestoredState(rule, state); err != nil {
				return nil, err
			}
			delete(remaining, rule.definition.ID)
		}
		entries[accessKeyID] = entry
	}
	if len(remaining) != 0 {
		ids := make([]uint, 0, len(remaining))
		for id := range remaining {
			ids = append(ids, id)
		}
		slices.Sort(ids)
		return nil, fmt.Errorf("restore access quota runtime: orphan state for rule %d", ids[0])
	}
	return entries, nil
}

func validateRestoredState(state RestoredState) error {
	if state.AccessKeyID == 0 || state.RuleID == 0 || state.RuleRevision == 0 ||
		state.UsedNanoUSD < 0 || state.SnapshotVersion == 0 {
		return fmt.Errorf("restore access quota rule %d: invalid persisted state", state.RuleID)
	}
	if (state.WindowStartedAtMS == nil) != (state.WindowEndsAtMS == nil) {
		return fmt.Errorf("restore access quota rule %d: incomplete window", state.RuleID)
	}
	if state.WindowStartedAtMS != nil &&
		(*state.WindowStartedAtMS < 0 || *state.WindowEndsAtMS <= *state.WindowStartedAtMS) {
		return fmt.Errorf("restore access quota rule %d: invalid window", state.RuleID)
	}
	return nil
}

func applyRestoredState(rule *runtimeRule, state RestoredState) error {
	if rule.definition.Kind == KindTotal {
		if state.WindowStartedAtMS != nil || state.WindowEndsAtMS != nil || state.WindowGeneration != 0 {
			return fmt.Errorf("restore total access quota rule %d: window state is not allowed", rule.definition.ID)
		}
	} else if state.WindowStartedAtMS == nil {
		if state.UsedNanoUSD != 0 || state.WindowGeneration != 0 {
			return fmt.Errorf("restore periodic access quota rule %d: inactive state has usage or generation", rule.definition.ID)
		}
	} else {
		if state.WindowGeneration == 0 {
			return fmt.Errorf("restore periodic access quota rule %d: active state has no generation", rule.definition.ID)
		}
		windowDurationMS := *state.WindowEndsAtMS - *state.WindowStartedAtMS
		if windowDurationMS != rule.definition.PeriodSeconds*int64(time.Second/time.Millisecond) {
			return fmt.Errorf("restore periodic access quota rule %d: window duration does not match period", rule.definition.ID)
		}
	}
	rule.usedNanoUSD = state.UsedNanoUSD
	rule.windowStartedAtMS = cloneInt64(state.WindowStartedAtMS)
	rule.windowEndsAtMS = cloneInt64(state.WindowEndsAtMS)
	rule.windowGeneration = state.WindowGeneration
	rule.snapshotVersion = state.SnapshotVersion
	return nil
}
