package accessquota

import (
	"math"
	"time"
)

// This file owns the write side of the runtime: claiming a cost limit for an
// attempt and settling the estimated cost it produced.

// Admit claims one attempt against every rule of the key. It rejects an
// exhausted key and, when allowed, rolls every expired periodic window forward
// to the current attempt. The returned ticket pins the rule revisions and window
// generations observed here so a later Complete can discard stale costs.
func (runtime *Runtime) Admit(accessKeyID uint, now time.Time) (Ticket, Decision) {
	entry := runtime.lockEntry(accessKeyID)
	if entry == nil {
		return Ticket{AccessKeyID: accessKeyID}, allowedDecision()
	}
	nowMS := now.UnixMilli()
	decision := decisionLocked(entry, nowMS)
	if !decision.Allowed {
		entry.mu.Unlock()
		return Ticket{AccessKeyID: accessKeyID}, decision
	}

	dirty := false
	for _, rule := range entry.rules {
		if rule.definition.Kind != KindPeriodic || !periodicInactive(rule, nowMS) {
			continue
		}
		startedAt := nowMS
		endsAt := now.Add(time.Duration(rule.definition.PeriodSeconds) * time.Second).UnixMilli()
		rule.windowStartedAtMS = &startedAt
		rule.windowEndsAtMS = &endsAt
		rule.usedNanoUSD = 0
		if rule.windowGeneration < math.MaxUint64 {
			rule.windowGeneration++
		}
		advanceVersion(rule)
		dirty = true
	}

	ticket := Ticket{AccessKeyID: accessKeyID, Rules: make([]TicketRule, 0, len(entry.rules))}
	for _, rule := range entry.rules {
		ticket.Rules = append(ticket.Rules, TicketRule{
			RuleID: rule.definition.ID, RuleRevision: rule.definition.Revision,
			WindowGeneration: rule.windowGeneration,
		})
	}
	entry.mu.Unlock()
	if dirty {
		runtime.notifyDirty()
	}
	return ticket, allowedDecision()
}

// Complete charges the estimated cost of a finished attempt. Costs from tickets
// whose rule revision or periodic window generation is stale are dropped, and an
// invalid or overflowing cost saturates the rule instead of wrapping.
func (runtime *Runtime) Complete(ticket Ticket, costNanoUSD int64) CompletionResult {
	if runtime == nil || ticket.AccessKeyID == 0 || len(ticket.Rules) == 0 {
		return CompletionResult{}
	}
	entry := runtime.lockEntry(ticket.AccessKeyID)
	if entry == nil {
		return CompletionResult{}
	}

	completion := CompletionResult{}
	dirty := false
	for _, ticketRule := range ticket.Rules {
		rule := entry.byID[ticketRule.RuleID]
		if rule == nil || rule.definition.Revision != ticketRule.RuleRevision {
			continue
		}
		if rule.definition.Kind == KindPeriodic && rule.windowGeneration != ticketRule.WindowGeneration {
			continue
		}
		used := rule.usedNanoUSD
		switch {
		case costNanoUSD < 0:
			used = math.MaxInt64
			completion.Fault = CompletionFaultNegativeEstimate
		case costNanoUSD == 0:
			continue
		case used > math.MaxInt64-costNanoUSD:
			used = math.MaxInt64
			if completion.Fault == "" {
				completion.Fault = CompletionFaultOverflow
			}
		default:
			used += costNanoUSD
		}
		if used == rule.usedNanoUSD {
			continue
		}
		rule.usedNanoUSD = used
		advanceVersion(rule)
		dirty = true
	}
	entry.mu.Unlock()
	if completion.Fault != "" {
		runtime.overflowFaultTotal.Add(1)
	}
	if dirty {
		runtime.notifyDirty()
	}
	return completion
}

// cloneRuntimeState carries live accounting state across a reconcile that keeps
// the same rule identity and revision.
func cloneRuntimeState(target, source *runtimeRule) {
	target.usedNanoUSD = source.usedNanoUSD
	target.windowStartedAtMS = cloneInt64(source.windowStartedAtMS)
	target.windowEndsAtMS = cloneInt64(source.windowEndsAtMS)
	target.windowGeneration = source.windowGeneration
	target.snapshotVersion = source.snapshotVersion
	target.dirty = source.dirty
}

// advanceVersion records an unpersisted accounting change. The version is
// monotonic per rule so a checkpoint can acknowledge exactly one snapshot.
func advanceVersion(rule *runtimeRule) {
	if rule.snapshotVersion < math.MaxUint64 {
		rule.snapshotVersion++
	}
	rule.dirty = true
}
