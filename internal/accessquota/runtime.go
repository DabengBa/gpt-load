// Package accessquota owns AccessKey estimated-cost limit runtime state.
package accessquota

import (
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

const (
	MinPeriodSeconds int64 = 60
	MaxPeriodSeconds int64 = 365 * 24 * 60 * 60
	MaxPeriodicRules       = 10
)

type Kind string

const (
	KindTotal    Kind = "total"
	KindPeriodic Kind = "periodic"
)

type Rule struct {
	ID            uint
	Revision      uint64
	Kind          Kind
	LimitNanoUSD  int64
	PeriodSeconds int64
}

type RestoredState struct {
	AccessKeyID       uint
	RuleID            uint
	RuleRevision      uint64
	UsedNanoUSD       int64
	WindowStartedAtMS *int64
	WindowEndsAtMS    *int64
	WindowGeneration  uint64
	SnapshotVersion   uint64
}

type TicketRule struct {
	RuleID           uint
	RuleRevision     uint64
	WindowGeneration uint64
}

type Ticket struct {
	AccessKeyID uint
	Rules       []TicketRule
}

type RuleStatus string

const (
	RuleStatusAvailable RuleStatus = "available"
	RuleStatusInactive  RuleStatus = "inactive"
	RuleStatusExhausted RuleStatus = "exhausted"
)

type RuleView struct {
	Rule
	UsedNanoUSD       int64
	RemainingNanoUSD  int64
	Status            RuleStatus
	WindowStartedAtMS *int64
	WindowEndsAtMS    *int64
	WindowGeneration  uint64
}

type Decision struct {
	Allowed           bool
	Recoverable       bool
	NextAvailableAtMS *int64
	BlockingRules     []RuleView
}

type View struct {
	ObservedAtMS      int64
	Allowed           bool
	Recoverable       bool
	NextAvailableAtMS *int64
	Rules             []RuleView
}

type Stats struct {
	OverflowFaultTotal uint64
}

type CompletionFault string

const (
	CompletionFaultNegativeEstimate CompletionFault = "negative_estimate"
	CompletionFaultOverflow         CompletionFault = "overflow"
)

type CompletionResult struct {
	Fault CompletionFault
}

type Runtime struct {
	mu             sync.RWMutex
	entries        map[uint]*accessKeyEntry
	restored       map[uint]RestoredState
	restorePending bool
	dirtyNotifier  func()

	overflowFaultTotal atomic.Uint64
}

type accessKeyEntry struct {
	mu    sync.Mutex
	rules []*runtimeRule
	byID  map[uint]*runtimeRule
}

type runtimeRule struct {
	definition        Rule
	usedNanoUSD       int64
	windowStartedAtMS *int64
	windowEndsAtMS    *int64
	windowGeneration  uint64
	snapshotVersion   uint64
	dirty             bool
}

func NewRuntime() *Runtime {
	return &Runtime{entries: make(map[uint]*accessKeyEntry)}
}

// SetDirtyNotifier installs the process-owned non-blocking checkpoint wake-up.
func (runtime *Runtime) SetDirtyNotifier(notifier func()) {
	if runtime == nil {
		return
	}
	runtime.mu.Lock()
	runtime.dirtyNotifier = notifier
	runtime.mu.Unlock()
}

// ValidateDefinitions verifies all per-key rule identities and limits.
func ValidateDefinitions(definitions map[uint][]Rule) error {
	_, err := normalizeDefinitions(definitions)
	return err
}

func (runtime *Runtime) Restore(states []RestoredState) error {
	if runtime == nil {
		return fmt.Errorf("restore access quota runtime: runtime is nil")
	}
	restored := make(map[uint]RestoredState, len(states))
	for _, state := range states {
		if err := validateRestoredState(state); err != nil {
			return err
		}
		if _, exists := restored[state.RuleID]; exists {
			return fmt.Errorf("restore access quota rule %d: duplicate state", state.RuleID)
		}
		restored[state.RuleID] = cloneRestoredState(state)
	}

	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if runtime.restorePending || len(runtime.entries) != 0 {
		return fmt.Errorf("restore access quota runtime: runtime is already initialized")
	}
	runtime.restored = restored
	runtime.restorePending = true
	return nil
}

func (runtime *Runtime) Reconcile(definitions map[uint][]Rule) error {
	if runtime == nil {
		return fmt.Errorf("reconcile access quota runtime: runtime is nil")
	}
	normalized, err := normalizeDefinitions(definitions)
	if err != nil {
		return err
	}

	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if runtime.entries == nil {
		runtime.entries = make(map[uint]*accessKeyEntry)
	}
	if runtime.restorePending {
		entries, err := entriesFromRestore(normalized, runtime.restored)
		if err != nil {
			return err
		}
		runtime.entries = entries
		runtime.restored = nil
		runtime.restorePending = false
		return nil
	}

	next := make(map[uint]*accessKeyEntry, len(normalized))
	for accessKeyID, rules := range normalized {
		if len(rules) == 0 {
			continue
		}
		entry := newAccessKeyEntry(rules)
		if current := runtime.entries[accessKeyID]; current != nil {
			current.mu.Lock()
			for _, rule := range entry.rules {
				previous := current.byID[rule.definition.ID]
				if previous == nil || previous.definition.Revision != rule.definition.Revision {
					continue
				}
				if previous.definition.Kind != rule.definition.Kind ||
					previous.definition.PeriodSeconds != rule.definition.PeriodSeconds {
					current.mu.Unlock()
					return fmt.Errorf(
						"reconcile access quota rule %d: kind or period changed without revision",
						rule.definition.ID,
					)
				}
				cloneRuntimeState(rule, previous)
			}
			current.mu.Unlock()
		}
		next[accessKeyID] = entry
	}
	runtime.entries = next
	return nil
}

func (runtime *Runtime) Stats() Stats {
	if runtime == nil {
		return Stats{}
	}
	return Stats{OverflowFaultTotal: runtime.overflowFaultTotal.Load()}
}

func (runtime *Runtime) notifyDirty() {
	if runtime == nil {
		return
	}
	runtime.mu.RLock()
	notifier := runtime.dirtyNotifier
	runtime.mu.RUnlock()
	if notifier != nil {
		notifier()
	}
}

func (runtime *Runtime) lockEntry(accessKeyID uint) *accessKeyEntry {
	if runtime == nil || accessKeyID == 0 {
		return nil
	}
	runtime.mu.RLock()
	entry := runtime.entries[accessKeyID]
	if entry != nil {
		entry.mu.Lock()
	}
	runtime.mu.RUnlock()
	return entry
}

func normalizeDefinitions(definitions map[uint][]Rule) (map[uint][]Rule, error) {
	normalized := make(map[uint][]Rule, len(definitions))
	globalRuleIDs := make(map[uint]uint)
	for accessKeyID, source := range definitions {
		if accessKeyID == 0 {
			return nil, fmt.Errorf("reconcile access quota runtime: access key ID is required")
		}
		if len(source) == 0 {
			continue
		}
		rules := append([]Rule(nil), source...)
		if err := validateRules(accessKeyID, rules, globalRuleIDs); err != nil {
			return nil, err
		}
		sortRules(rules)
		normalized[accessKeyID] = rules
	}
	return normalized, nil
}

func validateRules(accessKeyID uint, rules []Rule, globalRuleIDs map[uint]uint) error {
	totalCount := 0
	periodicCount := 0
	periods := make(map[int64]struct{})
	for _, rule := range rules {
		if rule.ID == 0 || rule.Revision == 0 || rule.LimitNanoUSD <= 0 {
			return fmt.Errorf("reconcile access quota rule for key %d: invalid identity, revision, or limit", accessKeyID)
		}
		if owner, exists := globalRuleIDs[rule.ID]; exists {
			return fmt.Errorf("reconcile access quota rule %d: duplicate across keys %d and %d", rule.ID, owner, accessKeyID)
		}
		globalRuleIDs[rule.ID] = accessKeyID
		switch rule.Kind {
		case KindTotal:
			totalCount++
			if rule.PeriodSeconds != 0 {
				return fmt.Errorf("reconcile total access quota rule %d: period must be zero", rule.ID)
			}
		case KindPeriodic:
			periodicCount++
			if rule.PeriodSeconds < MinPeriodSeconds || rule.PeriodSeconds > MaxPeriodSeconds {
				return fmt.Errorf("reconcile periodic access quota rule %d: period is out of range", rule.ID)
			}
			if _, exists := periods[rule.PeriodSeconds]; exists {
				return fmt.Errorf("reconcile periodic access quota rule %d: duplicate period", rule.ID)
			}
			periods[rule.PeriodSeconds] = struct{}{}
		default:
			return fmt.Errorf("reconcile access quota rule %d: invalid kind %q", rule.ID, rule.Kind)
		}
	}
	if totalCount > 1 || periodicCount > MaxPeriodicRules {
		return fmt.Errorf("reconcile access quota rules for key %d: rule count exceeds limit", accessKeyID)
	}
	return nil
}

func sortRules(rules []Rule) {
	sort.Slice(rules, func(i, j int) bool {
		left, right := rules[i], rules[j]
		if left.Kind != right.Kind {
			return left.Kind == KindTotal
		}
		if left.PeriodSeconds != right.PeriodSeconds {
			return left.PeriodSeconds < right.PeriodSeconds
		}
		return left.ID < right.ID
	})
}

func newAccessKeyEntry(rules []Rule) *accessKeyEntry {
	entry := &accessKeyEntry{rules: make([]*runtimeRule, 0, len(rules)), byID: make(map[uint]*runtimeRule, len(rules))}
	for _, definition := range rules {
		rule := &runtimeRule{definition: definition, snapshotVersion: 1}
		entry.rules = append(entry.rules, rule)
		entry.byID[definition.ID] = rule
	}
	return entry
}
