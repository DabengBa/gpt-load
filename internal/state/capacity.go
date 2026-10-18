package state

import "sync"

type entryCapacity struct {
	limit  int
	inUse  int
	active bool
}

// CapacityReservation owns one execution slot. Release is safe to repeat and
// retains its original record even if the entry is removed during execution.
type CapacityReservation struct {
	once    sync.Once
	manager *Manager
	entry   *entryCapacity
}

func (r *CapacityReservation) Release() {
	if r == nil {
		return
	}
	r.once.Do(func() {
		r.manager.publishMu.Lock()
		r.entry.inUse--
		r.manager.publishMu.Unlock()
	})
}

// AcquireCapacity uses only the most recently published limit. Request
// snapshots cannot restore an obsolete limit. Unlimited entries also count.
func (m *Manager) AcquireCapacity(key RouteEntryKey, affinity bool) (*CapacityReservation, bool) {
	m.publishMu.Lock()
	defer m.publishMu.Unlock()
	entry := m.capacity[key]
	if entry == nil || !entry.active {
		return nil, false
	}
	if !affinity && entry.limit > 0 && entry.inUse >= entry.limit {
		return nil, false
	}
	entry.inUse++
	return &CapacityReservation{manager: m, entry: entry}, true
}

func (m *Manager) CapacityInUse(key RouteEntryKey) int {
	m.publishMu.RLock()
	defer m.publishMu.RUnlock()
	if entry := m.capacity[key]; entry != nil {
		return entry.inUse
	}
	return 0
}

func (m *Manager) publishCapacityLocked(snapshot *ConfigSnapshot) {
	next := make(map[RouteEntryKey]*entryCapacity)
	publishGroup := func(groupID uint, group GroupView, active bool) {
		for _, model := range group.Models {
			key := RouteEntryKey{GroupID: groupID, EntryID: model.EntryID}
			entry := m.capacity[key]
			if entry == nil {
				entry = &entryCapacity{}
			}
			entry.limit = model.ConcurrencyLimit
			entry.active = active
			next[key] = entry
		}
	}
	for groupID, group := range snapshot.Groups {
		publishGroup(groupID, group, true)
	}
	for groupID, group := range snapshot.DisabledGroups {
		publishGroup(groupID, group, false)
	}
	// Some compiled snapshots (and lightweight gateway fixtures) expose a
	// route only through ExecutionCandidates. Keep those candidates addressable
	// by the same capacity key; a candidate from a disabled group remains
	// closed to new admissions.
	for operations := range snapshot.ExecutionCandidates {
		for models := range snapshot.ExecutionCandidates[operations] {
			for targets := range snapshot.ExecutionCandidates[operations][models] {
				for _, target := range snapshot.ExecutionCandidates[operations][models][targets] {
					key := RouteEntryKey{GroupID: target.GroupID, EntryID: target.EntryID}
					if next[key] != nil {
						continue
					}
					entry := m.capacity[key]
					if entry == nil {
						entry = &entryCapacity{}
					}
					_, disabled := snapshot.DisabledGroups[target.GroupID]
					active := !disabled
					entry.limit = target.ConcurrencyLimit
					entry.active = active
					next[key] = entry
				}
			}
		}
	}
	m.capacity = next
}
