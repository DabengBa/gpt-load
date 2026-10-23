package state

import "sync"

type entryCapacity struct {
	key     RouteEntryKey
	limit   int
	inUse   int
	active  bool
	present bool
}

type CapacityAdmission uint8

const (
	CapacityAdmitted CapacityAdmission = iota
	CapacityFull
	CapacityUnavailable
	CapacitySnapshotChanged
)

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
		if !r.entry.present && r.entry.inUse == 0 && r.manager.capacity[r.entry.key] == r.entry {
			delete(r.manager.capacity, r.entry.key)
		}
		r.manager.publishMu.Unlock()
	})
}

// AcquireCapacity uses only the most recently published limit. Request
// snapshots cannot restore an obsolete limit. Unlimited entries also count.
func (m *Manager) AcquireCapacity(key RouteEntryKey, affinity bool) (*CapacityReservation, bool) {
	reservation, result := m.acquireCapacity(nil, key, affinity, false)
	return reservation, result == CapacityAdmitted
}

func (m *Manager) AcquireCapacityForSnapshot(snapshot *ConfigSnapshot, key RouteEntryKey, affinity bool) (*CapacityReservation, CapacityAdmission) {
	return m.acquireCapacity(snapshot, key, affinity, true)
}

func (m *Manager) acquireCapacity(snapshot *ConfigSnapshot, key RouteEntryKey, affinity bool, checkSnapshot bool) (*CapacityReservation, CapacityAdmission) {
	m.publishMu.Lock()
	defer m.publishMu.Unlock()
	entry := m.capacity[key]
	if entry == nil || !entry.present || !entry.active {
		if checkSnapshot && m.current.Load() != snapshot {
			return nil, CapacitySnapshotChanged
		}
		return nil, CapacityUnavailable
	}
	if !affinity && entry.limit > 0 && entry.inUse >= entry.limit {
		return nil, CapacityFull
	}
	entry.inUse++
	return &CapacityReservation{manager: m, entry: entry}, CapacityAdmitted
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
				entry = &entryCapacity{key: key}
			}
			entry.limit = model.ConcurrencyLimit
			entry.active = active
			entry.present = true
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
						entry = &entryCapacity{key: key}
					}
					_, disabled := snapshot.DisabledGroups[target.GroupID]
					active := !disabled
					entry.limit = target.ConcurrencyLimit
					entry.active = active
					entry.present = true
					next[key] = entry
				}
			}
		}
	}
	for key, entry := range m.capacity {
		if next[key] == nil && entry.inUse > 0 {
			entry.active = false
			entry.present = false
			next[key] = entry
		}
	}
	m.capacity = next
}
