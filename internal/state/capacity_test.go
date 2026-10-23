package state

import (
	"sync"
	"sync/atomic"
	"testing"
)

func capacitySnapshot(limit int) *ConfigSnapshot {
	return &ConfigSnapshot{Groups: map[uint]GroupView{1: {Models: []ModelConfig{{EntryID: "e1", ConcurrencyLimit: limit}}}}}
}

func TestCapacityAtomicAndAffinity(t *testing.T) {
	m := NewManager()
	m.publishCompiled(capacitySnapshot(1), nil)
	key := RouteEntryKey{1, "e1"}
	var admitted atomic.Int32
	var wg sync.WaitGroup
	var attempted sync.WaitGroup
	attempted.Add(32)
	start, release := make(chan struct{}), make(chan struct{})
	for range 32 {
		wg.Go(func() {
			<-start
			reservation, ok := m.AcquireCapacity(key, false)
			attempted.Done()
			if ok {
				admitted.Add(1)
				<-release
				reservation.Release()
				reservation.Release()
			}
		})
	}
	close(start)
	attempted.Wait()
	close(release)
	wg.Wait()
	if admitted.Load() != 1 {
		t.Fatalf("admitted = %d, want 1", admitted.Load())
	}
	first, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("first admission rejected")
	}
	affinity, ok := m.AcquireCapacity(key, true)
	if !ok || m.CapacityInUse(key) != 2 {
		t.Fatal("affinity must exceed limit and count")
	}
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("ordinary admission exceeded limit")
	}
	first.Release()
	affinity.Release()
	if m.CapacityInUse(key) != 0 {
		t.Fatal("reservation leaked")
	}
}

func TestCapacityPublicationPreservesOccupancy(t *testing.T) {
	m := NewManager()
	m.publishCompiled(capacitySnapshot(0), nil)
	key := RouteEntryKey{1, "e1"}
	reservation, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("unlimited admission rejected")
	}
	m.publishCompiled(capacitySnapshot(1), nil)
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("publication reset unlimited occupancy")
	}
	m.publishCompiled(&ConfigSnapshot{}, nil)
	if _, ok := m.AcquireCapacity(key, true); ok {
		t.Fatal("deleted entry admitted")
	}
	reservation.Release()
	reservation.Release()
	if m.CapacityInUse(key) != 0 {
		t.Fatal("deleted reservation leaked")
	}
	m.publishCompiled(capacitySnapshot(2), nil)
	first, _ := m.AcquireCapacity(key, false)
	second, _ := m.AcquireCapacity(key, false)
	m.publishCompiled(capacitySnapshot(1), nil)
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("lower limit ignored in-flight reservations")
	}
	first.Release()
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("remaining in-flight reservation ignored")
	}
	second.Release()
	last, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("released capacity unavailable")
	}
	m.publishCompiled(&ConfigSnapshot{}, nil)
	m.publishCompiled(capacitySnapshot(1), nil)
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("recreated entry ignored old in-flight reservation")
	}
	last.Release()
	replacement, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("new entry unavailable after old reservation released")
	}
	replacement.Release()
}

func TestCapacityDeletionAndRecreationPreservesInFlightOccupancy(t *testing.T) {
	m := NewManager()
	key := RouteEntryKey{1, "e1"}
	m.publishCompiled(capacitySnapshot(1), nil)
	first, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("initial admission rejected")
	}
	m.publishCompiled(&ConfigSnapshot{}, nil)
	m.publishCompiled(capacitySnapshot(1), nil)
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("recreated entry admitted while old request remained in flight")
	}
	first.Release()
	second, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("capacity did not return after old request released")
	}
	second.Release()
}

func TestCapacityAdmissionDistinguishesFullFromStaleSnapshot(t *testing.T) {
	m := NewManager()
	key := RouteEntryKey{1, "e1"}
	old := capacitySnapshot(1)
	m.publishCompiled(old, nil)
	occupied, result := m.AcquireCapacityForSnapshot(m.Current(), key, false)
	if result != CapacityAdmitted {
		t.Fatalf("initial admission = %v", result)
	}
	defer occupied.Release()
	if _, result = m.AcquireCapacityForSnapshot(m.Current(), key, false); result != CapacityFull {
		t.Fatalf("full entry admission = %v, want CapacityFull", result)
	}
	m.publishCompiled(&ConfigSnapshot{}, nil)
	if _, result = m.AcquireCapacityForSnapshot(old, key, false); result != CapacitySnapshotChanged {
		t.Fatalf("stale snapshot admission = %v, want CapacitySnapshotChanged", result)
	}
	if _, result = m.AcquireCapacityForSnapshot(m.Current(), RouteEntryKey{1, "missing"}, false); result != CapacityUnavailable {
		t.Fatalf("missing entry admission = %v, want CapacityUnavailable", result)
	}
}

func TestCapacityDisabledGroupRetainsOccupancyUntilRelease(t *testing.T) {
	m := NewManager()
	key := RouteEntryKey{1, "e1"}
	m.publishCompiled(capacitySnapshot(1), nil)
	reservation, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("initial admission rejected")
	}
	disabled := &ConfigSnapshot{DisabledGroups: map[uint]GroupView{
		1: {Models: []ModelConfig{{EntryID: "e1", ConcurrencyLimit: 1}}},
	}}
	m.publishCompiled(disabled, nil)
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("disabled group admitted new request")
	}
	m.publishCompiled(capacitySnapshot(1), nil)
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("re-enabled group reset in-flight occupancy")
	}
	reservation.Release()
	if _, ok := m.AcquireCapacity(key, false); !ok {
		t.Fatal("capacity did not return after old request released")
	}
}

func TestCapacityLimitChangeTakesEffectWithoutLeaking(t *testing.T) {
	m := NewManager()
	key := RouteEntryKey{1, "e1"}
	m.publishCompiled(capacitySnapshot(0), nil)
	reservations := make([]*CapacityReservation, 0, 3)
	for range 3 {
		reservation, ok := m.AcquireCapacity(key, false)
		if !ok {
			t.Fatal("unlimited entry rejected")
		}
		reservations = append(reservations, reservation)
	}
	m.publishCompiled(capacitySnapshot(5), nil)
	for range 2 {
		reservation, ok := m.AcquireCapacity(key, false)
		if !ok {
			t.Fatal("raised limit rejected below the new cap")
		}
		reservations = append(reservations, reservation)
	}
	if m.CapacityInUse(key) != 5 {
		t.Fatalf("in use = %d, want 5", m.CapacityInUse(key))
	}
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("new cap admitted an extra request")
	}
	for _, reservation := range reservations {
		reservation.Release()
	}
	if m.CapacityInUse(key) != 0 {
		t.Fatalf("in use after release = %d, want 0", m.CapacityInUse(key))
	}
}

func TestCapacityCountsAnyRouteModeOfOneEntryOnce(t *testing.T) {
	m := NewManager()
	// One route entry reachable through several operations/protocols and both
	// route modes shares a single in-flight budget: the key is (group, entry).
	snapshot := &ConfigSnapshot{ExecutionCandidates: ExecutionCandidateIndex{
		"chat_completion": {
			"gpt-4o": {
				"public": {
					{GroupID: 1, EntryID: "e1", ConcurrencyLimit: 1},
				},
				"alias": {
					{GroupID: 1, EntryID: "e1", ConcurrencyLimit: 1},
				},
			},
		},
		"embedding": {
			"text-embedding-3": {
				"public": {
					{GroupID: 1, EntryID: "e1", ConcurrencyLimit: 1},
				},
			},
		},
	}}
	m.publishCompiled(snapshot, nil)
	key := RouteEntryKey{1, "e1"}
	reservation, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("candidate-only entry rejected")
	}
	defer reservation.Release()
	if _, ok := m.AcquireCapacity(key, false); ok {
		t.Fatal("fan-out over-counted capacity: entry admitted twice")
	}
	if _, ok := m.AcquireCapacity(key, true); !ok {
		t.Fatal("affinity overflow rejected")
	}
}

func TestCapacityConcurrentAcquireReleaseUnderPublication(t *testing.T) {
	m := NewManager()
	key := RouteEntryKey{1, "e1"}
	m.publishCompiled(capacitySnapshot(4), nil)
	var admitted atomic.Int32
	var wg sync.WaitGroup
	for range 64 {
		wg.Go(func() {
			reservation, ok := m.AcquireCapacity(key, false)
			if !ok {
				return
			}
			admitted.Add(1)
			reservation.Release()
		})
	}
	for range 8 {
		wg.Go(func() {
			m.publishCompiled(capacitySnapshot(4), nil)
		})
	}
	wg.Wait()
	if m.CapacityInUse(key) != 0 {
		t.Fatalf("in use = %d, want 0", m.CapacityInUse(key))
	}
	if admitted.Load() == 0 {
		t.Fatal("no admission under concurrent publication")
	}
}
