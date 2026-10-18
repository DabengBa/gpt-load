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
	replacement, ok := m.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("new entry unavailable")
	}
	last.Release()
	if m.CapacityInUse(key) != 1 {
		t.Fatal("old reservation released replacement occupancy")
	}
	replacement.Release()
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
