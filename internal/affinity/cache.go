package affinity

import (
	"container/list"
	"context"
	"sync"
	"time"
)

const (
	DefaultTTL      = time.Hour
	DefaultCapacity = 10_000
)

// DurablePolicy bounds the durable affinity binding set with the same runtime
// limits used by the hot cache.
type DurablePolicy struct {
	TTL      time.Duration
	Capacity int
}

func (policy DurablePolicy) Valid() bool {
	return policy.TTL > 0 && policy.Capacity > 0
}

// BindingStore is the durable authority for affinity bindings that the gateway
// coordinates with its evictable hot cache. A lookup error is an authoritative
// failure and must not be mistaken for a missing binding.
type BindingStore interface {
	Lookup(context.Context, Key, DurablePolicy) (Target, bool, error)
	Upsert(context.Context, Key, Target, DurablePolicy) error
}

// DurableBindingCleaner owns the lifecycle bound of persisted affinity rows.
// It is called by the control runtime using the current runtime policy.
type DurableBindingCleaner interface {
	SweepAffinityBindings(context.Context, time.Time, DurablePolicy) error
}

// Target is the exact Credential identity remembered as a preference.
type Target struct {
	GroupID            uint
	CredentialID       uint
	IdentityGeneration uint64
}

func (target Target) Valid() bool {
	return target.GroupID != 0 && target.CredentialID != 0 && target.IdentityGeneration != 0
}

// Observation is a versioned cache lookup used for conditional success updates.
type Observation struct {
	Target   Target
	key      Key
	revision uint64
	version  uint64
	found    bool
}

func (observation Observation) Found() bool {
	return observation.found
}

type cacheEntry struct {
	key       Key
	target    Target
	version   uint64
	expiresAt time.Time
}

// Cache is a bounded hot cache for affinity bindings. Eviction only removes
// the in-memory copy; the owning storage layer is responsible for recovery.
type Cache struct {
	mu          sync.Mutex
	entries     map[Key]*list.Element
	recent      list.List
	capacity    int
	ttl         time.Duration
	now         func() time.Time
	revision    uint64
	nextVersion uint64
}

func NewCache() *Cache {
	return newCache(DefaultCapacity, DefaultTTL, time.Now)
}

func newCache(capacity int, ttl time.Duration, now func() time.Time) *Cache {
	return &Cache{
		entries:  make(map[Key]*list.Element),
		capacity: capacity,
		ttl:      ttl,
		now:      now,
	}
}

// Configure applies one frozen runtime configuration revision. A newer revision
// adopts the given capacity and TTL but clears entries only when capacity or TTL
// changed relative to the current cache state, so unrelated revision bumps keep
// established bindings. Revision stays monotonic and doubles as the stale-write
// guard: RecordSuccess rejects observations taken under an older revision, while
// older Configure calls are refused outright.
func (cache *Cache) Configure(revision uint64, capacity int, ttl time.Duration) bool {
	if cache == nil || revision == 0 || capacity <= 0 || ttl <= 0 || cache.now == nil {
		return false
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if revision < cache.revision {
		return false
	}
	if revision == cache.revision {
		return cache.capacity == capacity && cache.ttl == ttl
	}
	cache.revision = revision
	if cache.capacity != capacity || cache.ttl != ttl {
		clear(cache.entries)
		cache.recent.Init()
	}
	cache.capacity = capacity
	cache.ttl = ttl
	return true
}

func (cache *Cache) Lookup(key Key) Observation {
	if cache == nil || !key.Valid() || cache.now == nil {
		return Observation{}
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	element := cache.currentElementLocked(key, cache.now())
	if element == nil {
		return Observation{key: key, revision: cache.revision}
	}
	cache.recent.MoveToFront(element)
	entry := element.Value.(*cacheEntry)
	return Observation{
		Target: entry.target, key: key, revision: cache.revision,
		version: entry.version, found: true,
	}
}

// RecordSuccess conditionally learns the successful target. It returns true
// when this call inserted, refreshed, or changed the current mapping.
func (cache *Cache) RecordSuccess(key Key, observed Observation, target Target) bool {
	if cache == nil || !key.Valid() || !target.Valid() || cache.now == nil {
		return false
	}
	if observed.key.Valid() && observed.key != key {
		return false
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()
	if cache.capacity <= 0 || cache.ttl <= 0 || observed.revision != cache.revision {
		return false
	}
	now := cache.now()
	element := cache.currentElementLocked(key, now)
	if observed.found {
		if element != nil {
			current := element.Value.(*cacheEntry)
			if current.version != observed.version || current.target != observed.Target {
				return false
			}
		} else {
			return cache.insertLocked(key, target, now)
		}
	} else if element != nil {
		return false
	}

	if element == nil {
		return cache.insertLocked(key, target, now)
	}
	cache.nextVersion++
	entry := element.Value.(*cacheEntry)
	entry.target = target
	entry.version = cache.nextVersion
	entry.expiresAt = now.Add(cache.ttl)
	cache.recent.MoveToFront(element)
	return true
}

func (cache *Cache) entryCount() int {
	if cache == nil {
		return 0
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	return len(cache.entries)
}

func (cache *Cache) currentElementLocked(key Key, now time.Time) *list.Element {
	element := cache.entries[key]
	if element == nil {
		return nil
	}
	entry := element.Value.(*cacheEntry)
	if !now.Before(entry.expiresAt) {
		cache.removeLocked(element)
		return nil
	}
	return element
}

func (cache *Cache) insertLocked(key Key, target Target, now time.Time) bool {
	for len(cache.entries) >= cache.capacity {
		oldest := cache.recent.Back()
		if oldest == nil {
			break
		}
		cache.removeLocked(oldest)
	}
	cache.nextVersion++
	entry := &cacheEntry{
		key: key, target: target, version: cache.nextVersion,
		expiresAt: now.Add(cache.ttl),
	}
	cache.entries[key] = cache.recent.PushFront(entry)
	return true
}

func (cache *Cache) removeLocked(element *list.Element) {
	entry := element.Value.(*cacheEntry)
	delete(cache.entries, entry.key)
	cache.recent.Remove(element)
}
