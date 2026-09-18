package affinity

import (
	"testing"
	"time"
)

func TestCacheLearnsAndRefreshesSuccessfulTarget(t *testing.T) {
	now := time.Date(2026, time.August, 11, 12, 0, 0, 0, time.UTC)
	cache := newCache(2, time.Hour, func() time.Time { return now })
	key := Key("one")
	target := Target{GroupID: 1, CredentialID: 11, IdentityGeneration: 101}

	miss := cache.Lookup(key)
	if miss.Found() {
		t.Fatalf("initial Lookup() = %#v, want miss", miss)
	}
	if !cache.RecordSuccess(key, miss, target) {
		t.Fatal("RecordSuccess() = false, want insert")
	}
	first := cache.Lookup(key)
	if !first.Found() || first.Target != target {
		t.Fatalf("Lookup() = %#v, want target %#v", first, target)
	}

	now = now.Add(50 * time.Minute)
	if !cache.RecordSuccess(key, first, target) {
		t.Fatal("RecordSuccess() = false, want TTL refresh")
	}
	now = now.Add(20 * time.Minute)
	if refreshed := cache.Lookup(key); !refreshed.Found() || refreshed.Target != target {
		t.Fatalf("Lookup() after refreshed TTL = %#v, want hit", refreshed)
	}
}

func TestCacheLookupDoesNotRefreshTTL(t *testing.T) {
	now := time.Date(2026, time.August, 11, 12, 0, 0, 0, time.UTC)
	cache := newCache(2, time.Hour, func() time.Time { return now })
	key := Key("one")
	target := Target{GroupID: 1, CredentialID: 11, IdentityGeneration: 101}
	cache.RecordSuccess(key, Observation{}, target)

	now = now.Add(50 * time.Minute)
	if !cache.Lookup(key).Found() {
		t.Fatal("Lookup() before expiry = miss, want hit")
	}
	now = now.Add(11 * time.Minute)
	if got := cache.Lookup(key); got.Found() {
		t.Fatalf("Lookup() after original expiry = %#v, want miss", got)
	}
}

func TestCacheFirstSuccessWinsAndFallbackUsesCompareAndSwap(t *testing.T) {
	cache := newCache(4, time.Hour, time.Now)
	key := Key("conversation")
	first := Target{GroupID: 1, CredentialID: 11, IdentityGeneration: 101}
	second := Target{GroupID: 1, CredentialID: 12, IdentityGeneration: 102}

	missOne := cache.Lookup(key)
	missTwo := cache.Lookup(key)
	if !cache.RecordSuccess(key, missOne, first) {
		t.Fatal("first miss success did not create mapping")
	}
	if cache.RecordSuccess(key, missTwo, second) {
		t.Fatal("second concurrent miss overwrote first-success mapping")
	}

	observedFirst := cache.Lookup(key)
	if !cache.RecordSuccess(key, observedFirst, second) {
		t.Fatal("fallback success did not switch observed mapping")
	}
	if cache.RecordSuccess(key, observedFirst, first) {
		t.Fatal("stale success overwrote newer fallback mapping")
	}
	got := cache.Lookup(key)
	if !got.Found() || got.Target != second {
		t.Fatalf("Lookup() = %#v, want fallback target %#v", got, second)
	}
}

func TestCacheEvictsLeastRecentlyUsedEntry(t *testing.T) {
	cache := newCache(2, time.Hour, time.Now)
	target := Target{GroupID: 1, CredentialID: 11, IdentityGeneration: 101}
	for _, key := range []Key{"one", "two"} {
		cache.RecordSuccess(key, Observation{}, target)
	}
	cache.Lookup("one")
	cache.RecordSuccess("three", Observation{}, target)

	if cache.Lookup("two").Found() {
		t.Fatal("least recently used entry remained cached")
	}
	if !cache.Lookup("one").Found() || !cache.Lookup("three").Found() || cache.entryCount() != 2 {
		t.Fatalf("cache state invalid after eviction; entries=%d", cache.entryCount())
	}
}

func TestCacheRejectsInvalidInputs(t *testing.T) {
	cache := newCache(2, time.Hour, time.Now)
	validTarget := Target{GroupID: 1, CredentialID: 11, IdentityGeneration: 101}
	for _, test := range []struct {
		key    Key
		target Target
	}{
		{target: validTarget},
		{key: "key"},
		{key: "key", target: Target{GroupID: 1, CredentialID: 11}},
	} {
		if cache.RecordSuccess(test.key, Observation{}, test.target) {
			t.Fatalf("RecordSuccess(%q, %#v) = true, want false", test.key, test.target)
		}
	}
	if cache.Lookup("").Found() || cache.entryCount() != 0 {
		t.Fatalf("invalid inputs changed cache; entries=%d", cache.entryCount())
	}
}

func TestCacheConfigureClearsEntriesAndRejectsOlderRevision(t *testing.T) {
	now := time.Date(2026, time.August, 12, 12, 0, 0, 0, time.UTC)
	cache := newCache(2, time.Hour, func() time.Time { return now })
	key := Key("one")
	target := Target{GroupID: 1, CredentialID: 11, IdentityGeneration: 101}
	if !cache.Configure(1, 2, time.Hour) {
		t.Fatal("Configure(1) = false")
	}
	observed := cache.Lookup(key)
	if !cache.RecordSuccess(key, observed, target) {
		t.Fatal("RecordSuccess() = false")
	}
	stale := cache.Lookup(key)
	if !cache.Configure(2, 2, time.Hour) {
		t.Fatal("Configure(2) with unchanged capacity and TTL = false")
	}
	if !cache.Lookup(key).Found() || cache.entryCount() != 1 {
		t.Fatal("revision change without configuration change cleared entries")
	}
	if !cache.Configure(3, 1, 30*time.Minute) {
		t.Fatal("Configure(3) = false")
	}
	if cache.Lookup(key).Found() || cache.entryCount() != 0 {
		t.Fatal("new configuration did not clear old entries")
	}
	if cache.Configure(1, 2, time.Hour) {
		t.Fatal("older configuration revision was accepted")
	}
	if cache.RecordSuccess(key, stale, target) {
		t.Fatal("request from an older configuration revision rewrote the cache")
	}
	for _, nextKey := range []Key{"two", "three"} {
		if !cache.RecordSuccess(nextKey, cache.Lookup(nextKey), target) {
			t.Fatalf("RecordSuccess(%q) = false", nextKey)
		}
	}
	if cache.Lookup("two").Found() || !cache.Lookup("three").Found() || cache.entryCount() != 1 {
		t.Fatal("configured capacity was not enforced")
	}
	now = now.Add(31 * time.Minute)
	if cache.Lookup("three").Found() {
		t.Fatal("configured TTL was not enforced")
	}
}

func TestCacheConfigureKeepsEntriesWhenCapacityAndTTLUnchanged(t *testing.T) {
	now := time.Date(2026, time.August, 12, 12, 0, 0, 0, time.UTC)
	cache := newCache(2, time.Hour, func() time.Time { return now })
	key := Key("one")
	target := Target{GroupID: 1, CredentialID: 11, IdentityGeneration: 101}
	if !cache.Configure(1, 2, time.Hour) {
		t.Fatal("Configure(1) = false")
	}
	observed := cache.Lookup(key)
	if !cache.RecordSuccess(key, observed, target) {
		t.Fatal("RecordSuccess() = false")
	}

	if !cache.Configure(2, 2, time.Hour) {
		t.Fatal("Configure(2) with unchanged capacity and TTL = false, want true")
	}
	got := cache.Lookup(key)
	if !got.Found() || got.Target != target {
		t.Fatalf("Lookup() after revision-only change = %#v, want hit on %#v", got, target)
	}
	if cache.entryCount() != 1 {
		t.Fatalf("entryCount() after revision-only change = %d, want 1", cache.entryCount())
	}
	if cache.RecordSuccess(key, observed, target) {
		t.Fatal("observation from an older revision rewrote the cache after a revision-only change")
	}
	next := Target{GroupID: 1, CredentialID: 12, IdentityGeneration: 102}
	if !cache.RecordSuccess(key, cache.Lookup(key), next) {
		t.Fatal("RecordSuccess() with current-revision observation = false, want update")
	}
	if refreshed := cache.Lookup(key); !refreshed.Found() || refreshed.Target != next {
		t.Fatalf("Lookup() after current-revision update = %#v, want %#v", refreshed, next)
	}
}

func TestCacheConfigureClearsEntriesOnCapacityOrTTLChange(t *testing.T) {
	now := time.Date(2026, time.August, 12, 12, 0, 0, 0, time.UTC)
	target := Target{GroupID: 1, CredentialID: 11, IdentityGeneration: 101}
	for _, test := range []struct {
		name     string
		capacity int
		ttl      time.Duration
	}{
		{name: "capacity change", capacity: 1, ttl: time.Hour},
		{name: "ttl change", capacity: 2, ttl: 30 * time.Minute},
		{name: "capacity and ttl change", capacity: 3, ttl: 2 * time.Hour},
	} {
		t.Run(test.name, func(t *testing.T) {
			cache := newCache(2, time.Hour, func() time.Time { return now })
			if !cache.Configure(1, 2, time.Hour) {
				t.Fatal("Configure(1) = false")
			}
			if !cache.RecordSuccess("one", cache.Lookup("one"), target) {
				t.Fatal("RecordSuccess() = false")
			}
			if !cache.Configure(2, test.capacity, test.ttl) {
				t.Fatalf("Configure(2, %d, %s) = false, want true", test.capacity, test.ttl)
			}
			if cache.Lookup("one").Found() || cache.entryCount() != 0 {
				t.Fatalf("changed capacity/TTL did not clear entries; entries=%d", cache.entryCount())
			}
		})
	}
}

func TestCacheConfigureReportsUnchangedRevisionMatchAndRejectsOlder(t *testing.T) {
	cache := newCache(2, time.Hour, time.Now)
	if !cache.Configure(1, 2, time.Hour) {
		t.Fatal("Configure(1, 2, hour) = false, want true")
	}
	if !cache.Configure(1, 2, time.Hour) {
		t.Fatal("Configure() with matching revision and configuration = false, want true")
	}
	for _, test := range []struct {
		name     string
		revision uint64
		capacity int
		ttl      time.Duration
	}{
		{name: "revision zero", revision: 0, capacity: 2, ttl: time.Hour},
		{name: "capacity mismatch", revision: 1, capacity: 3, ttl: time.Hour},
		{name: "ttl mismatch", revision: 1, capacity: 2, ttl: 30 * time.Minute},
	} {
		t.Run(test.name, func(t *testing.T) {
			if cache.Configure(test.revision, test.capacity, test.ttl) {
				t.Fatalf("Configure(%d, %d, %s) = true, want false", test.revision, test.capacity, test.ttl)
			}
		})
	}
	if !cache.Configure(2, 2, time.Hour) {
		t.Fatal("Configure(2, 2, hour) = false, want true")
	}
	if cache.Configure(1, 2, time.Hour) {
		t.Fatal("older configuration revision was accepted")
	}
}
