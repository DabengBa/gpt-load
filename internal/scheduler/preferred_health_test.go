package scheduler

import (
	"errors"
	"math/rand"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

// preferredHealthQuery binds the preferred credential to the exact route of
// twoGroupSnapshot: credential 11 on entry e000000000001 with model
// preferred-model. Route identity is mandatory, so the fixture can only express
// a real binding.
func preferredHealthQuery(preferred uint) Query {
	return Query{
		ClientProtocol:         protocol.OpenAICompletions,
		Operation:              execution.OperationChatCompletion,
		ExternalModel:          modelPointer("public"),
		PreferredCredentialID:  preferred,
		PreferredEntryID:       "e000000000001",
		PreferredUpstreamModel: "preferred-model",
	}
}

func twoGroupSnapshot(t *testing.T) *state.ConfigSnapshot {
	t.Helper()
	return routeEntrySnapshot(t, []state.GroupConfig{
		{ID: 1, Name: "preferred", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{{ID: "preferred-model", Alias: "public", EntryID: "e000000000001", Weight: intPointer(100)}}},
		{ID: 2, Name: "sibling", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{{ID: "sibling-model", Alias: "public", EntryID: "e000000000002", Weight: intPointer(100)}}},
	})
}

func twoGroupRegistry(t *testing.T) *state.CredentialRegistry {
	t.Helper()
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 11, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "f11", EncryptedValue: "c11"},
		{ID: 21, GroupID: 2, Version: 1, IdentityGeneration: 1, Fingerprint: "f21", EncryptedValue: "c21"},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	return registry
}

// R1: the preferred credential's only otherwise-eligible entry is cooled down,
// Next falls back to the sibling credential, and the observation is exposed.
func TestPreferredHealthBypassedOnEntryCooldown(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := twoGroupRegistry(t)
	registry.SetEntryCooldown(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now.Add(time.Minute))
	iterator := New(twoGroupSnapshot(t), registry, preferredHealthQuery(11), rand.New(zeroRandSource{}))
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 21 {
		t.Fatalf("selected credential = %d, want sibling 21", selection.CredentialID)
	}
	if !iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = false, want true after entry-cooldown exclusion")
	}
}

// R1: an entry blacklist on the preferred target is also a health bypass.
func TestPreferredHealthBypassedOnEntryBlacklist(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := twoGroupRegistry(t)
	registry.SetEntryBlacklisted(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"})
	iterator := New(twoGroupSnapshot(t), registry, preferredHealthQuery(11), rand.New(zeroRandSource{}))
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 21 || !iterator.PreferredHealthBypassed() {
		t.Fatalf("selection = %d, bypassed = %v, want 21/true", selection.CredentialID, iterator.PreferredHealthBypassed())
	}
}

// R1: a credential blacklist of the preferred credential is a health bypass.
func TestPreferredHealthBypassedOnCredentialBlacklist(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := twoGroupRegistry(t)
	if _, _, ok := registry.RecordFailureWithBlacklist(11, 1, now.Add(time.Hour)); !ok {
		t.Fatal("RecordFailureWithBlacklist() did not blacklist credential 11")
	}
	iterator := New(twoGroupSnapshot(t), registry, preferredHealthQuery(11), rand.New(zeroRandSource{}))
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 21 || !iterator.PreferredHealthBypassed() {
		t.Fatalf("selection = %d, bypassed = %v, want 21/true", selection.CredentialID, iterator.PreferredHealthBypassed())
	}
}

// R1: a healthy sibling entry of the preferred credential keeps the affinity
// credential selected, so no bypass is recorded.
func TestPreferredHealthySiblingEntryPreventsBypass(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{
		{ID: 1, Name: "preferred", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{
				{ID: "down-model", Alias: "public", EntryID: "e000000000001", Weight: intPointer(100)},
				{ID: "up-model", Alias: "public", EntryID: "e000000000003", Weight: intPointer(100)},
			}},
		{ID: 2, Name: "sibling", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{{ID: "sibling-model", Alias: "public", EntryID: "e000000000002", Weight: intPointer(100)}}},
	})
	registry := twoGroupRegistry(t)
	registry.SetEntryCooldown(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now.Add(time.Minute))
	iterator := New(snapshot, registry, preferredHealthQuery(11), rand.New(zeroRandSource{}))
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 11 {
		t.Fatalf("selected credential = %d, want preferred 11 via healthy sibling entry", selection.CredentialID)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true, want false when preferred stays selected")
	}
}

// R1: auth-unavailable credentials are static restrictions, not health bypasses.

// R1: static exclusions of the preferred candidate must not count as bypass.
func TestPreferredStaticExclusionsAreNotBypass(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name     string
		mutate   func(*state.CredentialRegistry, *Query)
		snapshot *state.ConfigSnapshot
	}{
		{
			name: "missing credential",
			mutate: func(registry *state.CredentialRegistry, _ *Query) {
				registry.RemoveCredential(11)
			},
		},
		{
			name: "not allowed",
			mutate: func(_ *state.CredentialRegistry, query *Query) {
				query.AllowedCredentialIDs = map[uint]struct{}{21: {}}
			},
		},
		{
			name:   "entry weight zero",
			mutate: func(_ *state.CredentialRegistry, _ *Query) {},
			snapshot: routeEntrySnapshot(t, []state.GroupConfig{
				{ID: 1, Name: "preferred", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true,
					Models: []state.ModelConfig{{ID: "preferred-model", Alias: "public", EntryID: "e000000000001", Weight: intPointer(100)}}},
				{ID: 2, Name: "sibling", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true,
					Models: []state.ModelConfig{{ID: "sibling-model", Alias: "public", EntryID: "e000000000002", Weight: intPointer(100)}}},
			}),
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			registry := twoGroupRegistry(t)
			query := preferredHealthQuery(11)
			testCase.mutate(registry, &query)
			snapshot := testCase.snapshot
			if snapshot == nil {
				snapshot = twoGroupSnapshot(t)
			}
			if testCase.name == "entry weight zero" {

				for i := range snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"] {
					if snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"][i].GroupID == 1 {
						snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"][i].EntryWeight = 0
					}

				}
			}

			iterator := New(snapshot, registry, query, rand.New(zeroRandSource{}))
			iterator.now = func() time.Time { return now }

			selection, err := iterator.Next()
			if err != nil {
				t.Fatal(err)
			}
			if selection.CredentialID != 21 {
				t.Fatalf("selected credential = %d, want 21", selection.CredentialID)
			}
			if iterator.PreferredHealthBypassed() {
				t.Fatalf("%s: PreferredHealthBypassed() = true, want false", testCase.name)
			}
		})
	}
}

func TestPreferredObservationResetsAfterNext(t *testing.T) {
	now := inspectNow()
	registry := twoGroupRegistry(t)
	registry.SetEntryCooldown(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now.Add(time.Minute))
	iterator := newWithClock(twoGroupSnapshot(t), registry, preferredHealthQuery(11), rand.New(zeroRandSource{}), func() time.Time { return now })
	if iterator.PreferredHealthBypassed() {
		t.Fatal("observation set before Next")
	}
	if _, err := iterator.Next(); err != nil || !iterator.PreferredHealthBypassed() {
		t.Fatalf("first Next: %v, bypass = %v", err, iterator.PreferredHealthBypassed())
	}
	now = now.Add(2 * time.Minute)
	if selected, err := iterator.Next(); err != nil || selected.CredentialID != 11 || iterator.PreferredHealthBypassed() {
		t.Fatalf("recovered Next: %d/%v, bypass = %v", selected.CredentialID, err, iterator.PreferredHealthBypassed())
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) || iterator.PreferredHealthBypassed() {
		t.Fatalf("exhausted Next: %v, bypass = %v", err, iterator.PreferredHealthBypassed())
	}
}

// R1: exhausted iteration never reports a bypass observation.
func TestPreferredHealthBypassedFalseWhenExhausted(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{
		{ID: 1, Name: "preferred", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{{ID: "preferred-model", Alias: "public", EntryID: "e000000000001", Weight: intPointer(100)}}},
	})
	registry := twoGroupRegistry(t)
	registry.RemoveCredential(21)
	registry.SetEntryCooldown(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now.Add(time.Minute))
	iterator := New(snapshot, registry, preferredHealthQuery(11), rand.New(zeroRandSource{}))
	iterator.now = func() time.Time { return now }

	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Next() error = %v, want ErrExhausted", err)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true after exhaustion, want false")
	}
}

type preferredRuntimeSource struct {
	fakeCredentialSource
	views []state.CredentialRuntimeView
}

func (source preferredRuntimeSource) Snapshot() []state.CredentialRuntimeView { return source.views }

func TestPreferredHealthWithOtherExclusions(t *testing.T) {
	for _, restriction := range []string{"none", "allowed", "skipped", "tried", "cardinality", "weight", "static"} {
		t.Run(restriction, func(t *testing.T) {
			t.Parallel()
			now := inspectNow()
			source := preferredRuntimeSource{
				fakeCredentialSource: fakeCredentialSource{keys: []state.CredentialMeta{{ID: 21, GroupID: 2}}, configuredCounts: map[uint]int{1: 1}},
				views:                []state.CredentialRuntimeView{{ID: 11, GroupID: 1, CooldownUntil: now.Add(time.Minute)}},
			}
			query := preferredHealthQuery(11)
			snapshot := twoGroupSnapshot(t)
			switch restriction {
			case "allowed":
				query.AllowedCredentialIDs = map[uint]struct{}{21: {}}
			case "cardinality":
				source.configuredCounts[1] = 2
			case "static":
				query.AccessKey.Filters.Groups = map[uint]struct{}{2: {}}
			case "weight":
				for i := range snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"] {
					if snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"][i].GroupID == 1 {
						snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"][i].EntryWeight = 0
					}
				}
			}
			iterator := newWithClock(snapshot, source, query, rand.New(zeroRandSource{}), func() time.Time { return now })
			if restriction == "skipped" {
				iterator.SkipGroup(1)
			}
			if restriction == "tried" {
				iterator.tried[candidateKey{credentialID: 11, upstreamModel: "preferred-model"}] = struct{}{}
			}
			selected, err := iterator.Next()
			want := restriction == "none"
			if err != nil || selected.CredentialID != 21 || iterator.PreferredHealthBypassed() != want {
				t.Fatalf("Next = %d/%v, bypass = %v, want 21/nil/%v", selected.CredentialID, err, iterator.PreferredHealthBypassed(), want)
			}
		})
	}
}

// R1: without a positive Snapshot observation of the preferred credential no
// health bypass is ever reported, even when entry health is unavailable.
func TestPreferredHealthRequiresSnapshotObservation(t *testing.T) {
	now := inspectNow()
	source := preferredRuntimeSource{
		fakeCredentialSource: fakeCredentialSource{keys: []state.CredentialMeta{{ID: 21, GroupID: 2}}, configuredCounts: map[uint]int{1: 1}},
		views:                []state.CredentialRuntimeView{{ID: 21, GroupID: 2, CooldownUntil: now.Add(time.Minute)}},
	}
	iterator := newWithClock(twoGroupSnapshot(t), source, preferredHealthQuery(11), rand.New(zeroRandSource{}), func() time.Time { return now })
	selected, err := iterator.Next()
	if err != nil || selected.CredentialID != 21 || iterator.PreferredHealthBypassed() {
		t.Fatalf("Next = %d/%v, bypass = %v, want 21/nil/false without preferred Snapshot view", selected.CredentialID, err, iterator.PreferredHealthBypassed())
	}
}

// U005: the health observation must belong to the frozen bound identity. A
// same-ID credential replacement (new IdentityGeneration) that is itself
// cooled down must never attribute its health to the old binding, so no
// health bypass is reported for a bound generation that no longer exists.
func TestPreferredHealthBypassedRequiresBoundIdentityGeneration(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := twoGroupRegistry(t)
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 11, GroupID: 1, Version: 2, IdentityGeneration: 2, Fingerprint: "f11b", EncryptedValue: "c11b"},
		{ID: 21, GroupID: 2, Version: 1, IdentityGeneration: 1, Fingerprint: "f21", EncryptedValue: "c21"},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	registry.SetEntryCooldown(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now.Add(time.Minute))
	query := preferredHealthQuery(11)
	query.PreferredIdentityGeneration = 1
	iterator := New(twoGroupSnapshot(t), registry, query, rand.New(zeroRandSource{}))
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 21 {
		t.Fatalf("selected credential = %d, want sibling 21", selection.CredentialID)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true for replaced identity, want false")
	}
}

// The same cooldown on the bound identity itself is still a health bypass.
func TestPreferredHealthBypassedForBoundIdentityGeneration(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := twoGroupRegistry(t)
	registry.SetEntryCooldown(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, now.Add(time.Minute))
	query := preferredHealthQuery(11)
	query.PreferredIdentityGeneration = 1
	iterator := New(twoGroupSnapshot(t), registry, query, rand.New(zeroRandSource{}))
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 21 || !iterator.PreferredHealthBypassed() {
		t.Fatalf("selection = %d, bypassed = %v, want 21/true for bound identity in cooldown", selection.CredentialID, iterator.PreferredHealthBypassed())
	}
}
