package scheduler

import (
	"errors"
	"math/rand"
	"slices"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

type routeStrategyRandSource int64

func (source routeStrategyRandSource) Int63() int64 { return int64(source) }
func (routeStrategyRandSource) Seed(int64)          {}

func setSnapshotRouteStrategy(t *testing.T, snapshot *state.ConfigSnapshot, strategy string) {
	t.Helper()
	settings, err := state.ResolveRuntimeSettings(config.Settings{"route_strategy": strategy})
	if err != nil {
		t.Fatalf("ResolveRuntimeSettings() error = %v", err)
	}
	snapshot.Settings = settings
}

func TestIteratorRouteStrategyUsesEntryWeightsAcrossModes(t *testing.T) {
	for _, test := range []struct {
		name          string
		strategy      string
		tickets       int64
		wantConverted int
	}{
		{name: "native first", strategy: "native_first", tickets: 2, wantConverted: 0},
		{name: "weighted mix", strategy: "weighted_mix", tickets: 100, wantConverted: 50},
	} {
		t.Run(test.name, func(t *testing.T) {
			snapshot := channelSchedulerSnapshot(t)
			setSnapshotRouteStrategy(t, snapshot, test.strategy)
			source := fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}, {ID: 21, GroupID: 2}}}
			converted := 0
			for ticket := range test.tickets {
				selection, err := New(snapshot, source, Query{ClientProtocol: protocol.OpenAICompletions, ExternalModel: modelPointer("public")}, rand.New(routeStrategyRandSource(ticket))).Next()
				if err != nil {
					t.Fatalf("ticket %d Next() error = %v", ticket, err)
				}
				if selection.CredentialID == 11 {
					converted++
				}
			}
			if converted != test.wantConverted {
				t.Fatalf("converted selections = %d/%d, want %d", converted, test.tickets, test.wantConverted)
			}
		})
	}
}

func TestIteratorWeightedMixFreezesStrategyAndPrefersEligibleConvertedCredential(t *testing.T) {
	snapshot := channelSchedulerSnapshot(t)
	setSnapshotRouteStrategy(t, snapshot, "weighted_mix")
	iterator := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}, {ID: 21, GroupID: 2}}}, Query{
		ClientProtocol: protocol.OpenAICompletions, ExternalModel: modelPointer("public"), PreferredCredentialID: 11,
	}, rand.New(routeStrategyRandSource(2500)))
	setSnapshotRouteStrategy(t, snapshot, "native_first")
	for index, want := range []uint{11, 21} {
		selection, err := iterator.Next()
		if err != nil || selection.CredentialID != want {
			t.Fatalf("Next() %d = (%#v, %v), want credential %d", index, selection, err, want)
		}
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Next() after exhaustion error = %v", err)
	}
}

func TestIteratorWeightedMixPreservesStoredResponsesPriority(t *testing.T) {
	for _, preferred := range []uint{0, 31} {
		snapshot := responsesStoreSchedulerSnapshot(t, true)
		setSnapshotRouteStrategy(t, snapshot, "weighted_mix")
		iterator := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}, {ID: 21, GroupID: 2}, {ID: 31, GroupID: 3}, {ID: 41, GroupID: 4}}}, Query{
			ClientProtocol: protocol.OpenAIResponses, Operation: execution.OperationResponsesCreate,
			ResponsesStorePreference: execution.ResponsesStorePreferencePreferStored, ExternalModel: modelPointer("gpt"), PreferredCredentialID: preferred,
		}, rand.New(routeStrategyRandSource(50)))
		for index := range []uint{21, 31} {
			selection, err := iterator.Next()
			if err != nil || selection.ResponsesStoreDowngraded != (index > 0) {
				t.Fatalf("preferred %d Next() %d = (%#v, %v), want stable store semantics", preferred, index, selection, err)
			}
		}
	}
}

func TestIteratorWeightedMixKeepsNativeRequirement(t *testing.T) {
	snapshot := responsesStoreSchedulerSnapshot(t, true)
	setSnapshotRouteStrategy(t, snapshot, "weighted_mix")
	query := Query{ClientProtocol: protocol.OpenAIResponses, Operation: execution.OperationResponsesCreate, RouteRequirement: execution.RouteRequirementNative, ExternalModel: modelPointer("gpt"), PreferredCredentialID: 31}
	if got := CandidateGroupIDsForQuery(snapshot, query); !slices.Equal(got, []uint{2}) {
		t.Fatalf("CandidateGroupIDsForQuery() = %#v, want lifecycle-capable native group [2]", got)
	}
	iterator := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}, {ID: 21, GroupID: 2}, {ID: 31, GroupID: 3}, {ID: 41, GroupID: 4}}}, query, rand.New(zeroRandSource{}))
	selection, err := iterator.Next()
	if err != nil || selection.CredentialID != 21 || selection.RouteMode != channel.RouteNative {
		t.Fatalf("Next() = (%#v, %v), want lifecycle-capable native credential 21", selection, err)
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Next() after native exhaustion error = %v", err)
	}
}

func TestIteratorWeightedMixHonorsLiveHealthAndRequestExclusions(t *testing.T) {
	now := time.Unix(100, 0)
	snapshot := channelSchedulerSnapshot(t)
	setSnapshotRouteStrategy(t, snapshot, "weighted_mix")
	registry := state.NewCredentialRegistry()
	entries := []state.CredentialEntry{
		{ID: 11, GroupID: 1, AuthState: state.CredentialAuthStateReady, Version: 1, IdentityGeneration: 1, Fingerprint: "fp", EncryptedValue: "one"},
		{ID: 21, GroupID: 2, AuthState: state.CredentialAuthStateReady, Version: 1, IdentityGeneration: 1, Fingerprint: "fp", EncryptedValue: "two"},
	}
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	allowed := map[uint]struct{}{11: {}, 21: {}}
	iterator := newWithClock(snapshot, registry, Query{ClientProtocol: protocol.OpenAICompletions, ExternalModel: modelPointer("public"), AllowedCredentialIDs: allowed}, rand.New(zeroRandSource{}), func() time.Time { return now })
	allowed[99] = struct{}{}
	for index, want := range []uint{11, 21} {
		selection, err := iterator.Next()
		if err != nil || selection.CredentialID != want {
			t.Fatalf("Next() %d = (%#v, %v), want credential %d", index, selection, err, want)
		}
		if index == 0 && !registry.SetCooldown(11, now.Add(time.Minute)) {
			t.Fatal("SetCooldown() did not find converted credential")
		}
	}
	iterator.SkipGroup(2)
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Next() after skip and exclusions error = %v", err)
	}
}
