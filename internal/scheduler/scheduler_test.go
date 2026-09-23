package scheduler

import (
	"errors"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

func modelPointer(value string) *string { return &value }

type fakeCredentialSource struct {
	keys             []state.CredentialMeta
	configuredCounts map[uint]int
}

func (source fakeCredentialSource) CollectCredentialCandidates(groupIDs []uint, excluded func(uint) bool, _ time.Time) []state.CredentialMeta {
	allowed := make(map[uint]struct{}, len(groupIDs))
	for _, id := range groupIDs {
		allowed[id] = struct{}{}
	}
	result := make([]state.CredentialMeta, 0, len(source.keys))
	for _, key := range source.keys {
		if _, ok := allowed[key.GroupID]; !ok || excluded != nil && excluded(key.ID) {
			continue
		}
		result = append(result, key)
	}
	return result
}

func (source fakeCredentialSource) CredentialCountsByGroup(groupIDs []uint) map[uint]int {
	counts := make(map[uint]int, len(groupIDs))
	for _, key := range source.keys {
		counts[key.GroupID]++
	}
	for groupID, count := range source.configuredCounts {
		counts[groupID] = count
	}
	return counts
}

type zeroRandSource struct{}

func (zeroRandSource) Int63() int64 { return 0 }
func (zeroRandSource) Seed(int64)   {}

func TestIteratorRejectsMultiCredentialGroupWithoutSilentSelection(t *testing.T) {
	query := Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")}
	if _, err := New(schedulerSnapshot(), fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}, {ID: 12, GroupID: 1}}}, query, rand.New(zeroRandSource{})).Next(); err != ErrExhausted {
		t.Fatalf("Next() error = %v", err)
	}
}

func TestIteratorRejectsMultiCredentialGroupWithOneHealthyCredential(t *testing.T) {
	query := Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")}
	source := fakeCredentialSource{
		keys:             []state.CredentialMeta{{ID: 11, GroupID: 1}},
		configuredCounts: map[uint]int{1: 2},
	}
	if _, err := New(schedulerSnapshot(), source, query, rand.New(zeroRandSource{})).Next(); err != ErrExhausted {
		t.Fatalf("Next() error = %v, want %v", err, ErrExhausted)
	}
}

func TestIteratorUsesEntryWeightOnly(t *testing.T) {
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{
		{ID: 1, Name: "one", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "light", Alias: "public", Weight: intPointer(1)}}},
		{ID: 2, Name: "two", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "heavy", Alias: "public", Weight: intPointer(100)}}},
	})
	selection, err := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}, {ID: 21, GroupID: 2}}}, routeEntryQuery(), rand.New(zeroRandSource{})).Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 11 {
		t.Fatalf("selected credential = %d", selection.CredentialID)
	}
}

func schedulerSnapshot() *state.ConfigSnapshot {
	return routeEntrySnapshot(nil, []state.GroupConfig{
		{ID: 1, Name: "group-one", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "gpt-4o", Alias: "gpt-4o", Weight: intPointer(100)}}},
		{ID: 2, Name: "group-two", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "gpt-4o", Alias: "gpt-4o", Weight: intPointer(100)}}},
	})
}

func TestFilterTargetsAppliesAccessKeyDimensions(t *testing.T) {
	snapshot := schedulerSnapshot()
	tests := []struct {
		name       string
		protocol   protocol.Protocol
		model      string
		filters    state.FilterSet
		wantGroups []uint
	}{
		{name: "unrestricted", protocol: protocol.OpenAICompletions, model: "gpt-4o", wantGroups: []uint{1, 2}},
		{
			name:       "group filter",
			protocol:   protocol.OpenAICompletions,
			model:      "gpt-4o",
			filters:    state.FilterSet{Groups: map[uint]struct{}{2: {}}},
			wantGroups: []uint{2},
		},
		{
			name:       "protocol allowed",
			protocol:   protocol.OpenAICompletions,
			model:      "gpt-4o",
			filters:    state.FilterSet{Protocols: map[protocol.Protocol]struct{}{protocol.OpenAICompletions: {}}},
			wantGroups: []uint{1, 2},
		},
		{
			name:       "protocol denied",
			protocol:   protocol.OpenAICompletions,
			model:      "gpt-4o",
			filters:    state.FilterSet{Protocols: map[protocol.Protocol]struct{}{protocol.Anthropic: {}}},
			wantGroups: []uint{},
		},
		{
			name:       "model allowed",
			protocol:   protocol.OpenAICompletions,
			model:      "gpt-4o",
			filters:    state.FilterSet{Models: map[string]struct{}{"gpt-4o": {}}},
			wantGroups: []uint{1, 2},
		},
		{
			name:       "model denied",
			protocol:   protocol.OpenAICompletions,
			model:      "gpt-4o",
			filters:    state.FilterSet{Models: map[string]struct{}{"gpt-4o-mini": {}}},
			wantGroups: []uint{},
		},
		{name: "unknown model", protocol: protocol.OpenAICompletions, model: "missing", wantGroups: []uint{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			targets, _ := filterTargetsWithReason(snapshot, Query{
				ClientProtocol: tt.protocol, Operation: execution.OperationChatCompletion,
				ExternalModel: modelPointer(tt.model),
				AccessKey:     state.AccessKeyView{ID: 10, Filters: tt.filters},
			})
			got := make([]uint, 0, len(targets))
			for _, target := range targets {
				got = append(got, target.target.GroupID)
			}
			if !reflect.DeepEqual(got, tt.wantGroups) {
				t.Fatalf("groups = %#v, want %#v", got, tt.wantGroups)
			}
		})
	}
}

func TestFilterTargetsSkipsCandidateWithoutGroupView(t *testing.T) {
	snapshot := schedulerSnapshot()
	delete(snapshot.Groups, 2)
	got, _ := filterTargetsWithReason(
		snapshot,
		Query{
			ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion,
			ExternalModel: modelPointer("gpt-4o"),
		},
	)
	if len(got) != 1 || got[0].target.GroupID != 1 {
		t.Fatalf("targets = %#v, want only group 1", got)
	}
}

func TestIteratorSkipGroupExcludesWholeGroup(t *testing.T) {
	source := fakeCredentialSource{keys: []state.CredentialMeta{
		{ID: 11, GroupID: 1}, {ID: 21, GroupID: 2},
	}}
	iterator := New(schedulerSnapshot(), source,
		Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")},
		rand.New(zeroRandSource{}))
	first, err := iterator.Next()
	if err != nil || first.CredentialID != 11 {
		t.Fatalf("first Next() = (%#v, %v), want key 11", first, err)
	}
	iterator.SkipGroup(1)
	iterator.SkipGroup(1)
	second, err := iterator.Next()
	if err != nil || second.CredentialID != 21 {
		t.Fatalf("second Next() = (%#v, %v), want key 21", second, err)
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Next() after skip/exhaustion error = %v, want ErrExhausted", err)
	}
}

func TestIteratorSkipGroupIsRequestLocal(t *testing.T) {
	source := fakeCredentialSource{keys: []state.CredentialMeta{
		{ID: 11, GroupID: 1}, {ID: 21, GroupID: 2},
	}}
	query := Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")}
	first := New(schedulerSnapshot(), source, query, rand.New(zeroRandSource{}))
	first.SkipGroup(1)
	selection, err := first.Next()
	if err != nil || selection.GroupID != 2 {
		t.Fatalf("skipping iterator Next() = (%#v, %v), want group 2", selection, err)
	}
	second := New(schedulerSnapshot(), source, query, rand.New(zeroRandSource{}))
	selection, err = second.Next()
	if err != nil || selection.GroupID != 1 {
		t.Fatalf("fresh iterator Next() = (%#v, %v), want group 1", selection, err)
	}
}

func TestIteratorSkipGroupIgnoresNilReceiverAndZeroID(t *testing.T) {
	var nilIterator *Iterator
	nilIterator.SkipGroup(1)

	iterator := New(schedulerSnapshot(), fakeCredentialSource{keys: []state.CredentialMeta{
		{ID: 11, GroupID: 1}, {ID: 21, GroupID: 2},
	}}, Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")}, rand.New(zeroRandSource{}))
	iterator.SkipGroup(0)
	selection, err := iterator.Next()
	if err != nil || selection.GroupID != 1 {
		t.Fatalf("Next() after SkipGroup(0) = (%#v, %v), want group 1", selection, err)
	}
}

func TestIteratorNextNeverRepeatsAndExhausts(t *testing.T) {
	source := fakeCredentialSource{keys: []state.CredentialMeta{
		{ID: 11, GroupID: 1},
		{ID: 21, GroupID: 2},
	}}
	iterator := New(
		schedulerSnapshot(),
		source,
		Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")},
		rand.New(rand.NewSource(7)),
	)

	seen := make(map[uint]struct{})
	for range 2 {
		selection, err := iterator.Next()
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if _, duplicate := seen[selection.CredentialID]; duplicate {
			t.Fatalf("key %d selected twice", selection.CredentialID)
		}
		seen[selection.CredentialID] = struct{}{}
		if selection.Group.ID != selection.GroupID ||
			selection.UpstreamModelID == nil ||
			*selection.UpstreamModelID == "" {
			t.Fatalf("invalid selection: %#v", selection)
		}
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Next() after pool exhaustion error = %v, want ErrExhausted", err)
	}
}

func TestIteratorUsesDefaultWeights(t *testing.T) {
	iterator := New(
		schedulerSnapshot(),
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")},
		rand.New(rand.NewSource(1)),
	)
	selection, err := iterator.Next()
	if err != nil || selection.CredentialID != 11 {
		t.Fatalf("Next() with default weights = (%#v, %v), want key 11", selection, err)
	}
}

func TestIteratorPropertyNeverEscapesAccessFilters(t *testing.T) {
	snapshot := schedulerSnapshot()
	source := fakeCredentialSource{keys: []state.CredentialMeta{
		{ID: 11, GroupID: 1},
		{ID: 21, GroupID: 2},
		{ID: 31, GroupID: 3},
	}}
	generator := rand.New(rand.NewSource(20260717))

	for caseIndex := range 300 {
		allowedGroup := uint(generator.Intn(2) + 1)
		filters := state.FilterSet{}
		if generator.Intn(2) == 1 {
			filters.Groups = map[uint]struct{}{allowedGroup: {}}
		}
		if generator.Intn(2) == 1 {
			filters.Protocols = map[protocol.Protocol]struct{}{protocol.OpenAICompletions: {}}
		}
		if generator.Intn(2) == 1 {
			filters.Models = map[string]struct{}{"gpt-4o": {}}
		}
		query := Query{
			ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion,
			ExternalModel: modelPointer("gpt-4o"),
			AccessKey:     state.AccessKeyView{ID: uint(caseIndex + 1), Filters: filters},
		}
		frozenGroups := make(map[uint]struct{})
		targets, _ := filterTargetsWithReason(snapshot, query)
		for _, target := range targets {
			frozenGroups[target.target.GroupID] = struct{}{}
		}
		iterator := New(snapshot, source, query, rand.New(rand.NewSource(int64(caseIndex+1))))

		skipped := make(map[uint]struct{})
		for {
			selection, err := iterator.Next()
			if errors.Is(err, ErrExhausted) {
				break
			}
			if err != nil {
				t.Fatalf("case %d Next() error = %v", caseIndex, err)
			}
			if _, blocked := skipped[selection.GroupID]; blocked {
				t.Fatalf("case %d selected skipped group %d", caseIndex, selection.GroupID)
			}
			if _, ok := frozenGroups[selection.GroupID]; !ok {
				t.Fatalf("case %d selection %#v escaped frozen target groups %#v", caseIndex, selection, frozenGroups)
			}
			if len(filters.Groups) > 0 {
				if _, ok := filters.Groups[selection.GroupID]; !ok {
					t.Fatalf("case %d selection %#v escaped group filter %#v", caseIndex, selection, filters.Groups)
				}
			}
			if selection.UpstreamModelID == nil ||
				*selection.UpstreamModelID == "" ||
				selection.GroupID == 0 {
				t.Fatalf("case %d invalid selection %#v", caseIndex, selection)
			}
			if generator.Intn(2) == 1 {
				skipped[selection.GroupID] = struct{}{}
				iterator.SkipGroup(selection.GroupID)
			}
		}
	}
}

func TestIteratorExhaustsForNilOrEmptyDependencies(t *testing.T) {
	tests := []struct {
		name     string
		iterator *Iterator
	}{
		{name: "nil snapshot", iterator: New(nil, fakeCredentialSource{}, Query{}, rand.New(rand.NewSource(1)))},
		{name: "nil key source", iterator: New(schedulerSnapshot(), nil, Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")}, rand.New(rand.NewSource(1)))},
		{name: "nil random", iterator: New(schedulerSnapshot(), fakeCredentialSource{}, Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")}, nil)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.iterator.Next(); !errors.Is(err, ErrExhausted) {
				t.Fatalf("Next() error = %v, want ErrExhausted", err)
			}
		})
	}
}

func TestNormalizeQueryDoesNotDefaultOpenAIImagesOperation(t *testing.T) {
	t.Parallel()

	normalized := normalizeQuery(Query{
		ClientProtocol: protocol.OpenAIImages,
		ExternalModel:  modelPointer("gpt-image-2"),
	})
	if normalized.operation != "" {
		t.Fatalf("operation = %q, want empty", normalized.operation)
	}
}
