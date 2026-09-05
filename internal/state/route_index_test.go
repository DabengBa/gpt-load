package state

import (
	"encoding/json"
	"reflect"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

// TestSortExecutionRouteIndexOrdersByPriorityGroupAndUpstream locks the
// snapshot determinism rule of design §4: targets under one external model
// sort by (Priority, GroupID, UpstreamModelID) ascending.
func TestSortExecutionRouteIndexOrdersByPriorityGroupAndUpstream(t *testing.T) {
	t.Parallel()

	index := ExecutionCandidateIndex{
		protocol.OpenAICompletions: {
			execution.OperationChatCompletion: {
				"shared": []RouteTarget{
					{GroupID: 2, UpstreamModelID: "zeta", Mode: channel.RouteNative, EntryWeight: 40, Priority: 1},
					{GroupID: 1, UpstreamModelID: "fallback", Mode: channel.RouteNative, EntryWeight: 20, Priority: 2},
					{GroupID: 1, UpstreamModelID: "beta", Mode: channel.RouteNative, EntryWeight: 10, Priority: 1},
					{GroupID: 2, UpstreamModelID: "alpha", Mode: channel.RouteNative, EntryWeight: 30, Priority: 1},
				},
			},
		},
	}

	sortExecutionRouteIndex(index)

	got := index[protocol.OpenAICompletions][execution.OperationChatCompletion]["shared"]
	want := []RouteTarget{
		{GroupID: 1, UpstreamModelID: "beta", Mode: channel.RouteNative, EntryWeight: 10, Priority: 1},
		{GroupID: 2, UpstreamModelID: "alpha", Mode: channel.RouteNative, EntryWeight: 30, Priority: 1},
		{GroupID: 2, UpstreamModelID: "zeta", Mode: channel.RouteNative, EntryWeight: 40, Priority: 1},
		{GroupID: 1, UpstreamModelID: "fallback", Mode: channel.RouteNative, EntryWeight: 20, Priority: 2},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sorted targets = %#v, want %#v", got, want)
	}
}

// TestSortExecutionRouteIndexIsStableForEqualKeys keeps compile-order output
// for targets whose sort keys tie, so repeated compiles stay reproducible.
func TestSortExecutionRouteIndexIsStableForEqualKeys(t *testing.T) {
	t.Parallel()

	first := RouteTarget{GroupID: 1, UpstreamModelID: "same", Mode: channel.RouteNative, EntryWeight: 1, Priority: 1}
	second := RouteTarget{GroupID: 1, UpstreamModelID: "same", Mode: channel.RouteConverted, EntryWeight: 1, Priority: 1}
	index := ExecutionCandidateIndex{
		protocol.OpenAICompletions: {
			execution.OperationChatCompletion: {
				"shared": []RouteTarget{first, second},
			},
		},
	}

	sortExecutionRouteIndex(index)

	got := index[protocol.OpenAICompletions][execution.OperationChatCompletion]["shared"]
	if len(got) != 2 || got[0].Mode != channel.RouteNative || got[1].Mode != channel.RouteConverted {
		t.Fatalf("stable sort reordered equal keys: %#v", got)
	}
}

// TestCompileRouteIndexStructure verifies the compiled index shape for a
// multi-mapping benchmark: several entries per group share one external name,
// entries spread over groups interleave by priority, every (group, upstream)
// pair appears exactly once, and repeated compiles are byte-identical.
func TestCompileRouteIndexStructure(t *testing.T) {
	t.Parallel()

	newInput := func() CompileInput {
		return CompileInput{
			ChannelRegistry: channel.NewRegistry(),
			Groups: []GroupConfig{
				{ConnectionType: "api_key", ID: 2, Name: "two", ChannelID: channel.OpenAI,
					Params: json.RawMessage(`{}`), Enabled: true,
					Models: []ModelConfig{
						{ID: "zeta", Alias: "shared", Weight: intPointer(40)},
						{ID: "alpha", Alias: "shared", Weight: intPointer(30)},
					}},
				{ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAICompatible,
					Params: json.RawMessage(`{"base_url":"https://proxy.example/v1"}`), Enabled: true,
					Models: []ModelConfig{
						{ID: "fallback", Alias: "shared", Weight: intPointer(20), Priority: intPointer(2)},
						{ID: "solo"},
					}},
			},
		}
	}

	first, err := Compile(newInput())
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	second, err := Compile(newInput())
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if !reflect.DeepEqual(first.ExecutionCandidates, second.ExecutionCandidates) ||
		!reflect.DeepEqual(first.ExecutionRouteCatalog, second.ExecutionRouteCatalog) {
		t.Fatal("repeated compiles produced different route indexes")
	}

	shared := first.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["shared"]
	want := []struct {
		groupID         uint
		upstreamModelID string
		entryWeight     int
		priority        int
	}{
		{groupID: 2, upstreamModelID: "alpha", entryWeight: 30, priority: 1},
		{groupID: 2, upstreamModelID: "zeta", entryWeight: 40, priority: 1},
		{groupID: 1, upstreamModelID: "fallback", entryWeight: 20, priority: 2},
	}
	if len(shared) != len(want) {
		t.Fatalf("shared targets = %#v, want %d entries", shared, len(want))
	}
	for i, expected := range want {
		got := shared[i]
		if got.GroupID != expected.groupID || got.UpstreamModelID != expected.upstreamModelID ||
			got.EntryWeight != expected.entryWeight || got.Priority != expected.priority {
			t.Fatalf("shared targets[%d] = %#v, want %#v", i, got, expected)
		}
	}
	if !reflect.DeepEqual(shared, first.ExecutionRouteCatalog[protocol.OpenAICompletions][execution.OperationChatCompletion]["shared"]) {
		t.Fatalf("catalog targets diverge from candidates: %#v vs %#v", shared,
			first.ExecutionRouteCatalog[protocol.OpenAICompletions][execution.OperationChatCompletion]["shared"])
	}

	// Every (group, upstream) pair must appear exactly once per index key.
	seen := make(map[struct {
		groupID         uint
		upstreamModelID string
	}]struct{}, len(shared))
	for _, target := range shared {
		key := struct {
			groupID         uint
			upstreamModelID string
		}{target.GroupID, target.UpstreamModelID}
		if _, duplicate := seen[key]; duplicate {
			t.Fatalf("duplicate route target for %#v", key)
		}
		seen[key] = struct{}{}
	}

	solo := first.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["solo"]
	if len(solo) != 1 || solo[0].GroupID != 1 || solo[0].UpstreamModelID != "solo" ||
		solo[0].EntryWeight != 1 || solo[0].Priority != 1 {
		t.Fatalf("solo targets = %#v", solo)
	}
}

// TestCompileOrdersNoModelRouteKeyTargetsByGroup keeps resource-operation
// targets (Responses retrieve/delete/passthrough) grouped by ascending group
// ID so affinity resolution stays deterministic.
func TestCompileOrdersNoModelRouteKeyTargetsByGroup(t *testing.T) {
	t.Parallel()

	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{
			{ConnectionType: "api_key", ID: 3, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "model-c"}}, Enabled: true},
			{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "model-a"}}, Enabled: true},
		},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	targets := snapshot.ExecutionCandidates[protocol.OpenAIResponses][execution.OperationResponsesRetrieve][NoModelRouteKey]
	if len(targets) != 2 || targets[0].GroupID != 1 || targets[1].GroupID != 3 {
		t.Fatalf("NoModelRouteKey targets = %#v, want group order [1 3]", targets)
	}
}
