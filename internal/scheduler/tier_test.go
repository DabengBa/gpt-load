package scheduler

import (
	"encoding/json"
	"errors"
	"math/rand"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

func routeEntryQuery() Query {
	return Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("public")}
}
func routeEntrySnapshot(_ *testing.T, groups []state.GroupConfig) *state.ConfigSnapshot {
	snapshot, err := state.Compile(state.CompileInput{ChannelRegistry: channel.NewRegistry(), Groups: groups})
	if err != nil {
		panic(err)
	}
	return snapshot
}
func intPointer(value int) *int { return &value }

func TestIteratorKeepsPriorityTiersSeparate(t *testing.T) {
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{ID: 1, Name: "group", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "primary", Alias: "public", Weight: intPointer(1), Priority: intPointer(1)}, {ID: "fallback", Alias: "public", Weight: intPointer(100), Priority: intPointer(2)}}}})
	selection, err := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}}, routeEntryQuery(), rand.New(zeroRandSource{})).Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.UpstreamModelID == nil || *selection.UpstreamModelID != "primary" {
		t.Fatalf("selected upstream = %v", selection.UpstreamModelID)
	}
}

func TestIteratorWeightedMixUsesEntryWeight(t *testing.T) {
	snapshot, err := state.Compile(state.CompileInput{ChannelRegistry: channel.NewRegistry(), SystemSettings: config.Settings{state.SettingRouteStrategy: string(state.RouteStrategyWeightedMix)}, Groups: []state.GroupConfig{{ID: 1, Name: "group", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: []byte(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "model", Alias: "public", Weight: intPointer(100)}}}}})
	if err != nil {
		t.Fatal(err)
	}
	selection, err := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 1, GroupID: 1}}}, routeEntryQuery(), rand.New(zeroRandSource{})).Next()
	if err != nil || selection.CredentialID != 1 {
		t.Fatalf("selection = %#v, err = %v", selection, err)
	}
}

func TestIteratorSupportsMultipleTargetsInOneGroupAndPairRetry(t *testing.T) {
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI,
		Params: json.RawMessage(`{}`), Enabled: true,
		Models: []state.ModelConfig{
			{ID: "upstream-a", Alias: "public", Weight: intPointer(30)},
			{ID: "upstream-b", Alias: "public", Weight: intPointer(70)},
		},
	}})
	iterator := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}}, routeEntryQuery(), rand.New(zeroRandSource{}))

	first, err := iterator.Next()
	if err != nil || first.CredentialID != 11 || first.UpstreamModelID == nil || *first.UpstreamModelID != "upstream-a" {
		t.Fatalf("first Next() = (%#v, %v), want credential 11 on upstream-a", first, err)
	}
	second, err := iterator.Next()
	if err != nil || second.CredentialID != 11 || second.UpstreamModelID == nil || *second.UpstreamModelID != "upstream-b" {
		t.Fatalf("second Next() = (%#v, %v), want same credential 11 on upstream-b", second, err)
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("third Next() error = %v, want ErrExhausted", err)
	}
}

func TestIteratorUsesCombinedWeightsWithinPriorityTier(t *testing.T) {
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI,
		Params: json.RawMessage(`{}`), Enabled: true,
		Models: []state.ModelConfig{
			{ID: "light", Alias: "public", Weight: intPointer(30)},
			{ID: "heavy", Alias: "public", Weight: intPointer(70)},
		},
	}})
	source := fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}}
	random := rand.New(rand.NewSource(20260905))
	counts := map[string]int{}
	for range 10000 {
		selection, err := New(snapshot, source, routeEntryQuery(), random).Next()
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		counts[*selection.UpstreamModelID]++
	}
	ratio := float64(counts["light"]) / float64(counts["heavy"])
	t.Logf("combined entry distribution: light=%d (%.2f%%), heavy=%d (%.2f%%), ratio=%.3f", counts["light"], 100*float64(counts["light"])/10000, counts["heavy"], 100*float64(counts["heavy"])/10000, ratio)
	if ratio < 0.39 || ratio > 0.47 {
		t.Fatalf("weighted counts = %#v, ratio = %.3f, want about 30:70", counts, ratio)
	}
}
