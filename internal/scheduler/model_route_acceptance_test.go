package scheduler

import (
	"encoding/json"
	"errors"
	"math/rand"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

// TestModelRouteEntriesAcceptance exercises the complete §1.2 benchmark with
// independent samples and a deterministic PRNG. Group 2 and 3 deliberately set
// weight explicitly: nil means the compatibility default of one.
func TestModelRouteEntriesAcceptance(t *testing.T) {
	groups := []state.GroupConfig{
		{ID: 1, Name: "openai", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: json.RawMessage(`{}`), Enabled: true, WeightManual: mrInt(60), Models: []state.ModelConfig{{ID: "A", Alias: "A", Weight: mrInt(30), Priority: mrInt(1)}, {ID: "B", Alias: "A", Weight: mrInt(50), Priority: mrInt(1)}, {ID: "C", Alias: "A", Weight: mrInt(20), Priority: mrInt(2)}}},
		{ID: 2, Name: "claude", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: json.RawMessage(`{}`), Enabled: true, WeightManual: mrInt(30), Models: []state.ModelConfig{{ID: "B", Alias: "A", Weight: mrInt(100), Priority: mrInt(1)}}},
		{ID: 3, Name: "gemini", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: json.RawMessage(`{}`), Enabled: true, WeightManual: mrInt(10), Models: []state.ModelConfig{{ID: "D", Alias: "A", Weight: mrInt(100), Priority: mrInt(1)}}},
	}
	snapshot, err := state.Compile(state.CompileInput{ChannelRegistry: channel.NewRegistry(), Groups: groups})
	if err != nil {
		t.Fatal(err)
	}
	query := Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: mrString("A"), AccessKey: state.AccessKeyView{Status: state.AccessKeyStatusActive}}
	source := fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1, WeightAuto: state.DefaultWeight}, {ID: 21, GroupID: 2, WeightAuto: state.DefaultWeight}, {ID: 31, GroupID: 3, WeightAuto: state.DefaultWeight}}}
	type target struct {
		group uint
		model string
	}
	counts := map[target]int{}
	rng := rand.New(rand.NewSource(20260905))
	const (
		primarySamples  = 8800
		fallbackSamples = 1200
		samples         = primarySamples + fallbackSamples
	)
	for range primarySamples {
		sel, err := New(snapshot, source, query, rng).Next()
		if err != nil {
			t.Fatal(err)
		}
		if sel.UpstreamModelID == nil || *sel.UpstreamModelID == "C" {
			t.Fatalf("P2 reached while P1 available: %#v", sel)
		}
		counts[target{sel.GroupID, *sel.UpstreamModelID}]++
	}
	for range fallbackSamples {
		it := New(snapshot, source, query, rng)
		for range 4 {
			sel, err := it.Next()
			if err != nil || sel.UpstreamModelID == nil || *sel.UpstreamModelID == "C" {
				t.Fatalf("P1 exhaustion step = %#v, %v", sel, err)
			}
		}
		sel, err := it.Next()
		if err != nil || sel.UpstreamModelID == nil || *sel.UpstreamModelID != "C" {
			t.Fatalf("fallback = %#v, %v", sel, err)
		}
		counts[target{sel.GroupID, *sel.UpstreamModelID}]++
	}
	if counts[target{1, "C"}] != fallbackSamples {
		t.Fatalf("C fallback count=%d, want %d", counts[target{1, "C"}], fallbackSamples)
	}
	for _, want := range []struct {
		name string
		key  target
		pct  float64
	}{{"G1/A", target{1, "A"}, 18}, {"G1/B", target{1, "B"}, 30}, {"G2/B", target{2, "B"}, 30}, {"G3/D", target{3, "D"}, 10}} {
		pct := 100 * float64(counts[want.key]) / samples
		t.Logf("%s sample count=%d/%d (%.2f%%), want %.0f%% +/- 3pp", want.name, counts[want.key], samples, pct, want.pct)
		if pct < want.pct-3 || pct > want.pct+3 {
			t.Fatalf("%s = %.2f%%", want.name, pct)
		}
	}
	it := New(snapshot, source, query, rand.New(rand.NewSource(7)))
	for i := 0; i < 4; i++ {
		sel, err := it.Next()
		if err != nil || sel.UpstreamModelID == nil || *sel.UpstreamModelID == "C" {
			t.Fatalf("P1 step %d = %#v, %v", i, sel, err)
		}
	}
	fallback, err := it.Next()
	if err != nil || fallback.UpstreamModelID == nil || *fallback.UpstreamModelID != "C" {
		t.Fatalf("fallback = %#v, %v", fallback, err)
	}
	if _, err := it.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("after exhaustion = %v", err)
	}
}

func mrInt(v int) *int          { return &v }
func mrString(v string) *string { return &v }
