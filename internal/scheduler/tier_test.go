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

func TestCombinedWeightUsesGroupEntryAndCredentialFactors(t *testing.T) {
	groupWeight, entryWeight, credentialWeight := 60, 30, 40
	if got, want := combinedWeight(&groupWeight, entryWeight, &credentialWeight, 99), int64(60*30*40); got != want {
		t.Fatalf("combinedWeight() = %d, want %d", got, want)
	}
	zero := 0
	for name, test := range map[string]struct {
		group      *int
		entry      int
		credential *int
	}{
		"group":      {group: &zero, entry: 30, credential: &credentialWeight},
		"entry":      {group: &groupWeight, entry: 0, credential: &credentialWeight},
		"credential": {group: &groupWeight, entry: 30, credential: &zero},
	} {
		t.Run(name, func(t *testing.T) {
			if got := combinedWeight(test.group, test.entry, test.credential, 1); got != 0 {
				t.Fatalf("combinedWeight() = %d, want zero", got)
			}
		})
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

func TestIteratorKeepsPriorityTiersSeparate(t *testing.T) {
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI,
		Params: json.RawMessage(`{}`), Enabled: true,
		Models: []state.ModelConfig{
			{ID: "primary-a", Alias: "public", Weight: intPointer(10), Priority: intPointer(1)},
			{ID: "primary-b", Alias: "public", Weight: intPointer(10), Priority: intPointer(1)},
			{ID: "fallback", Alias: "public", Weight: intPointer(100), Priority: intPointer(2)},
		},
	}})
	iterator := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}}, routeEntryQuery(), rand.New(zeroRandSource{}))

	for _, want := range []string{"primary-a", "primary-b", "fallback"} {
		selection, err := iterator.Next()
		if err != nil || selection.UpstreamModelID == nil || *selection.UpstreamModelID != want {
			t.Fatalf("Next() = (%#v, %v), want upstream %q", selection, err, want)
		}
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Next() after all tiers error = %v, want ErrExhausted", err)
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

func TestIteratorMatchesDesignBenchmarkAcrossGroupsAndPriorityFallback(t *testing.T) {
	snapshot := routeEntryBenchmarkSnapshot(t)
	source := fakeCredentialSource{keys: []state.CredentialMeta{
		{ID: 11, GroupID: 1, WeightAuto: state.DefaultWeight},
		{ID: 21, GroupID: 2, WeightAuto: state.DefaultWeight},
		{ID: 31, GroupID: 3, WeightAuto: state.DefaultWeight},
	}}
	query := routeEntryQuery()
	query.ExternalModel = modelPointer("A")
	random := rand.New(rand.NewSource(1))
	type target struct {
		group    uint
		upstream string
	}
	counts := make(map[target]int)
	const (
		primarySamples  = 8800
		fallbackSamples = 1200
		samples         = primarySamples + fallbackSamples
	)
	for range primarySamples {
		selection, err := New(snapshot, source, query, random).Next()
		if err != nil {
			t.Fatalf("benchmark P1 Next() error = %v", err)
		}
		if selection.UpstreamModelID == nil {
			t.Fatalf("benchmark selection has no upstream model: %#v", selection)
		}
		selected := target{group: selection.GroupID, upstream: *selection.UpstreamModelID}
		if selected.upstream == "C" {
			t.Fatalf("P2 target C was reachable while P1 was available: %#v", selection)
		}
		counts[selected]++
	}
	for range fallbackSamples {
		iterator := New(snapshot, source, query, random)
		for range 4 {
			selection, err := iterator.Next()
			if err != nil {
				t.Fatalf("benchmark P1 exhaustion Next() error = %v", err)
			}
			if selection.UpstreamModelID == nil || *selection.UpstreamModelID == "C" {
				t.Fatalf("P2 target became reachable before P1 exhaustion: %#v", selection)
			}
		}
		selection, err := iterator.Next()
		if err != nil {
			t.Fatalf("benchmark P2 fallback Next() error = %v", err)
		}
		if selection.UpstreamModelID == nil || *selection.UpstreamModelID != "C" {
			t.Fatalf("P2 fallback selection = %#v, want upstream C after P1 exhaustion", selection)
		}
		counts[target{group: selection.GroupID, upstream: *selection.UpstreamModelID}]++
	}

	want := []struct {
		name      string
		target    target
		expectPct float64
	}{
		{name: "G1/A", target: target{group: 1, upstream: "A"}, expectPct: 18},
		{name: "G1/B", target: target{group: 1, upstream: "B"}, expectPct: 30},
		{name: "G2/B", target: target{group: 2, upstream: "B"}, expectPct: 30},
		{name: "G3/D", target: target{group: 3, upstream: "D"}, expectPct: 10},
	}
	for _, test := range want {
		count := counts[test.target]
		percent := 100 * float64(count) / samples
		t.Logf("design benchmark %s: %d/%d (%.2f%%), expected %.0f%% +/- 3pp", test.name, count, samples, percent, test.expectPct)
		if percent < test.expectPct-3 || percent > test.expectPct+3 {
			t.Fatalf("design benchmark %s = %.2f%%, want %.0f%% +/- 3pp", test.name, percent, test.expectPct)
		}
	}
	fallbackCount := counts[target{group: 1, upstream: "C"}]
	t.Logf("design benchmark G1/C fallback: %d/%d (%.2f%%), P1 requests=%d, fallback requests=%d", fallbackCount, samples, 100*float64(fallbackCount)/samples, primarySamples, fallbackSamples)
	if fallbackCount != fallbackSamples {
		t.Fatalf("G1/C fallback count = %d, want %d", fallbackCount, fallbackSamples)
	}

	iterator := New(snapshot, source, query, rand.New(rand.NewSource(20260905)))
	firstFour := make([]string, 0, 4)
	for range 4 {
		selection, err := iterator.Next()
		if err != nil {
			t.Fatalf("P1 exhaustion Next() error = %v", err)
		}
		if selection.UpstreamModelID == nil || *selection.UpstreamModelID == "C" {
			t.Fatalf("P2 target became reachable before P1 exhaustion: %#v", selection)
		}
		firstFour = append(firstFour, *selection.UpstreamModelID)
	}
	fallback, err := iterator.Next()
	if err != nil {
		t.Fatalf("P2 fallback Next() error = %v", err)
	}
	if fallback.UpstreamModelID == nil || *fallback.UpstreamModelID != "C" {
		t.Fatalf("P2 fallback selection = %#v, want upstream C after P1 exhaustion", fallback)
	}
	t.Logf("priority fallback: first four P1 upstreams=%v, fifth selection=G%d/%s", firstFour, fallback.GroupID, *fallback.UpstreamModelID)
}

func TestZeroWeightEntryReportsInspectionAndIteratorReasons(t *testing.T) {
	zero := 0
	one := 1
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ConnectionType: "api_key", ID: 1, Name: "zero-entry", ChannelID: channel.OpenAI,
		Params: json.RawMessage(`{}`), Enabled: true,
		Models: []state.ModelConfig{
			{ID: "a-zero", Alias: "public", Weight: &zero, Priority: &one},
			{ID: "z-fallback", Alias: "public", Weight: &one, Priority: intPointer(2)},
		},
	}})
	query := routeEntryQuery()
	credentials := []state.CredentialRuntimeView{{
		ID: 11, GroupID: 1, Status: state.CredentialStatusActive,
		WeightAuto: state.DefaultWeight,
	}}
	inspection, err := Inspect(snapshot, credentials, query, inspectNow())
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	var zeroEntry GroupInspection
	foundZeroEntry := false
	for _, group := range inspection.Groups {
		if group.UpstreamModelID != nil && *group.UpstreamModelID == "a-zero" {
			zeroEntry = group
			foundZeroEntry = true
			break
		}
	}
	if !inspection.Routable || !foundZeroEntry {
		t.Fatalf("zero-entry inspection = %#v", inspection)
	}
	if zeroEntry.Routable || zeroEntry.Reason != ReasonEntryWeightZero ||
		len(zeroEntry.Credentials) != 1 || !zeroEntry.Credentials[0].Available ||
		zeroEntry.Credentials[0].EffectiveWeight != 0 {
		t.Fatalf("zero-entry inspection group = %#v", zeroEntry)
	}

	for index := range snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"] {
		snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"][index].EntryWeight = 0
	}
	iterator := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{
		ID: 11, GroupID: 1, WeightAuto: state.DefaultWeight,
	}}}, query, rand.New(zeroRandSource{}))
	if got := iterator.StaticReason(); got != ReasonEntryWeightZero {
		t.Fatalf("StaticReason() = %q, want %q", got, ReasonEntryWeightZero)
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("zero-entry Next() error = %v, want ErrExhausted", err)
	}
}

func routeEntryBenchmarkSnapshot(t *testing.T) *state.ConfigSnapshot {
	t.Helper()
	return routeEntrySnapshot(t, []state.GroupConfig{
		{
			ConnectionType: "api_key", ID: 1, Name: "G1", ChannelID: channel.OpenAI,
			Params: json.RawMessage(`{}`), WeightManual: intPointer(60), Enabled: true,
			Models: []state.ModelConfig{
				{ID: "A", Alias: "A", Weight: intPointer(30), Priority: intPointer(1)},
				{ID: "B", Alias: "A", Weight: intPointer(50), Priority: intPointer(1)},
				{ID: "C", Alias: "A", Weight: intPointer(20), Priority: intPointer(2)},
			},
		},
		{
			ConnectionType: "api_key", ID: 2, Name: "G2", ChannelID: channel.OpenAI,
			Params: json.RawMessage(`{}`), WeightManual: intPointer(30), Enabled: true,
			Models: []state.ModelConfig{
				{ID: "B", Alias: "A", Weight: intPointer(100), Priority: intPointer(1)},
			},
		},
		{
			ConnectionType: "api_key", ID: 3, Name: "G3", ChannelID: channel.OpenAI,
			Params: json.RawMessage(`{}`), WeightManual: intPointer(10), Enabled: true,
			Models: []state.ModelConfig{
				{ID: "D", Alias: "A", Weight: intPointer(100), Priority: intPointer(1)},
			},
		},
	})
}
func TestIteratorUpstreamMarginalIsIndependentOfCredentialCount(t *testing.T) {
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI,
		Params: json.RawMessage(`{}`), Enabled: true,
		Models: []state.ModelConfig{
			{ID: "light", Alias: "public", Weight: intPointer(30)},
			{ID: "heavy", Alias: "public", Weight: intPointer(70)},
		},
	}})
	countByCredentialNumber := func(keys []state.CredentialMeta) map[string]int {
		counts := map[string]int{}
		random := rand.New(rand.NewSource(17))
		for range 10000 {
			selection, err := New(snapshot, fakeCredentialSource{keys: keys}, routeEntryQuery(), random).Next()
			if err != nil {
				t.Fatalf("Next() error = %v", err)
			}
			counts[*selection.UpstreamModelID]++
		}
		return counts
	}
	one := countByCredentialNumber([]state.CredentialMeta{{ID: 11, GroupID: 1}})
	two := countByCredentialNumber([]state.CredentialMeta{{ID: 11, GroupID: 1}, {ID: 12, GroupID: 1}})
	oneRatio := float64(one["light"]) / float64(one["light"]+one["heavy"])
	twoRatio := float64(two["light"]) / float64(two["light"]+two["heavy"])
	t.Logf("credential-count marginal: one key light=%d (%.2f%%), heavy=%d (%.2f%%); two keys light=%d (%.2f%%), heavy=%d (%.2f%%)", one["light"], 100*oneRatio, one["heavy"], 100*(1-oneRatio), two["light"], 100*twoRatio, two["heavy"], 100*(1-twoRatio))
	if oneRatio < 0.27 || oneRatio > 0.33 || twoRatio < 0.27 || twoRatio > 0.33 {
		t.Fatalf("marginal counts = one key %#v (%.3f), two keys %#v (%.3f), want light about 30%%", one, oneRatio, two, twoRatio)
	}
	if difference := oneRatio - twoRatio; difference < -0.03 || difference > 0.03 {
		t.Fatalf("credential count changed upstream marginal: one=%.3f two=%.3f", oneRatio, twoRatio)
	}
}

func routeEntryQuery() Query {
	return Query{
		ClientProtocol: protocol.OpenAICompletions,
		Operation:      execution.OperationChatCompletion,
		ExternalModel:  modelPointer("public"),
		AccessKey:      state.AccessKeyView{Status: state.AccessKeyStatusActive},
	}
}

func routeEntrySnapshot(t *testing.T, groups []state.GroupConfig) *state.ConfigSnapshot {
	t.Helper()
	snapshot, err := state.Compile(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups:          groups,
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return snapshot
}

func intPointer(value int) *int {
	return &value
}
