package state

import (
	"encoding/json"
	"net/netip"
	"reflect"
	"strings"
	"testing"

	"gpt-load/internal/accessquota"
	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
)

func TestCompileIndexesExternalModelsAndPreservesUpstreamIDs(t *testing.T) {
	t.Parallel()

	input := CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{
			{ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{
					{ID: "provider-a", Alias: "public"},
					{ID: "provider-a", Alias: "secondary"},
					{ID: "plain"},
				},
				Enabled: true,
			},
			{ConnectionType: "api_key", ID: 2, Name: "two", ChannelID: channel.OpenAICompatible,
				Params:  json.RawMessage(`{"base_url":"https://proxy.example/v1"}`),
				Models:  []ModelConfig{{ID: "provider-b", Alias: "public"}},
				Enabled: true,
			},
		},
	}

	snapshot, err := Compile(input)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	index := snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]
	public := index["public"]
	if len(public) != 2 || public[0].UpstreamModelID != "provider-a" || public[1].UpstreamModelID != "provider-b" {
		t.Fatalf("public targets = %#v", public)
	}
	if got := index["secondary"]; len(got) != 1 || got[0].UpstreamModelID != "provider-a" {
		t.Fatalf("secondary targets = %#v", got)
	}
	if got := index["plain"]; len(got) != 1 || got[0].UpstreamModelID != "plain" {
		t.Fatalf("plain targets = %#v", got)
	}
	if _, exists := index["provider-a"]; exists {
		t.Fatal("aliased upstream id entered external index")
	}
}

func TestCompileIndexesOpenAIImagesOperationsForAllConfiguredModels(t *testing.T) {
	t.Parallel()

	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{
			{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "official-image", Alias: "public"}}, Enabled: true},
			{ConnectionType: "api_key", ID: 2, ChannelID: channel.OpenAICompatible,
				Params: json.RawMessage(`{"base_url":"https://proxy.example/api/v4"}`),
				Models: []ModelConfig{{ID: "compatible-image", Alias: "public"}}, Enabled: true},
		},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	for _, operation := range []execution.Operation{execution.OperationImagesGenerate, execution.OperationImagesEdit} {
		got := snapshot.ExecutionCandidates[protocol.OpenAIImages][operation]["public"]
		if len(got) != 2 || got[0].GroupID != 1 || got[1].GroupID != 2 ||
			got[0].Mode != channel.RouteNative || got[1].Mode != channel.RouteNative {
			t.Errorf("Images candidates for %q = %#v", operation, got)
		}
	}
}

func TestCompileIndexesOpenAIEmbeddingsForAllConfiguredModels(t *testing.T) {
	t.Parallel()

	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{
			{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "text-embedding-3-small", Alias: "public"}}, Enabled: true},
			{ConnectionType: "api_key", ID: 2, ChannelID: channel.OpenAICompatible,
				Params: json.RawMessage(`{"base_url":"https://proxy.example/api/v4"}`),
				Models: []ModelConfig{{ID: "Qwen/Qwen3-Embedding-8B", Alias: "public"}}, Enabled: true},
			{ConnectionType: "api_key", ID: 3, ChannelID: channel.OpenRouter, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "openai/text-embedding-3-small", Alias: "public"}}, Enabled: true},
		},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	got := snapshot.ExecutionCandidates[protocol.OpenAIEmbeddings][execution.OperationEmbeddingsCreate]["public"]
	if len(got) != 3 || got[0].GroupID != 1 || got[1].GroupID != 2 || got[2].GroupID != 3 {
		t.Fatalf("Embeddings candidates = %#v", got)
	}
	for _, target := range got {
		if target.Mode != channel.RouteNative {
			t.Fatalf("Embeddings target = %#v, want native", target)
		}
	}
}

func TestCompileSubscriptionPublishesOnlyVerifiedCodexOperations(t *testing.T) {
	t.Parallel()
	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{{
			ID: 1, Name: "subscription", ChannelID: channel.Codex, ConnectionType: "subscription",
			Params: json.RawMessage(`{}`), Models: []ModelConfig{{ID: "gpt-5", Alias: "public"}}, Enabled: true,
		}},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if got := snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"]; len(got) != 1 {
		t.Fatalf("chat targets = %#v", got)
	}
	if got := snapshot.ExecutionCandidates[protocol.OpenAIResponses][execution.OperationResponsesCreate]["public"]; len(got) != 1 {
		t.Fatalf("responses create targets = %#v", got)
	}
	if got := snapshot.ExecutionCandidates[protocol.OpenAIResponses][execution.OperationResponsesInputTokens]["public"]; len(got) != 1 {
		t.Fatalf("responses input token targets = %#v", got)
	}
	if got := snapshot.ExecutionCandidates[protocol.Anthropic][execution.OperationCountTokens]["public"]; len(got) != 1 {
		t.Fatalf("Anthropic count token targets = %#v", got)
	}
	if got := snapshot.ExecutionCandidates[protocol.Gemini][execution.OperationCountTokens]["public"]; len(got) != 1 {
		t.Fatalf("Gemini count token targets = %#v", got)
	}
	if got := snapshot.ExecutionCandidates[protocol.OpenAIEmbeddings]; len(got) != 0 {
		t.Fatalf("subscription Embeddings targets = %#v, want none", got)
	}
	for _, operation := range []execution.Operation{
		execution.OperationResponsesRetrieve, execution.OperationResponsesDelete,
		execution.OperationResponsesCancel, execution.OperationResponsesCompact,
		execution.OperationResponsesInputItems,
		execution.OperationResponsesPassthrough,
	} {
		if got := snapshot.ExecutionCandidates[protocol.OpenAIResponses][operation]; len(got) != 0 {
			t.Fatalf("unsupported subscription operation %q was published: %#v", operation, got)
		}
	}
}

func TestCompileBuildsManagementCatalogsWithoutChangingActiveIndexes(t *testing.T) {
	t.Parallel()

	input := CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{
			{ConnectionType: "api_key", ID: 2, Name: "disabled", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "provider-disabled", Alias: "public"}},
			},
			{ConnectionType: "api_key", ID: 1, Name: "active", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "provider-active", Alias: "public"}}, Enabled: true,
			},
		},
		AccessKeys: []AccessKeyConfig{
			{ID: 11, Name: "active-client", KeyHash: "active-hash", Status: AccessKeyStatusActive, Filters: FilterSet{Groups: map[uint]struct{}{1: {}}}, RPMLimit: 10},
			{ID: 12, Name: "disabled-client", KeyHash: "disabled-hash", Status: AccessKeyStatusDisabled, Filters: FilterSet{Models: map[string]struct{}{"public": {}}}, RPMLimit: 20},
		},
	}

	snapshot, err := Compile(input)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if len(snapshot.Groups) != 1 || snapshot.Groups[1].Name != "active" {
		t.Fatalf("active Groups = %#v", snapshot.Groups)
	}
	active := snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"]
	if len(active) != 1 || active[0].GroupID != 1 {
		t.Fatalf("active candidates = %#v", active)
	}
	routes := snapshot.ExecutionRouteCatalog[protocol.OpenAICompletions][execution.OperationChatCompletion]["public"]
	if len(routes) != 2 || routes[0].GroupID != 1 || routes[1].GroupID != 2 {
		t.Fatalf("route catalog = %#v", routes)
	}
	if got := snapshot.GroupCatalog[2]; got.Enabled {
		t.Fatalf("disabled group catalog = %#v", got)
	}
	if _, ok := snapshot.AccessKeysByHash["disabled-hash"]; ok {
		t.Fatal("disabled access key entered active hash index")
	}
	if got := snapshot.AccessKeysByID[12]; got.Status != AccessKeyStatusDisabled || got.RPMLimit != 20 {
		t.Fatalf("disabled access key catalog = %#v", got)
	}
}

func TestCompileCarriesSettingsAndValidationModel(t *testing.T) {
	t.Parallel()

	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		SystemSettings:  config.Settings{"first_byte_timeout": json.Number("20")},
		Groups: []GroupConfig{{ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
			ValidationModel: "  probe-model  ",
			Models:          []ModelConfig{{ID: "real-model", Alias: "public-model"}},
			Settings:        config.Settings{"request_timeout": json.Number("30")}, Enabled: true,
		}},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	view := snapshot.Groups[1]
	if view.ValidationModel != "probe-model" || view.Timeouts.FirstByte.Seconds() != 20 || view.Timeouts.Request.Seconds() != 30 {
		t.Fatalf("group runtime view = %#v", view)
	}
}

func TestCompileOwnsInputData(t *testing.T) {
	t.Parallel()

	expiresAtMS := int64(1_900_000_000_000)
	filters := FilterSet{
		Groups:    map[uint]struct{}{1: {}},
		Protocols: map[protocol.Protocol]struct{}{protocol.OpenAICompletions: {}},
		Models:    map[string]struct{}{"public": {}},
	}
	input := CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
			Models: []ModelConfig{{ID: "upstream", Alias: "public"}}, Enabled: true,
		}},
		AccessKeys: []AccessKeyConfig{{
			ID: 1, KeyHash: "hash", Status: AccessKeyStatusActive, Filters: filters,
			ExpiresAtMS: &expiresAtMS,
			AllowedPeerCIDRs: []netip.Prefix{
				netip.MustParsePrefix("192.0.2.0/24"),
			},
			CostLimitRules: []accessquota.Rule{{
				ID: 7, Revision: 1, Kind: accessquota.KindTotal, LimitNanoUSD: 100,
			}},
		}},
	}
	snapshot, err := Compile(input)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	input.Groups[0].Models[0] = ModelConfig{ID: "changed"}
	filters.Groups[2] = struct{}{}
	filters.Protocols[protocol.Gemini] = struct{}{}
	filters.Models["changed"] = struct{}{}
	expiresAtMS = 1
	input.AccessKeys[0].AllowedPeerCIDRs[0] = netip.MustParsePrefix("198.51.100.0/24")
	input.AccessKeys[0].CostLimitRules[0].LimitNanoUSD = 1

	view := snapshot.Groups[1]
	if !reflect.DeepEqual(view.Models, []ModelConfig{{ID: "upstream", Alias: "public"}}) {
		t.Fatalf("group view changed with input = %#v", view)
	}
	gotFilters := snapshot.AccessKeysByID[1].Filters
	if _, ok := gotFilters.Groups[2]; ok {
		t.Fatal("group filter retained caller mutation")
	}
	if _, ok := gotFilters.Protocols[protocol.Gemini]; ok {
		t.Fatal("protocol filter retained caller mutation")
	}
	if _, ok := gotFilters.Models["changed"]; ok {
		t.Fatal("model filter retained caller mutation")
	}
	if got := snapshot.AccessKeysByID[1].CostLimitRules; len(got) != 1 || got[0].LimitNanoUSD != 100 {
		t.Fatalf("cost limit rules changed with input = %#v", got)
	}
	accessKey := snapshot.AccessKeysByID[1]
	if accessKey.ExpiresAtMS == nil || *accessKey.ExpiresAtMS != 1_900_000_000_000 {
		t.Fatalf("access key expiry changed with input = %#v", accessKey.ExpiresAtMS)
	}
	if !reflect.DeepEqual(accessKey.AllowedPeerCIDRs, []netip.Prefix{netip.MustParsePrefix("192.0.2.0/24")}) {
		t.Fatalf("access key CIDRs changed with input = %#v", accessKey.AllowedPeerCIDRs)
	}
}

func TestCompileRejectsInvalidCoreConfiguration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		input   CompileInput
		wantErr string
	}{
		{
			name:    "duplicate route entry",
			input:   CompileInput{ChannelRegistry: channel.NewRegistry(), Groups: []GroupConfig{{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`), Models: []ModelConfig{{ID: "a"}, {ID: "a", Alias: "a"}}, Enabled: true}}},
			wantErr: "duplicate route entry",
		},
		{
			name:    "zero external model weight sum",
			input:   CompileInput{ChannelRegistry: channel.NewRegistry(), Groups: []GroupConfig{{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`), Models: []ModelConfig{{ID: "a", Weight: intPointer(0)}}, Enabled: true}}},
			wantErr: "entry weights must sum to a positive value",
		},
		{
			name:    "negative entry weight",
			input:   CompileInput{ChannelRegistry: channel.NewRegistry(), Groups: []GroupConfig{{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`), Models: []ModelConfig{{ID: "a", Weight: intPointer(-1)}}, Enabled: true}}},
			wantErr: "weight must be between 0 and",
		},
		{
			name:    "non positive entry priority",
			input:   CompileInput{ChannelRegistry: channel.NewRegistry(), Groups: []GroupConfig{{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`), Models: []ModelConfig{{ID: "a", Priority: intPointer(0)}}, Enabled: true}}},
			wantErr: "priority must be at least 1",
		},
		{
			name:    "blank model id",
			input:   CompileInput{ChannelRegistry: channel.NewRegistry(), Groups: []GroupConfig{{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`), Models: []ModelConfig{{ID: "  "}}, Enabled: true}}},
			wantErr: "model id is required",
		},
		{
			name: "duplicate group id",
			input: CompileInput{ChannelRegistry: channel.NewRegistry(), Groups: []GroupConfig{
				{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`)},
				{ConnectionType: "api_key", ID: 1, ChannelID: channel.Anthropic, Params: json.RawMessage(`{}`)},
			}},
			wantErr: "duplicate group id",
		},
		{
			name: "duplicate access key hash",
			input: CompileInput{AccessKeys: []AccessKeyConfig{
				{ID: 1, KeyHash: "same", Status: AccessKeyStatusActive},
				{ID: 2, KeyHash: "same", Status: AccessKeyStatusActive},
			}},
			wantErr: "duplicate access key hash",
		},
		{
			name: "invalid cost limit rules",
			input: CompileInput{AccessKeys: []AccessKeyConfig{{
				ID: 1, KeyHash: "hash", Status: AccessKeyStatusActive,
				CostLimitRules: []accessquota.Rule{
					{ID: 1, Revision: 1, Kind: accessquota.KindTotal, LimitNanoUSD: 100},
					{ID: 2, Revision: 1, Kind: accessquota.KindTotal, LimitNanoUSD: 200},
				},
			}}},
			wantErr: "cost limit",
		},
		{
			name: "negative access key expiry",
			input: CompileInput{AccessKeys: []AccessKeyConfig{{
				ID: 1, KeyHash: "hash", Status: AccessKeyStatusActive,
				ExpiresAtMS: int64Pointer(-1),
			}}},
			wantErr: "expiry",
		},
		{
			name: "unsafe access key expiry",
			input: CompileInput{AccessKeys: []AccessKeyConfig{{
				ID: 1, KeyHash: "hash", Status: AccessKeyStatusActive,
				ExpiresAtMS: int64Pointer(9_007_199_254_740_992),
			}}},
			wantErr: "expiry",
		},
		{
			name: "invalid allowed peer cidr",
			input: CompileInput{AccessKeys: []AccessKeyConfig{{
				ID: 1, KeyHash: "hash", Status: AccessKeyStatusActive,
				AllowedPeerCIDRs: []netip.Prefix{{}},
			}}},
			wantErr: "CIDR",
		},
		{
			name: "non canonical allowed peer cidr",
			input: CompileInput{AccessKeys: []AccessKeyConfig{{
				ID: 1, KeyHash: "hash", Status: AccessKeyStatusActive,
				AllowedPeerCIDRs: []netip.Prefix{netip.MustParsePrefix("192.0.2.99/24")},
			}}},
			wantErr: "CIDR",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := Compile(test.input)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("Compile() error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func int64Pointer(value int64) *int64 { return &value }

// TestCompileEmitsOneRouteTargetPerModelEntry covers design §1.2 group one:
// three entries share external model "A" with per-entry weights and a
// fallback priority, plus a plain single-entry model. Every entry compiles
// into its own RouteTarget under the shared external name.
func TestCompileEmitsOneRouteTargetPerModelEntry(t *testing.T) {
	t.Parallel()

	weightA, weightB, weightC, fallback := 30, 50, 20, 2
	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{{
			ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI,
			Params: json.RawMessage(`{}`), Enabled: true,
			Models: []ModelConfig{
				{ID: "upstream-a", Alias: "A", Weight: &weightA},
				{ID: "upstream-b", Alias: "A", Weight: &weightB},
				{ID: "upstream-c", Alias: "A", Weight: &weightC, Priority: &fallback},
				{ID: "gpt-4o-mini"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	index := snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]
	shared := index["A"]
	if len(shared) != 3 {
		t.Fatalf("external model A targets = %#v", shared)
	}
	want := []RouteTarget{
		{GroupID: 1, UpstreamModelID: "upstream-a", Mode: channel.RouteNative, EntryWeight: 30, Priority: 1},
		{GroupID: 1, UpstreamModelID: "upstream-b", Mode: channel.RouteNative, EntryWeight: 50, Priority: 1},
		{GroupID: 1, UpstreamModelID: "upstream-c", Mode: channel.RouteNative, EntryWeight: 20, Priority: 2},
	}
	for i, target := range want {
		if shared[i].GroupID != target.GroupID || shared[i].UpstreamModelID != target.UpstreamModelID ||
			shared[i].EntryWeight != target.EntryWeight || shared[i].Priority != target.Priority {
			t.Fatalf("external model A targets[%d] = %#v, want %#v", i, shared[i], target)
		}
	}
	plain := index["gpt-4o-mini"]
	if len(plain) != 1 || plain[0].UpstreamModelID != "gpt-4o-mini" ||
		plain[0].EntryWeight != 1 || plain[0].Priority != 1 {
		t.Fatalf("single entry targets = %#v, want one default target", plain)
	}

	catalog := snapshot.ExecutionRouteCatalog[protocol.OpenAICompletions][execution.OperationChatCompletion]["A"]
	if len(catalog) != 3 || catalog[0].EntryWeight != 30 || catalog[2].Priority != 2 {
		t.Fatalf("route catalog targets = %#v", catalog)
	}
}

// TestCompileNormalizesUnsetEntryFields verifies legacy-shaped entries (no
// weight/priority) compile to the design defaults and that an explicit zero
// weight keeps the target indexed while marking it excluded from splitting.
func TestCompileNormalizesUnsetEntryFields(t *testing.T) {
	t.Parallel()

	zero := 0
	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{{
			ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI,
			Params: json.RawMessage(`{}`), Enabled: true,
			Models: []ModelConfig{
				{ID: "legacy"},
				{ID: "paused", Alias: "public"},
				{ID: "active", Alias: "public", Weight: &zero},
			},
		}},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	index := snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]
	if got := index["legacy"]; len(got) != 1 || got[0].EntryWeight != 1 || got[0].Priority != 1 {
		t.Fatalf("legacy entry targets = %#v, want normalized defaults", got)
	}
	public := index["public"]
	if len(public) != 2 || public[0].UpstreamModelID != "active" || public[1].UpstreamModelID != "paused" {
		t.Fatalf("public targets = %#v, want upstream-sorted entries", public)
	}
	if public[0].EntryWeight != 0 || public[0].Priority != 1 {
		t.Fatalf("zero weight target = %#v, want EntryWeight 0 with Priority 1", public[0])
	}
	if public[1].EntryWeight != 1 || public[1].Priority != 1 {
		t.Fatalf("default weight target = %#v, want EntryWeight 1 with Priority 1", public[1])
	}
}

// TestCompileKeepsNoModelRouteKeyResourcesUnchanged locks the resource
// operation paths (Responses retrieve/delete/passthrough) so entry-level
// compilation leaves them untouched.
func TestCompileKeepsNoModelRouteKeyResourcesUnchanged(t *testing.T) {
	t.Parallel()

	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{{
			ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI,
			Params: json.RawMessage(`{}`), Enabled: true,
			Models: []ModelConfig{{ID: "upstream-a", Alias: "A"}, {ID: "upstream-b", Alias: "A"}},
		}},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	for _, operation := range []execution.Operation{
		execution.OperationResponsesRetrieve,
		execution.OperationResponsesDelete,
		execution.OperationResponsesCancel,
		execution.OperationResponsesInputItems,
		execution.OperationResponsesPassthrough,
	} {
		targets := snapshot.ExecutionCandidates[protocol.OpenAIResponses][operation][NoModelRouteKey]
		if len(targets) != 1 {
			t.Fatalf("operation %q NoModelRouteKey targets = %#v, want one", operation, targets)
		}
		target := targets[0]
		if target.GroupID != 1 || target.UpstreamModelID != "" || target.EntryWeight != 1 || target.Priority != 1 {
			t.Fatalf("operation %q resource target = %#v", operation, target)
		}
	}
}

// TestCompileKeepsSingleEntryRoutingUnchanged is the C3 regression: a group
// with one entry per model (1:1 alias or none) compiles to exactly the
// pre-upgrade index shape with default entry weight and priority.
func TestCompileKeepsSingleEntryRoutingUnchanged(t *testing.T) {
	t.Parallel()

	legacy, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{
			{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "provider-a", Alias: "public"}}, Enabled: true},
			{ConnectionType: "api_key", ID: 2, ChannelID: channel.OpenAICompatible,
				Params: json.RawMessage(`{"base_url":"https://proxy.example/v1"}`),
				Models: []ModelConfig{{ID: "public"}}, Enabled: true},
		},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	upgraded, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{
			{ConnectionType: "api_key", ID: 1, ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
				Models: []ModelConfig{{ID: "provider-a", Alias: "public", Weight: intPointer(1), Priority: intPointer(1)}}, Enabled: true},
			{ConnectionType: "api_key", ID: 2, ChannelID: channel.OpenAICompatible,
				Params: json.RawMessage(`{"base_url":"https://proxy.example/v1"}`),
				Models: []ModelConfig{{ID: "public", Weight: intPointer(1), Priority: intPointer(1)}}, Enabled: true},
		},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	legacyIndex := legacy.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]
	upgradedIndex := upgraded.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]
	if !reflect.DeepEqual(stripResolvedTargets(legacyIndex["public"]), stripResolvedTargets(upgradedIndex["public"])) {
		t.Fatalf("single entry index changed with explicit defaults: legacy = %#v, upgraded = %#v",
			legacyIndex["public"], upgradedIndex["public"])
	}
	if len(legacyIndex["public"]) != 2 || legacyIndex["public"][0].GroupID != 1 || legacyIndex["public"][1].GroupID != 2 {
		t.Fatalf("single entry index = %#v", legacyIndex["public"])
	}
}

func stripResolvedTargets(targets []RouteTarget) []RouteTarget {
	stripped := make([]RouteTarget, len(targets))
	for i, target := range targets {
		target.ResolvedTarget = channel.ResolvedTarget{}
		stripped[i] = target
	}
	return stripped
}
