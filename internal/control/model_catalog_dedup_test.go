package control

import (
	"encoding/json"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

// 基准：同一对外名的多条目（跨条目、跨分组）在目录与统计中只计一次（设计 §8.3）。
func newCatalogDedupSnapshot(t *testing.T) *state.ConfigSnapshot {
	t.Helper()
	snapshot, err := state.Compile(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{ConnectionType: "api_key", ID: 1, Name: "one", ChannelID: channel.OpenAI,
				Params: mustRawJSON(`{}`), Enabled: true,
				Models: []state.ModelConfig{
					{ID: "up-a", Alias: "pub"},
					{ID: "up-b", Alias: "pub"},
					{ID: "solo"},
				},
			},
			{ConnectionType: "api_key", ID: 2, Name: "two", ChannelID: channel.OpenAI,
				Params: mustRawJSON(`{}`), Enabled: true,
				Models: []state.ModelConfig{{ID: "up-c", Alias: "pub"}},
			},
			{ConnectionType: "api_key", ID: 3, Name: "three", ChannelID: channel.OpenAI,
				Params: mustRawJSON(`{}`), Enabled: false,
				Models: []state.ModelConfig{{ID: "up-d", Alias: "pub"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return snapshot
}

func mustRawJSON(raw string) json.RawMessage {
	return json.RawMessage(raw)
}

func TestCatalogDedupCountsExternalNamesAcrossEntriesAndGroups(t *testing.T) {
	t.Parallel()
	snapshot := newCatalogDedupSnapshot(t)

	// 分组 1 两条目同对外名 pub + 单条目 solo；分组 2 复用 pub → 总对外名 2。
	if got := countHomeModels(snapshot); got != 2 {
		t.Fatalf("countHomeModels() = %d, want 2", got)
	}

	accessKey := state.AccessKeyView{}
	allowedGroups := accessibleHomeGroups(snapshot, accessKey)
	// 禁用分组 3 不参与统计。
	if got := countScopedHomeModels(snapshot, allowedGroups, accessKey); got != 2 {
		t.Fatalf("countScopedHomeModels() = %d, want 2", got)
	}

	// 模型过滤器按对外名匹配：pub 命中全部三条目所在分组。
	filtered := state.AccessKeyView{Filters: state.FilterSet{
		Models: map[string]struct{}{"pub": {}},
	}}
	scoped := accessibleHomeGroups(snapshot, filtered)
	if _, exists := scoped[1]; !exists {
		t.Fatal("model filter pub should admit group 1")
	}
	if _, exists := scoped[2]; !exists {
		t.Fatal("model filter pub should admit group 2")
	}
	if got := countScopedHomeModels(snapshot, scoped, filtered); got != 1 {
		t.Fatalf("scoped count with pub filter = %d, want 1", got)
	}
}

func TestCatalogDedupKeepsSingleEntryBehavior(t *testing.T) {
	t.Parallel()
	// C1：无别名/1:1 别名单条目行为与升级前一致。
	snapshot, err := state.Compile(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{{
			ConnectionType: "api_key", ID: 1, Name: "single", ChannelID: channel.OpenAI,
			Params: mustRawJSON(`{}`), Enabled: true,
			Models: []state.ModelConfig{{ID: "plain"}, {ID: "up", Alias: "aliased"}},
		}},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if got := countHomeModels(snapshot); got != 2 {
		t.Fatalf("countHomeModels() = %d, want 2", got)
	}
	if got := countScopedHomeModels(snapshot, map[uint]struct{}{1: {}}, state.AccessKeyView{}); got != 2 {
		t.Fatalf("countScopedHomeModels() = %d, want 2", got)
	}
}
