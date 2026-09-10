package scheduler

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

func inspectNow() time.Time { return time.Date(2026, time.July, 24, 10, 0, 0, 0, time.UTC) }

func inspectQuery() Query {
	return Query{ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("gpt-4o")}
}

func inspectSnapshot(t *testing.T) *state.ConfigSnapshot {
	t.Helper()
	snapshot, err := state.Compile(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{ConnectionType: "api_key", ID: 2, Name: "disabled", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`), Models: []state.ModelConfig{{ID: "provider-disabled", Alias: "public"}}, Enabled: false},
			{ConnectionType: "api_key", ID: 1, Name: "active", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`), Models: []state.ModelConfig{{ID: "provider-active", Alias: "public", EntryID: "e000000000001"}}, Enabled: true},
			{ConnectionType: "api_key", ID: 3, Name: "active-two", ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`), Models: []state.ModelConfig{{ID: "provider-zero", Alias: "public"}}, Enabled: true},
		},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return snapshot
}

func singleInspectSnapshot(t *testing.T) *state.ConfigSnapshot {
	return routeEntrySnapshot(t, []state.GroupConfig{{ID: 1, Name: "group", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: json.RawMessage(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "gpt-4o", Alias: "gpt-4o", Weight: intPointer(100)}}}})
}

func TestInspectReportsMultiCredentialGroupUnavailable(t *testing.T) {
	inspection, err := Inspect(singleInspectSnapshot(t), []state.CredentialRuntimeView{{ID: 11, GroupID: 1}, {ID: 12, GroupID: 1}}, inspectQuery(), inspectNow())
	if err != nil {
		t.Fatal(err)
	}
	if inspection.Routable || inspection.Reason != ReasonNoAvailableCredential || len(inspection.Groups) != 1 {
		t.Fatalf("inspection = %#v", inspection)
	}
}

func TestInspectReportsSingleCredentialGroup(t *testing.T) {
	inspection, err := Inspect(singleInspectSnapshot(t), []state.CredentialRuntimeView{{ID: 11, GroupID: 1}}, inspectQuery(), inspectNow())
	if err != nil {
		t.Fatal(err)
	}
	if !inspection.Routable || inspection.Reason != "" || len(inspection.Groups) != 1 || !inspection.Groups[0].Routable {
		t.Fatalf("inspection = %#v", inspection)
	}
}

func TestInspectReportsCredentialHealthReasons(t *testing.T) {
	now := inspectNow()
	for _, test := range []struct {
		name  string
		state state.CredentialRuntimeView
		want  ReasonCode
	}{
		{name: "auth unavailable", state: state.CredentialRuntimeView{ID: 11, GroupID: 1, AuthState: state.CredentialAuthStateRefreshing}, want: ReasonCredentialAuthUnavailable},
		{name: "blacklisted", state: state.CredentialRuntimeView{ID: 11, GroupID: 1, Blacklisted: true}, want: ReasonCredentialBlacklisted},
		{name: "cooldown", state: state.CredentialRuntimeView{ID: 11, GroupID: 1, CooldownUntil: now.Add(time.Minute)}, want: ReasonCredentialCooldown},
	} {
		t.Run(test.name, func(t *testing.T) {
			inspection, err := Inspect(singleInspectSnapshot(t), []state.CredentialRuntimeView{test.state}, inspectQuery(), now)
			if err != nil {
				t.Fatal(err)
			}
			if inspection.Routable || len(inspection.Groups) != 1 || len(inspection.Groups[0].Credentials) != 1 || inspection.Groups[0].Credentials[0].Reason != test.want {
				t.Fatalf("inspection = %#v, want reason %q", inspection, test.want)
			}
		})
	}
}

func TestInspectUsesEntryRuntimeAndWeightReasons(t *testing.T) {
	snapshot := singleInspectSnapshot(t)
	baseline, err := Inspect(snapshot, []state.CredentialRuntimeView{{ID: 11, GroupID: 1}}, inspectQuery(), inspectNow())
	if err != nil {
		t.Fatal(err)
	}
	if len(baseline.Groups) != 1 {
		t.Fatalf("baseline inspection = %#v", baseline)
	}
	inspection, err := InspectWithEntryRuntime(snapshot, []state.CredentialRuntimeView{{ID: 11, GroupID: 1}}, []state.EntryRuntimeView{{
		Key: state.RouteEntryKey{GroupID: 1, EntryID: baseline.Groups[0].EntryID}, CooldownUntil: inspectNow().Add(time.Minute),
	}}, inspectQuery(), inspectNow())
	if err != nil {
		t.Fatal(err)
	}
	if inspection.Routable || inspection.Groups[0].Reason != ReasonEntryCooldown {
		t.Fatalf("inspection = %#v, want entry cooldown reason", inspection)
	}
}

func TestInspectRejectsUnknownCredentialGroup(t *testing.T) {
	if _, err := Inspect(schedulerSnapshot(), []state.CredentialRuntimeView{{ID: 99, GroupID: 999}}, inspectQuery(), inspectNow()); err == nil {
		t.Fatal("Inspect() accepted credential from unknown group")
	}
}

func TestInspectHonorsAllowedCredentialScope(t *testing.T) {
	query := inspectQuery()
	query.AllowedCredentialIDs = map[uint]struct{}{99: {}}
	inspection, err := Inspect(schedulerSnapshot(), []state.CredentialRuntimeView{{ID: 11, GroupID: 1}}, query, inspectNow())
	if err != nil {
		t.Fatal(err)
	}
	if inspection.Routable || inspection.Groups[0].Credentials[0].Reason != ReasonCredentialNotAllowed {
		t.Fatalf("inspection = %#v", inspection)
	}
}

func TestInspectAppliesTopLevelReasonPriority(t *testing.T) {
	snapshot := inspectSnapshot(t)
	tests := []struct {
		name   string
		query  Query
		reason ReasonCode
	}{
		{
			name: "access key disabled",
			query: Query{
				ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("public"),
				AccessKey: state.AccessKeyView{Status: state.AccessKeyStatusDisabled},
			},
			reason: ReasonAccessKeyDisabled,
		},
		{
			name: "protocol filtered before model",
			query: Query{
				ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("public"),
				AccessKey: state.AccessKeyView{
					Status: state.AccessKeyStatusActive,
					Filters: state.FilterSet{
						Protocols: map[protocol.Protocol]struct{}{protocol.Anthropic: {}},
						Models:    map[string]struct{}{"other": {}},
					},
				},
			},
			reason: ReasonProtocolFiltered,
		},
		{
			name: "model filtered",
			query: Query{
				ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("public"),
				AccessKey: state.AccessKeyView{
					Status:  state.AccessKeyStatusActive,
					Filters: state.FilterSet{Models: map[string]struct{}{"other": {}}},
				},
			},
			reason: ReasonModelFiltered,
		},
		{
			name: "no route target",
			query: Query{
				ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion, ExternalModel: modelPointer("missing"),
				AccessKey: state.AccessKeyView{Status: state.AccessKeyStatusActive},
			},
			reason: ReasonNoRouteTarget,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := Inspect(snapshot, nil, test.query, inspectNow())
			if err != nil {
				t.Fatalf("Inspect() error = %v", err)
			}
			if got.Routable || got.Reason != test.reason || len(got.Groups) != 0 {
				t.Fatalf("Inspection = %#v, want reason %q and empty groups", got, test.reason)
			}
		})
	}
}

func TestInspectRequiresModelWhenAccessKeyHasModelFilter(t *testing.T) {
	t.Parallel()

	got, err := Inspect(inspectSnapshot(t), nil, Query{
		ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion,
		ExternalModel: nil,
		AccessKey: state.AccessKeyView{
			Status: state.AccessKeyStatusActive,
			Filters: state.FilterSet{
				Models: map[string]struct{}{"public": {}},
			},
		},
	}, inspectNow())
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if got.Routable || got.Reason != ReasonModelRequiredByFilter ||
		len(got.Groups) != 0 {
		t.Fatalf(
			"Inspection = %#v, want reason %q and empty groups",
			got,
			ReasonModelRequiredByFilter,
		)
	}
}

func floatPointer(value float64) *float64 { return &value }

func TestApplyEffectiveSharesUsesLowestAvailableTier(t *testing.T) {
	groups := []GroupInspection{
		{Priority: 1, Routable: false, Credentials: []CredentialInspection{{Available: true, EffectiveWeight: 100}}},
		{Priority: 2, Routable: true, Credentials: []CredentialInspection{{Available: true, EffectiveWeight: 25}}},
		{Priority: 2, Routable: true, Credentials: []CredentialInspection{{Available: true, EffectiveWeight: 75}}},
		{Priority: 3, Routable: true, Credentials: []CredentialInspection{{Available: true, EffectiveWeight: 100}}},
	}
	applyEffectiveShares(groups)
	if groups[0].EffectiveShare != 0 || groups[3].EffectiveShare != 0 {
		t.Fatalf("inactive tiers received shares: %#v", groups)
	}
	if groups[1].EffectiveShare != 0.25 || groups[2].EffectiveShare != 0.75 {
		t.Fatalf("active tier shares = %v, %v, want 0.25, 0.75", groups[1].EffectiveShare, groups[2].EffectiveShare)
	}
}

func TestInspectReportsNoKeysForIncludedGroup(t *testing.T) {
	got, err := Inspect(inspectSnapshot(t), nil, Query{
		ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion,
		ExternalModel: modelPointer("public"),
		AccessKey: state.AccessKeyView{
			Status:  state.AccessKeyStatusActive,
			Filters: state.FilterSet{Groups: map[uint]struct{}{1: {}}},
		},
	}, inspectNow())
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if got.Routable || got.Reason != ReasonNoAvailableCredential ||
		len(got.Groups) != 3 ||
		!got.Groups[0].Included ||
		got.Groups[0].Reason != ReasonNoCredentials ||
		got.Groups[0].Credentials == nil ||
		len(got.Groups[0].Credentials) != 0 {
		t.Fatalf("no-key Inspection = %#v", got)
	}
}

func TestInspectSummarizesStaticGroupExclusions(t *testing.T) {
	group := func(id uint, enabled bool) state.GroupConfig {
		return state.GroupConfig{ConnectionType: "api_key", ID: id, Name: fmt.Sprintf("group-%d", id),
			ChannelID: channel.OpenAI, Params: json.RawMessage(`{}`),
			Models:  []state.ModelConfig{{ID: fmt.Sprintf("provider-%d", id), Alias: "public"}},
			Enabled: enabled,
		}
	}
	tests := []struct {
		name          string
		groups        []state.GroupConfig
		allowedGroups map[uint]struct{}
		want          ReasonCode
	}{
		{
			name:   "all disabled",
			groups: []state.GroupConfig{group(1, false), group(2, false)},
			want:   ReasonGroupDisabled,
		},
		{
			name:          "all filtered",
			groups:        []state.GroupConfig{group(1, true), group(2, true)},
			allowedGroups: map[uint]struct{}{999: {}},
			want:          ReasonGroupFiltered,
		},
		{
			name:          "mixed disabled and filtered",
			groups:        []state.GroupConfig{group(1, false), group(2, true)},
			allowedGroups: map[uint]struct{}{999: {}},
			want:          ReasonNoAvailableGroup,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot, err := state.Compile(state.CompileInput{
				ChannelRegistry: channel.NewRegistry(),
				Groups:          test.groups,
			})
			if err != nil {
				t.Fatalf("Compile() error = %v", err)
			}
			result, err := Inspect(snapshot, nil, Query{
				ClientProtocol: protocol.OpenAICompletions, Operation: execution.OperationChatCompletion,
				ExternalModel: modelPointer("public"),
				AccessKey: state.AccessKeyView{
					Status:  state.AccessKeyStatusActive,
					Filters: state.FilterSet{Groups: test.allowedGroups},
				},
			}, inspectNow())
			if err != nil {
				t.Fatalf("Inspect() error = %v", err)
			}
			if result.Routable || result.Reason != test.want ||
				len(result.Groups) != 2 {
				t.Fatalf("Inspection = %#v, want reason %q", result, test.want)
			}
		})
	}
}
