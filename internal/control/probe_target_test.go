package control

import (
	"encoding/json"
	"net/http"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

// probeTestGroup builds a minimal group view whose resolved target comes from
// the real channel registry, so probe targets are exercised against the
// code-owned declarations rather than a hand-built stub.
func probeTestGroup(channelID channel.ID, params json.RawMessage, models ...string) state.GroupView {
	resolved, err := channel.NewRegistry().Resolve(channelID, params)
	if err != nil {
		panic(err)
	}
	group := state.GroupView{
		ChannelID:      channelID,
		Params:         append(json.RawMessage(nil), params...),
		ResolvedTarget: resolved,
	}
	for _, model := range models {
		group.Models = append(group.Models, state.ModelConfig{ID: model})
	}
	return group
}

func TestBuildGroupProbeTargetSkipsSubscriptionGroups(t *testing.T) {
	t.Parallel()

	group := probeTestGroup(channel.Anthropic, json.RawMessage(`{}`), "claude-test")
	group.ConnectionType = "subscription"
	if _, ok := buildGroupProbeTarget(group, "claude-test"); ok {
		t.Fatal("subscription group must not have a credential probe target")
	}
}

func TestBuildGroupProbeTargetUsesAuthoritativeContractProtocol(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		channelID channel.ID
		params    json.RawMessage
		model     string
		want      protocol.Protocol
		wantMode  channel.RouteMode
	}{
		{name: "openai", channelID: channel.OpenAI, params: json.RawMessage(`{}`), model: "gpt-4o", want: protocol.OpenAIResponses, wantMode: channel.RouteNative},
		{name: "anthropic", channelID: channel.Anthropic, params: json.RawMessage(`{}`), model: "claude-test", want: protocol.Anthropic, wantMode: channel.RouteNative},
		{name: "gemini", channelID: channel.Gemini, params: json.RawMessage(`{}`), model: "gemini-2.5-pro", want: protocol.Gemini, wantMode: channel.RouteNative},
		{name: "vertex", channelID: channel.GoogleVertex, params: json.RawMessage(`{}`), model: "gemini-2.5-pro", want: protocol.Gemini, wantMode: channel.RouteNative},
		{name: "openrouter", channelID: channel.OpenRouter, params: json.RawMessage(`{}`), model: "vendor/model", want: protocol.OpenAIResponses, wantMode: channel.RouteNative},
		{name: "deepseek", channelID: channel.DeepSeek, params: json.RawMessage(`{}`), model: "deepseek-chat", want: protocol.OpenAIResponses, wantMode: channel.RouteNative},
		{name: "xai", channelID: channel.XAI, params: json.RawMessage(`{}`), model: "grok-test", want: protocol.OpenAIResponses, wantMode: channel.RouteNative},
		{name: "groq", channelID: channel.Groq, params: json.RawMessage(`{}`), model: "llama-test", want: protocol.OpenAICompletions, wantMode: channel.RouteNative},
		{name: "compatible", channelID: channel.OpenAICompatible, params: json.RawMessage(`{"base_url":"https://upstream.example/v1"}`), model: "vendor-model", want: protocol.OpenAICompletions, wantMode: channel.RouteNative},
		{name: "gateway", channelID: channel.GPTLoad, params: json.RawMessage(`{"base_url":"https://gateway.example"}`), model: "vendor-model", want: protocol.OpenAICompletions, wantMode: channel.RouteNative},
		{name: "azure", channelID: channel.AzureOpenAI, params: json.RawMessage(`{"endpoint":"https://example.openai.azure.com"}`), model: "deployment", want: protocol.OpenAICompletions, wantMode: channel.RouteConverted},
		{name: "bedrock", channelID: channel.AWSBedrock, params: json.RawMessage(`{"region":"us-east-1"}`), model: "anthropic.claude-test", want: protocol.OpenAICompletions, wantMode: channel.RouteConverted},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			group := probeTestGroup(test.channelID, test.params, test.model)
			target, ok := buildGroupProbeTarget(group, test.model)
			if !ok {
				t.Fatal("buildGroupProbeTarget() ok = false")
			}
			if target.protocol != test.want || target.routeMode != test.wantMode {
				t.Fatalf("target = %q/%q, want %q/%q", target.protocol, target.routeMode, test.want, test.wantMode)
			}
			if target.maxOutputTokens < 3 {
				t.Fatalf("target budget = %d, want at least 3", target.maxOutputTokens)
			}
			if target.model != test.model {
				t.Fatalf("target model = %q, want %q", target.model, test.model)
			}
		})
	}
}

func TestBuildGroupProbeTargetUsesFirstConfiguredModel(t *testing.T) {
	t.Parallel()

	group := probeTestGroup(channel.OpenAI, json.RawMessage(`{}`), "first-model", "second-model")
	target, ok := buildGroupProbeTarget(group, "")
	if !ok {
		t.Fatal("buildGroupProbeTarget() ok = false")
	}
	if target.model != "first-model" {
		t.Fatalf("target model = %q, want first configured model", target.model)
	}
}

func TestBuildGroupProbeTargetRejectsModelWithoutDeclaredProbeRoute(t *testing.T) {
	t.Parallel()

	// GoogleVertex resolves a Gemini model with the native route resolver but a
	// non-Gemini model keeps the converted route; both must remain supported.
	group := probeTestGroup(channel.GoogleVertex, json.RawMessage(`{}`), "custom-endpoint")
	if target, ok := buildGroupProbeTarget(group, "custom-endpoint"); !ok || target.routeMode != channel.RouteConverted {
		t.Fatalf("custom-endpoint target = %#v/%t, want converted", target, ok)
	}

	empty := probeTestGroup(channel.OpenAI, json.RawMessage(`{}`))
	if _, ok := buildGroupProbeTarget(empty, ""); ok {
		t.Fatal("group without models must not produce a probe target")
	}
}

func TestValidationSignatureIsStableAndCoversInputs(t *testing.T) {
	t.Parallel()

	base := probeTestGroup(
		channel.OpenAICompatible,
		json.RawMessage(`{"base_url":"https://upstream.example/v1"}`),
		"model-a",
	)
	base.ID = 7
	base.HeaderRules = state.HeaderRules{
		Set:    map[string]string{"X-Alpha": "first", "X-Zeta": "last"},
		Remove: []string{"X-Old", "X-Beta"},
	}

	renamed := cloneProbeGroup(base)
	renamed.HeaderRules.Set = map[string]string{"x-zeta": "last", "x-alpha": "first"}
	renamed.HeaderRules.Remove = []string{"x-beta", "x-old"}
	baseTarget, ok := buildGroupValidationTarget(base)
	if !ok {
		t.Fatal("base target ok = false")
	}
	renamedTarget, ok := buildGroupValidationTarget(renamed)
	if !ok {
		t.Fatal("renamed target ok = false")
	}
	if baseTarget.signature != renamedTarget.signature {
		t.Fatal("signature must be stable across normalized header order")
	}

	mutations := map[string]func(*state.GroupView){
		"group id": func(group *state.GroupView) { group.ID++ },
		"model":    func(group *state.GroupView) { group.Models[0].ID = "model-b" },
		"params": func(group *state.GroupView) {
			*group = probeTestGroup(channel.OpenAICompatible, json.RawMessage(`{"base_url":"https://changed.example/v1"}`), "model-a")
			group.ID = 7
			group.HeaderRules = base.HeaderRules
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			changed := cloneProbeGroup(base)
			mutate(&changed)
			changedTarget, ok := buildGroupValidationTarget(changed)
			if !ok {
				t.Fatal("changed target ok = false")
			}
			if changedTarget.signature == baseTarget.signature {
				t.Fatalf("signature did not change for %s", name)
			}
		})
	}

	// The executed route mode and output budget are part of the probe target:
	// a restore proof must not survive a change to either, so the signature must
	// cover both values directly.
	budgetVariants := map[string]groupValidationSignature{
		"route mode": computeGroupValidationSignature(
			base, baseTarget.protocol, "model-a", channel.RouteConverted, baseTarget.maxOutputTokens,
		),
		"output budget": computeGroupValidationSignature(
			base, baseTarget.protocol, "model-a", baseTarget.routeMode, baseTarget.maxOutputTokens+1,
		),
	}
	for name, signature := range budgetVariants {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if signature == baseTarget.signature {
				t.Fatalf("signature did not change for %s", name)
			}
		})
	}
}

func cloneProbeGroup(group state.GroupView) state.GroupView {
	cloned := group
	cloned.Params = append(json.RawMessage(nil), group.Params...)
	cloned.ResolvedTarget.TargetConfig = append(json.RawMessage(nil), group.ResolvedTarget.TargetConfig...)
	cloned.Models = append([]state.ModelConfig(nil), group.Models...)
	cloned.HeaderRules.Set = make(map[string]string, len(group.HeaderRules.Set))
	for name, value := range group.HeaderRules.Set {
		cloned.HeaderRules.Set[name] = value
	}
	cloned.HeaderRules.Remove = append([]string(nil), group.HeaderRules.Remove...)
	return cloned
}

func TestGroupCredentialProbeWithoutGeneratedTextIsNoAnswer(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-no-answer-secret")
	var credential models.Credential
	if err := fixture.db.Where("group_id = ?", groupID).Take(&credential).Error; err != nil {
		t.Fatal(err)
	}
	executor := &credentialProbeTestExecutor{execute: func(execution.AttemptSpec) execution.AttemptResult {
		return execution.AttemptResult{
			DispatchState:   execution.DispatchMaybeSent,
			ResponseStarted: true,
			StatusCode:      http.StatusOK,
			Header:          http.Header{},
			Body:            []byte(`{"choices":[]}`),
		}
	}}
	fixture.service.executor = executor

	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil {
		t.Fatal(err)
	}
	if response.Outcome != ProbeOutcomeFailed || response.Reason == nil ||
		*response.Reason != ProbeReasonNoAnswer || response.Recovered {
		t.Fatalf("probe response = %#v", response)
	}
	if calls := executor.recordedCalls(); len(calls) != 1 {
		t.Fatalf("probe calls = %d, want exactly one", len(calls))
	}
}

func TestGroupCredentialProbeUnparseableBodyIsInvalidResponse(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-invalid-body-secret")
	var credential models.Credential
	if err := fixture.db.Where("group_id = ?", groupID).Take(&credential).Error; err != nil {
		t.Fatal(err)
	}
	executor := &credentialProbeTestExecutor{execute: func(execution.AttemptSpec) execution.AttemptResult {
		return execution.AttemptResult{
			DispatchState:        execution.DispatchMaybeSent,
			ResponseStarted:      true,
			StatusCode:           http.StatusOK,
			Header:               http.Header{},
			Body:                 []byte(`<html>gateway error</html>`),
			ProbeResponseInvalid: true,
		}
	}}
	fixture.service.executor = executor

	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil {
		t.Fatal(err)
	}
	if response.Outcome != ProbeOutcomeFailed || response.Reason == nil ||
		*response.Reason != ProbeReasonInvalidResponse || response.Recovered {
		t.Fatalf("probe response = %#v", response)
	}
	if calls := executor.recordedCalls(); len(calls) != 1 {
		t.Fatalf("probe calls = %d, want exactly one", len(calls))
	}
}
