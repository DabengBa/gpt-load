package channel

import (
	"encoding/json"
	"reflect"
	"testing"

	"gpt-load/internal/channel/spec"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestFinalRegistryContainsOnlyApprovedChannels(t *testing.T) {
	registry := NewRegistry()
	want := []ID{
		OpenAI,
		Codex,
		Claude,
		Antigravity,
		Grok,
		Anthropic,
		Gemini,
		AzureOpenAI,
		AWSBedrock,
		GoogleVertex,
		DeepSeek,
		MoonshotAI,
		SiliconFlow,
		ZhipuAI,
		Alibaba,
		Volcengine,
		OpenRouter,
		Groq,
		XAI,
		GPTLoad,
		NewAPI,
		CLIProxyAPI,
		Sub2API,
		OpenAICompatible,
	}
	if got := descriptorIDs(registry.List()); !reflect.DeepEqual(got, want) {
		t.Fatalf("List() IDs = %v, want %v", got, want)
	}
	for _, removed := range []ID{"anthropic_compatible", "gemini_compatible"} {
		if _, ok := registry.Get(removed); ok {
			t.Fatalf("removed channel %q is still registered", removed)
		}
	}
}

func TestNativeProviderChannelsResolveWithoutCompatibleFallback(t *testing.T) {
	registry := NewRegistry()
	tests := []struct {
		channelID         ID
		providerKind      ProviderKind
		responsesMode     RouteMode
		anthropicMode     RouteMode
		catalogProviderID string
	}{
		{channelID: DeepSeek, providerKind: ProviderKind("deepseek"), responsesMode: RouteNative, anthropicMode: RouteNative, catalogProviderID: "deepseek"},
		{channelID: OpenRouter, providerKind: ProviderKind("openrouter"), responsesMode: RouteNative, anthropicMode: RouteConverted, catalogProviderID: "openrouter"},
		{channelID: Groq, providerKind: ProviderKind("groq"), responsesMode: RouteConverted, anthropicMode: RouteConverted, catalogProviderID: "groq"},
		{channelID: XAI, providerKind: ProviderKind("xai"), responsesMode: RouteNative, anthropicMode: RouteConverted, catalogProviderID: "xai"},
	}
	for _, test := range tests {
		t.Run(string(test.channelID), func(t *testing.T) {
			target, err := registry.Resolve(test.channelID, nil)
			if err != nil {
				t.Fatalf("Resolve() error = %v", err)
			}
			if target.ProviderKind != test.providerKind || target.CatalogProviderID != test.catalogProviderID {
				t.Fatalf("target = %#v", target)
			}
			if got := string(target.TargetConfig); got != `{}` {
				t.Fatalf("TargetConfig = %s, want {}", got)
			}
			if mode, ok := target.Mode(protocol.OpenAICompletions, execution.OperationChatCompletion); !ok || mode != RouteNative {
				t.Fatalf("chat mode = %q, %t", mode, ok)
			}
			if mode, ok := target.Mode(protocol.OpenAIResponses, execution.OperationResponsesCreate); !ok || mode != test.responsesMode {
				t.Fatalf("Responses mode = %q, %t, want %q", mode, ok, test.responsesMode)
			}
			if mode, ok := target.Mode(protocol.Anthropic, execution.OperationChatCompletion); !ok || mode != test.anthropicMode {
				t.Fatalf("Anthropic mode = %q, %t, want %q", mode, ok, test.anthropicMode)
			}
			if test.channelID == DeepSeek {
				if mode, ok := target.Mode(protocol.Anthropic, execution.OperationListModels); !ok || mode != RouteConverted {
					t.Fatalf("DeepSeek Anthropic model-list mode = %q, %t", mode, ok)
				}
			}
			if _, ok := target.Mode(protocol.OpenAIResponses, execution.OperationResponsesRetrieve); ok {
				t.Fatal("target unexpectedly supports Responses lifecycle")
			}
		})
	}
}

func TestNativeProviderBaseURLUsesSDKPrefixContract(t *testing.T) {
	registry := NewRegistry()
	tests := []struct {
		channelID ID
		baseURL   string
	}{
		{channelID: OpenAI, baseURL: "https://mirror.example"},
		{channelID: Anthropic, baseURL: "https://mirror.example"},
		{channelID: Gemini, baseURL: "https://mirror.example/v1beta"},
		{channelID: DeepSeek, baseURL: "https://mirror.example/v1"},
		{channelID: OpenRouter, baseURL: "https://mirror.example/api"},
		{channelID: Groq, baseURL: "https://mirror.example/openai"},
		{channelID: XAI, baseURL: "https://mirror.example"},
	}
	for _, test := range tests {
		t.Run(string(test.channelID), func(t *testing.T) {
			target, err := registry.Resolve(test.channelID, json.RawMessage(`{"base_url":"`+test.baseURL+`"}`))
			if err != nil {
				t.Fatalf("Resolve() error = %v", err)
			}
			if got := string(target.TargetConfig); got != `{"base_url":"`+test.baseURL+`"}` {
				t.Fatalf("TargetConfig = %s", got)
			}
		})
	}

	if _, err := registry.Resolve(OpenAI, json.RawMessage(`{"base_url":"https://mirror.example?tenant=one"}`)); err == nil {
		t.Fatal("Resolve(BaseURL query) error = nil")
	}
}

func TestBaseURLCanonicalizesDefaultPortsForRuntimeReuse(t *testing.T) {
	registry := NewRegistry()
	tests := []struct {
		raw  string
		want string
	}{
		{raw: `{"base_url":"HTTPS://API.OPENAI.COM:443/"}`, want: `{"base_url":"https://api.openai.com"}`},
		{raw: `{"base_url":"http://relay.example:80/v1/"}`, want: `{"base_url":"http://relay.example/v1"}`},
		{raw: `{"base_url":"https://relay.example:8443/v1/"}`, want: `{"base_url":"https://relay.example:8443/v1"}`},
	}
	for _, test := range tests {
		t.Run(test.raw, func(t *testing.T) {
			target, err := registry.Resolve(OpenAI, json.RawMessage(test.raw))
			if err != nil {
				t.Fatal(err)
			}
			if got := string(target.TargetConfig); got != test.want {
				t.Fatalf("TargetConfig = %s, want %s", got, test.want)
			}
		})
	}
}

// TestAPIKeyChannelsDeclareAuthoritativeProbeContract locks the recommended
// mapping: exactly one code-owned probe protocol per API-key channel, a declared
// probe route for it, and a budget above the observed upstream floor.
func TestAPIKeyChannelsDeclareAuthoritativeProbeContract(t *testing.T) {
	registry := NewRegistry()
	params := map[ID]json.RawMessage{
		OpenAICompatible: json.RawMessage(`{"base_url":"https://compatible.example"}`),
		GPTLoad:          json.RawMessage(`{"base_url":"https://gateway.example"}`),
		NewAPI:           json.RawMessage(`{"base_url":"https://newapi.example"}`),
		CLIProxyAPI:      json.RawMessage(`{"base_url":"https://cliproxy.example"}`),
		Sub2API:          json.RawMessage(`{"base_url":"https://sub2api.example"}`),
		AzureOpenAI:      json.RawMessage(`{"endpoint":"https://azure.example"}`),
		AWSBedrock:       json.RawMessage(`{"region":"us-east-1"}`),
	}
	tests := []struct {
		channelID ID
		protocol  protocol.Protocol
	}{
		{channelID: OpenAI, protocol: protocol.OpenAIResponses},
		{channelID: OpenAICompatible, protocol: protocol.OpenAICompletions},
		{channelID: Anthropic, protocol: protocol.Anthropic},
		{channelID: Gemini, protocol: protocol.Gemini},
		{channelID: GoogleVertex, protocol: protocol.Gemini},
		{channelID: DeepSeek, protocol: protocol.OpenAIResponses},
		{channelID: Groq, protocol: protocol.OpenAICompletions},
		{channelID: XAI, protocol: protocol.OpenAIResponses},
		{channelID: OpenRouter, protocol: protocol.OpenAIResponses},
		{channelID: AzureOpenAI, protocol: protocol.OpenAICompletions},
		{channelID: AWSBedrock, protocol: protocol.OpenAICompletions},
		{channelID: GPTLoad, protocol: protocol.OpenAICompletions},
		{channelID: NewAPI, protocol: protocol.OpenAICompletions},
		{channelID: CLIProxyAPI, protocol: protocol.OpenAICompletions},
		{channelID: Sub2API, protocol: protocol.OpenAICompletions},
		{channelID: Alibaba, protocol: protocol.OpenAICompletions},
		{channelID: MoonshotAI, protocol: protocol.OpenAICompletions},
		{channelID: SiliconFlow, protocol: protocol.OpenAICompletions},
		{channelID: ZhipuAI, protocol: protocol.OpenAICompletions},
		{channelID: Volcengine, protocol: protocol.OpenAICompletions},
	}
	for _, test := range tests {
		t.Run(string(test.channelID), func(t *testing.T) {
			target, err := registry.Resolve(test.channelID, params[test.channelID])
			if err != nil {
				t.Fatalf("Resolve() error = %v", err)
			}
			contract, ok := target.ProbeContract()
			if !ok {
				t.Fatal("API-key channel has no probe contract")
			}
			if contract.Protocol != test.protocol {
				t.Fatalf("probe protocol = %q, want %q", contract.Protocol, test.protocol)
			}
			if contract.MinOutputTokens < 3 {
				t.Fatalf("probe budget = %d, want at least 3", contract.MinOutputTokens)
			}
			if _, ok := target.Mode(contract.Protocol, execution.OperationProbe); !ok {
				t.Fatalf("probe protocol %q has no declared probe route", contract.Protocol)
			}
		})
	}

	for _, subscription := range []ID{Claude, Codex, Antigravity, Grok} {
		t.Run("subscription/"+string(subscription), func(t *testing.T) {
			target, err := registry.Resolve(subscription, nil)
			if err != nil {
				t.Fatalf("Resolve() error = %v", err)
			}
			if _, ok := target.ProbeContract(); ok {
				t.Fatal("subscription channel must not declare a probe contract")
			}
		})
	}
}

func TestCompilerRejectsInvalidProbeContracts(t *testing.T) {
	probeRoute := map[protocol.Protocol]map[execution.Operation]RouteMode{
		protocol.OpenAICompletions: {execution.OperationProbe: execution.RouteNative},
		// Declared routes make the non-generative rejection the failing check.
		protocol.OpenAIEmbeddings: {execution.OperationProbe: execution.RouteNative},
		protocol.Rerank:           {execution.OperationProbe: execution.RouteNative},
	}
	tests := []struct {
		name       string
		connection spec.ConnectionType
		contract   spec.ProbeContract
		wantErr    bool
	}{
		{
			name: "api key without contract", connection: spec.ConnectionAPIKey,
			contract: spec.ProbeContract{}, wantErr: true,
		},
		{
			name: "api key budget below floor", connection: spec.ConnectionAPIKey,
			contract: spec.ProbeContract{Protocol: protocol.OpenAICompletions, MinOutputTokens: 1}, wantErr: true,
		},
		{
			name: "api key embeddings contract", connection: spec.ConnectionAPIKey,
			contract: spec.ProbeContract{Protocol: protocol.OpenAIEmbeddings, MinOutputTokens: 16}, wantErr: true,
		},
		{
			name: "api key rerank contract", connection: spec.ConnectionAPIKey,
			contract: spec.ProbeContract{Protocol: protocol.Rerank, MinOutputTokens: 16}, wantErr: true,
		},
		{
			name: "api key without declared route", connection: spec.ConnectionAPIKey,
			contract: spec.ProbeContract{Protocol: protocol.Anthropic, MinOutputTokens: 16}, wantErr: true,
		},
		{
			name: "api key valid", connection: spec.ConnectionAPIKey,
			contract: spec.ProbeContract{Protocol: protocol.OpenAICompletions, MinOutputTokens: 16},
		},
		{
			name: "subscription with contract", connection: spec.ConnectionSubscription,
			contract: spec.ProbeContract{Protocol: protocol.OpenAICompletions, MinOutputTokens: 16}, wantErr: true,
		},
		{
			name: "subscription without contract", connection: spec.ConnectionSubscription,
			contract: spec.ProbeContract{},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			definition := spec.Definition{
				ID:         spec.OpenAI,
				Connection: spec.Connection{Type: test.connection},
				Provider:   spec.ProviderBinding{ProbeContract: test.contract},
			}
			err := validateProbeContract(definition, probeRoute)
			if test.wantErr && err == nil {
				t.Fatal("validateProbeContract() error = nil")
			}
			if !test.wantErr && err != nil {
				t.Fatalf("validateProbeContract() error = %v", err)
			}
		})
	}
}
