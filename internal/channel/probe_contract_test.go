package channel

import (
	"encoding/json"
	"testing"

	"gpt-load/internal/channel/spec"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestProbeContractDeclaresSingleGenerativeProtocol(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	expected := map[ID]protocol.Protocol{
		OpenAI:           protocol.OpenAIResponses,
		AzureOpenAI:      protocol.OpenAICompletions,
		AWSBedrock:       protocol.OpenAICompletions,
		GoogleVertex:     protocol.Gemini,
		Gemini:           protocol.Gemini,
		Anthropic:        protocol.Anthropic,
		DeepSeek:         protocol.OpenAIResponses,
		MoonshotAI:       protocol.OpenAICompletions,
		SiliconFlow:      protocol.OpenAICompletions,
		ZhipuAI:          protocol.OpenAICompletions,
		Alibaba:          protocol.OpenAICompletions,
		Volcengine:       protocol.OpenAICompletions,
		OpenRouter:       protocol.OpenAIResponses,
		Groq:             protocol.OpenAICompletions,
		XAI:              protocol.OpenAIResponses,
		GPTLoad:          protocol.OpenAICompletions,
		NewAPI:           protocol.OpenAICompletions,
		CLIProxyAPI:      protocol.OpenAICompletions,
		Sub2API:          protocol.OpenAICompletions,
		OpenAICompatible: protocol.OpenAICompletions,
	}
	unsupported := map[ID]struct{}{Codex: {}, Claude: {}, Antigravity: {}, Grok: {}}

	seen := make(map[ID]struct{}, len(registry.List()))
	for _, descriptor := range registry.List() {
		seen[descriptor.ID] = struct{}{}
		want, supported := expected[descriptor.ID]
		_, isUnsupported := unsupported[descriptor.ID]
		if !supported && !isUnsupported {
			t.Errorf("channel %q is missing from the probe support matrix", descriptor.ID)
			continue
		}
		target := probeResolvedTarget(t, registry, descriptor.ID)
		if isUnsupported {
			if probeProtocol, ok := target.ProbeProtocol(); ok {
				t.Errorf("%q unexpectedly declares probe protocol %q", descriptor.ID, probeProtocol)
			}
			if _, _, ok := target.ProbeRoute("probe-upstream"); ok {
				t.Errorf("%q unexpectedly resolves a probe route", descriptor.ID)
			}
			continue
		}
		probeProtocol, ok := target.ProbeProtocol()
		if !ok {
			t.Errorf("%q declares no probe protocol", descriptor.ID)
			continue
		}
		if probeProtocol != want {
			t.Errorf("%q probe protocol = %q, want %q", descriptor.ID, probeProtocol, want)
		}
		if !probeProtocol.SupportsGeneratedText() {
			t.Errorf("%q probe protocol %q does not carry generated text", descriptor.ID, probeProtocol)
		}
		routeProtocol, mode, routeOK := target.ProbeRoute("probe-upstream")
		if !routeOK || routeProtocol != probeProtocol || !mode.Valid() {
			t.Errorf("%q probe route = %q, %q, %t; want %q and a valid mode",
				descriptor.ID, routeProtocol, mode, routeOK, probeProtocol)
		}
	}
	for id := range expected {
		if _, ok := seen[id]; !ok {
			t.Errorf("expected probe-support channel %q is not registered", id)
		}
	}
	for id := range unsupported {
		if _, ok := seen[id]; !ok {
			t.Errorf("expected unsupported channel %q is not registered", id)
		}
	}
}

// TestProbeContractIsNotTheProtocolOrderPick proves the probe protocol is the
// explicit channel declaration, not the DataPlaneProtocols order or the
// PreferredProtocol fallback.
func TestProbeContractIsNotTheProtocolOrderPick(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	target := probeResolvedTarget(t, registry, OpenAI)
	probeProtocol, ok := target.ProbeProtocol()
	if !ok || probeProtocol != protocol.OpenAIResponses {
		t.Fatalf("openai probe protocol = %q, %t; want openai-responses", probeProtocol, ok)
	}
	preferred, _ := target.PreferredProtocol(execution.OperationProbe, "probe-upstream")
	if preferred == probeProtocol {
		t.Fatalf("openai probe protocol follows the PreferredProtocol order pick %q", preferred)
	}
}

// TestProbeRouteResolvesModelDependentContract proves the declared probe route
// keeps model-dependent channel behavior for the single probe protocol.
func TestProbeRouteResolvesModelDependentContract(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	target := probeResolvedTarget(t, registry, GoogleVertex)

	nativeProtocol, nativeMode, ok := target.ProbeRoute("publishers/google/models/gemini-test")
	if !ok || nativeProtocol != protocol.Gemini || nativeMode != RouteNative {
		t.Fatalf("vertex gemini probe route = %q, %q, %t; want gemini/native", nativeProtocol, nativeMode, ok)
	}
	convertedProtocol, convertedMode, ok := target.ProbeRoute("vendor-model")
	if !ok || convertedProtocol != protocol.Gemini || convertedMode != RouteConverted {
		t.Fatalf("vertex custom probe route = %q, %q, %t; want gemini/converted", convertedProtocol, convertedMode, ok)
	}
}

func TestProbeContractValidationRejectsImplicitProtocols(t *testing.T) {
	t.Parallel()

	base := findModule(t, builtInModules(), OpenAICompatible)

	nonGenerative := base
	nonGenerative.Definition.Provider.ProbeContract.Protocol = protocol.OpenAIEmbeddings
	if _, err := compileBuiltInModules([]spec.Module{nonGenerative}); err == nil {
		t.Fatal("compileBuiltInModules accepted a non-generative probe protocol")
	}

	noRoute := base
	filteredRoutes := make([]spec.Route, 0, len(noRoute.Definition.Routes))
	for _, route := range noRoute.Definition.Routes {
		if route.Operation != execution.OperationProbe {
			filteredRoutes = append(filteredRoutes, route)
		}
	}
	noRoute.Definition.Routes = filteredRoutes
	if _, err := compileBuiltInModules([]spec.Module{noRoute}); err == nil {
		t.Fatal("compileBuiltInModules accepted a probe contract without a probe route")
	}

	noDeclaration := base
	noDeclaration.Definition.Provider.ProbeContract = spec.ProbeContract{}
	if _, err := compileBuiltInModules([]spec.Module{noDeclaration}); err == nil {
		t.Fatal("compileBuiltInModules accepted a channel without a declared probe contract")
	}
}

var probeChannelParams = []json.RawMessage{
	nil,
	[]byte(`{"base_url":"https://probe.example/v1"}`),
	[]byte(`{"endpoint":"https://probe.example/openai"}`),
	[]byte(`{"region":"us-east-1"}`),
	[]byte(`{"location":"global"}`),
}

func probeResolvedTarget(t *testing.T, registry *Registry, id ID) ResolvedTarget {
	t.Helper()
	for _, params := range probeChannelParams {
		target, err := registry.Resolve(id, params)
		if err == nil {
			return target
		}
	}
	t.Fatalf("channel %q could not be resolved for a probe target", id)
	return ResolvedTarget{}
}
