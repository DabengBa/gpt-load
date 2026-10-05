package bifrost

import (
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestCohereDeclaredRerankProbeCapability(t *testing.T) {
	manager := &RuntimeManager{}
	registry := channel.NewRegistry()
	descriptor, ok := registry.Get(channel.Cohere)
	if !ok {
		t.Fatal("Cohere channel is not registered")
	}
	kind, ok := registry.ProviderKind(channel.Cohere)
	if !ok {
		t.Fatal("Cohere provider binding is missing")
	}
	var probePresent bool
	for _, route := range descriptor.Routes {
		if route.ClientProtocol == protocol.Rerank && route.Operation == execution.OperationProbe {
			probePresent = true
		}
		if err := manager.ValidateRouteCapability(kind, route); err != nil {
			t.Errorf("Cohere declared route %s/%s/%s: %v", route.ClientProtocol, route.Operation, route.RouteMode, err)
		}
	}
	if !probePresent {
		t.Fatal("Cohere rerank probe route is missing")
	}
	for _, kind := range []channel.ProviderKind{channel.ProviderOpenAI, channel.ProviderGemini, channel.ProviderAnthropic} {
		if err := manager.ValidateRouteCapability(kind, channel.RouteDescriptor{
			ClientProtocol: protocol.Rerank, Operation: execution.OperationProbe, RouteMode: execution.RouteNative,
		}); err == nil {
			t.Errorf("unsupported provider %s accepted rerank probe", kind)
		}
	}
}

func TestGeminiEmbeddingProbeCapabilityIsNativeOnly(t *testing.T) {
	manager := &RuntimeManager{}
	for _, kind := range []channel.ProviderKind{channel.ProviderGemini, channel.ProviderMultiProtocolGateway} {
		for _, operation := range []execution.Operation{execution.OperationEmbeddingsCreate, execution.OperationProbe} {
			if err := manager.ValidateRouteCapability(kind, channel.RouteDescriptor{
				ClientProtocol: protocol.GeminiEmbeddings, Operation: operation, RouteMode: execution.RouteNative,
			}); err != nil {
				t.Errorf("%s/%s: %v", kind, operation, err)
			}
		}
	}
	if err := manager.ValidateRouteCapability(channel.ProviderGemini, channel.RouteDescriptor{
		ClientProtocol: protocol.GeminiEmbeddings, Operation: execution.OperationProbe, RouteMode: execution.RouteConverted,
	}); err == nil {
		t.Fatal("unimplemented converted Gemini embedding probe accepted")
	}
}
