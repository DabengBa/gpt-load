package bifrost

import (
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestValidateRouteCapabilityRejectsNonGenerativeProbeRoutes(t *testing.T) {
	t.Parallel()

	manager := &RuntimeManager{}
	for _, route := range []channel.RouteDescriptor{
		{ClientProtocol: protocol.OpenAIEmbeddings, Operation: execution.OperationProbe, RouteMode: execution.RouteNative},
		{ClientProtocol: protocol.OpenAIEmbeddings, Operation: execution.OperationProbe, RouteMode: execution.RouteConverted},
		{ClientProtocol: protocol.Rerank, Operation: execution.OperationProbe, RouteMode: execution.RouteNative},
		{ClientProtocol: protocol.Rerank, Operation: execution.OperationProbe, RouteMode: execution.RouteConverted},
	} {
		if err := manager.ValidateRouteCapability(channel.ProviderOpenAICompatible, route); err == nil {
			t.Errorf("non-generative probe route accepted: %#v", route)
		}
	}

	for _, route := range []channel.RouteDescriptor{
		{ClientProtocol: protocol.OpenAIEmbeddings, Operation: execution.OperationEmbeddingsCreate, RouteMode: execution.RouteNative},
		{ClientProtocol: protocol.Rerank, Operation: execution.OperationRerank, RouteMode: execution.RouteNative},
	} {
		if err := manager.ValidateRouteCapability(channel.ProviderOpenAICompatible, route); err != nil {
			t.Errorf("normal data-plane route rejected: %#v: %v", route, err)
		}
	}
}
