package channel

import (
	"testing"

	"gpt-load/internal/channel/spec"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestProbeRoutesExcludeNonGenerativeProtocols(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	for _, descriptor := range registry.List() {
		definition, ok := registry.lookup(descriptor.ID)
		if !ok {
			t.Fatalf("lookup(%q) missing", descriptor.ID)
		}
		for _, clientProtocol := range []protocol.Protocol{protocol.OpenAIEmbeddings, protocol.Rerank} {
			if _, ok := definition.modes[clientProtocol][execution.OperationProbe]; ok {
				t.Errorf("%q advertises non-generative %s probe route", descriptor.ID, clientProtocol)
			}
		}
	}
}

func TestCompileRoutesRejectsNonGenerativeProbe(t *testing.T) {
	t.Parallel()

	base := findModule(t, builtInModules(), OpenAICompatible)
	base.Definition.Routes = filterProbeRoutes(base.Definition.Routes)
	base.Definition.Routes = append(base.Definition.Routes,
		spec.NewRoute(protocol.OpenAIEmbeddings, execution.OperationProbe, execution.RouteNative),
	)
	if _, err := compileBuiltInModules([]spec.Module{base}); err == nil {
		t.Fatal("compileBuiltInModules accepted a non-generative probe route")
	}
}

func filterProbeRoutes(routes []spec.Route) []spec.Route {
	filtered := make([]spec.Route, 0, len(routes))
	for _, route := range routes {
		if route.Operation == execution.OperationProbe {
			continue
		}
		filtered = append(filtered, route)
	}
	return filtered
}
