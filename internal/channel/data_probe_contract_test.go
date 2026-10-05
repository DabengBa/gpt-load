package channel

import (
	"gpt-load/internal/channel/modules"
	"gpt-load/internal/channel/spec"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"testing"
)

func TestRerankProbeRequiresExplicitZeroBudgetContract(t *testing.T) {
	module := modules.OpenAICompatible()
	module.Definition.Provider.ProbeContract = spec.ProbeContract{Protocol: protocol.Rerank}
	module.Definition.Routes = append(module.Definition.Routes, spec.NewRoute(protocol.Rerank, execution.OperationProbe, execution.RouteNative))
	if _, err := compileBuiltInModules([]spec.Module{module}); err != nil {
		t.Fatalf("explicit rank contract rejected: %v", err)
	}
	module.Definition.Provider.ProbeContract = spec.ProbeContract{Protocol: protocol.OpenAICompletions, MinOutputTokens: 128}
	if _, err := compileBuiltInModules([]spec.Module{module}); err == nil {
		t.Fatal("rank probe without rank contract accepted")
	}
	if (spec.ProbeContract{Protocol: protocol.Rerank, MinOutputTokens: 128}).Valid() {
		t.Fatal("rank probe must not invent a generation budget")
	}
}
