package channel

import (
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"testing"
)

func TestCohereRerankOnlyPreset(t *testing.T) {
	r := NewRegistry()
	d, ok := r.Get(ID("cohere"))
	if !ok {
		t.Fatal("missing Cohere preset")
	}
	if d.Connection.CredentialInput != "batch_text" || len(d.CredentialFields) != 1 {
		t.Fatal("Cohere must retain local single credential schema")
	}
	target, err := r.Resolve(ID("cohere"), nil)
	if err != nil {
		t.Fatal(err)
	}
	c, ok := target.ProbeContract()
	if !ok || c.Protocol != protocol.Rerank || c.MinOutputTokens != 0 {
		t.Fatalf("probe contract = %+v, %t", c, ok)
	}
	def, _ := r.lookup(ID("cohere"))
	if len(def.modes) != 1 || len(def.modes[protocol.Rerank]) != 2 {
		t.Fatalf("unexpected routes: %+v", def.modes)
	}
	for _, op := range []execution.Operation{execution.OperationRerank, execution.OperationProbe} {
		if def.modes[protocol.Rerank][op] != RouteNative {
			t.Fatalf("%s must be native", op)
		}
	}
	m := findModule(t, builtInModules(), ID("cohere"))
	if m.Definition.Provider.FixedBaseURL != "https://api.cohere.ai/v2" {
		t.Fatal("incorrect Cohere base URL")
	}
}
