package channel

import (
	"encoding/json"
	"reflect"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestMultiProtocolChannelContract(t *testing.T) {
	t.Parallel()
	const id = ID("multi_protocol")
	registry := NewRegistry()
	descriptor, ok := registry.Get(id)
	if !ok {
		t.Fatal("multi_protocol channel is not registered")
	}
	if descriptor.Name != "Multi-Protocol" || descriptor.Connection.CredentialInput != "batch_text" {
		t.Fatalf("identity or connection = %#v", descriptor)
	}
	if len(descriptor.ParamFields) != 1 || descriptor.ParamFields[0].Key != "base_url" ||
		!descriptor.ParamFields[0].Required || descriptor.ParamFields[0].InputKind != InputURL {
		t.Fatalf("params = %#v", descriptor.ParamFields)
	}
	if len(descriptor.CredentialFields) != 1 || descriptor.CredentialFields[0].Key != "api_key" ||
		!descriptor.CredentialFields[0].Required || !descriptor.CredentialFields[0].Sensitive || descriptor.CredentialFields[0].InputKind != InputSecret {
		t.Fatalf("credentials = %#v", descriptor.CredentialFields)
	}
	for _, raw := range []string{`{}`, `{"base_url":""}`, `{"base_url":"ftp://example.com"}`} {
		if _, err := registry.Resolve(id, json.RawMessage(raw)); err == nil {
			t.Errorf("Resolve(%s) accepted invalid base URL", raw)
		}
	}
	for _, raw := range []string{`{}`, `{"api_key":" "}`} {
		if _, err := registry.ValidateCredential(id, json.RawMessage(raw)); err == nil {
			t.Errorf("ValidateCredential(%s) accepted invalid key", raw)
		}
	}
	if _, err := registry.ValidateCredential(id, json.RawMessage(`{"api_key":"example-key"}`)); err != nil {
		t.Fatalf("ValidateCredential() = %v", err)
	}
	target, err := registry.Resolve(id, json.RawMessage(`{"base_url":"HTTPS://RELAY.EXAMPLE:443/team-a/"}`))
	if err != nil {
		t.Fatal(err)
	}
	if target.ProviderKind != ProviderMultiProtocolGateway || target.CatalogProviderID != "" || string(target.TargetConfig) != `{"base_url":"https://relay.example/team-a"}` {
		t.Fatalf("target = %#v", target)
	}
	old, _ := registry.Get(NewAPI)
	if !reflect.DeepEqual(descriptor.Routes, old.Routes) {
		t.Fatalf("route matrix differs from New API: got %#v, want %#v", descriptor.Routes, old.Routes)
	}
	contract, ok := target.ProbeContract()
	if !ok || contract.Protocol != protocol.OpenAICompletions || contract.MinOutputTokens != 128 {
		t.Fatalf("probe contract = %#v, %t", contract, ok)
	}
	if target.ResponsesStoreHandling(protocol.OpenAIResponses, execution.OperationResponsesCreate) != ResponsesStoreHandlingUpstreamManaged {
		t.Fatal("Responses create must use upstream-managed storage")
	}
	if target.SupportsResponsesLifecycle() || target.ResponsesWebsocket.Native || target.ResponsesWebsocket.StoredResponses || target.ResponsesWebsocket.Multiplex || target.ResponsesWebsocket.Continuation || target.ResponsesWebsocket.Prewarm {
		t.Fatal("undeclared Responses lifecycle or websocket support")
	}
	for _, route := range []struct {
		p  protocol.Protocol
		op execution.Operation
	}{
		{protocol.OpenAIResponses, execution.OperationResponsesRetrieve},
		{protocol.OpenAIImages, execution.OperationProbe},
		{protocol.Anthropic, execution.OperationListModels},
	} {
		if mode, ok := target.Mode(route.p, route.op); ok {
			t.Errorf("unexpected route %s/%s = %s", route.p, route.op, mode)
		}
	}
}
