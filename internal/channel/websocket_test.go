package channel

import (
	"testing"
)

func TestResponsesWebsocketCapabilitiesAreIndependent(t *testing.T) {
	registry := NewRegistry()
	for _, test := range []struct {
		id                        ID
		params                    string
		native, stored, multiplex bool
	}{
		{id: OpenAI, params: `{"base_url":"https://example.test"}`, native: true, stored: true, multiplex: true},
		{id: XAI, params: `{"base_url":"https://example.test"}`, native: true, stored: true},
		// Codex 与 Grok 是固定官方端点的订阅渠道，不接受 base_url。
		{id: Codex, params: `{}`, native: true},
		{id: CLIProxyAPI, params: `{"base_url":"https://example.test"}`, native: true},
		{id: Sub2API, params: `{"base_url":"https://example.test"}`, native: true},
		{id: GPTLoad, params: `{"base_url":"https://example.test"}`, native: true, stored: true, multiplex: true},
		{id: NewAPI, params: `{"base_url":"https://example.test"}`},
		{id: Grok, params: `{}`},
	} {
		t.Run(string(test.id), func(t *testing.T) {
			target, err := registry.Resolve(test.id, []byte(test.params))
			if err != nil {
				t.Fatal(err)
			}
			c := target.ResponsesWebsocket
			if c.Native != test.native || c.StoredResponses != test.stored || c.Multiplex != test.multiplex ||
				c.Continuation != test.native || c.Prewarm != test.native {
				t.Fatalf("unexpected WS capabilities: %+v", c)
			}
		})
	}
}
