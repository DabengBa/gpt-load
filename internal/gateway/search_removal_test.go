package gateway

import "testing"

func TestRemovedCodexSearchEndpoint(t *testing.T) {
	t.Parallel()
	for _, endpoint := range dataPlaneEndpointCatalog() {
		if endpoint.path == "/v1/alpha/search" {
			t.Fatal("removed Codex search endpoint remains registered")
		}
	}
}
