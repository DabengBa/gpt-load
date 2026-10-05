package gateway

import (
	"gpt-load/internal/protocol"
	"net/http/httptest"
	"testing"
)

func TestGeminiNativeEmbeddingActions(t *testing.T) {
	for _, action := range []string{"embedContent", "batchEmbedContents"} {
		request := httptest.NewRequest("POST", "/v1beta/models/public:"+action, nil)
		for _, endpoint := range dataPlaneEndpointCatalog() {
			if endpoint.name != "data.gemini.generate" {
				continue
			}
			if !endpoint.pathValidator(request) {
				t.Errorf("%s rejected", action)
			}
			if got := endpoint.resolve(request).Protocol; got != protocol.Protocol("gemini-embeddings") {
				t.Errorf("%s protocol = %s", action, got)
			}
		}
	}
}
