package bifrost

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/protocol"
)

func TestPreparedGatewayResponsesProbeAcceptsCompletedSSE(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["temperature"] != 0.3 || body["stream"] != false {
			t.Errorf("prepared request = %#v", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(responsesSSEReadyTerminal()))
	}))
	defer server.Close()
	manager, spec := gatewayProbeForTest(t, channel.NewAPI, protocol.OpenAIResponses, server.URL)
	spec.ProbePreparedBody = []byte(`{"model":"probe-upstream","input":[{"role":"user","content":[{"type":"input_text","text":"What is 2 + 2? Please answer briefly."}]}],"max_output_tokens":128,"temperature":0.3,"store":false,"stream":false}`)
	result := manager.Execute(t.Context(), spec)
	if err := result.Validate(); err != nil || result.Error != nil || !result.ProbeAnswerPresent || result.ProbeResponseInvalid {
		t.Fatalf("prepared SSE result = %+v; validation = %v", result, err)
	}
}
