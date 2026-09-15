package bifrost

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestNonGenerativeProbePreflightRejectsBeforeDispatch(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	tests := []struct {
		name     string
		protocol protocol.Protocol
	}{
		{name: "embeddings", protocol: protocol.OpenAIEmbeddings},
		{name: "rerank", protocol: protocol.Rerank},
		{name: "images", protocol: protocol.OpenAIImages},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			spec := utilitySpec(channel.NewAPI, test.protocol, execution.OperationProbe, "", "", nil)
			spec.ClientModel, spec.UpstreamModel = "probe-client", "probe-upstream"
			target, err := json.Marshal(map[string]string{"base_url": server.URL})
			if err != nil {
				t.Fatalf("marshal target: %v", err)
			}
			spec.TargetConfig = target
			spec = freezeTestAttempt(spec)

			result := runtime.Execute(context.Background(), spec)
			if err := result.Validate(); err != nil {
				t.Fatalf("result validation: %v; result=%+v", err, result)
			}
			if result.Error == nil || result.Error.Kind != execution.ErrorKindInvalidRequest ||
				result.Error.Summary != "probe protocol does not support generated text" ||
				result.DispatchState != execution.DispatchNotSent {
				t.Fatalf("result = %+v; want explicit non-generative preflight rejection", result)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("upstream calls = %d, want zero", calls.Load())
	}
}
