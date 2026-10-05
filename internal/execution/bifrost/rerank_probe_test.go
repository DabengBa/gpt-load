package bifrost

import (
	"encoding/json"
	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRerankProbeValidatesOneRankedDocument(t *testing.T) {
	for _, response := range []string{`{"results":[]}`, `{"results":[{"index":1,"relevance_score":0.9}]}`, `{"results":[{"index":0,"relevance_score":null}]}`, `{"results":[{"index":0,"relevance_score":"0.9"}]}`, `{"results":[{"index":0,"relevance_score":-1}]}`} {
		if validDataProbeResponse(protocol.Rerank, []byte(response)) {
			t.Errorf("invalid rank accepted: %s", response)
		}
	}
}

// Cohere registration belongs to its module owner. This integration proof runs
// the common probe builder and real executor wire without registering a preset.
func TestRerankProbeCommonExecutorWire(t *testing.T) {
	for _, response := range []string{`{"results":[{"index":0,"relevance_score":0.9}]}`, `{"results":[{"index":0,"relevance_score":null}]}`} {
		t.Run(response, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v2/rerank" {
					t.Errorf("path=%s", r.URL.Path)
				}
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if string(body["model"]) != `"provider-model"` || string(body["query"]) != `"hello"` || string(body["documents"]) != `["hello"]` {
					t.Errorf("probe payload=%v", body)
				}
				for _, field := range []string{"max_tokens", "max_output_tokens", "generationConfig"} {
					if _, ok := body[field]; ok {
						t.Errorf("invented generation budget %s", field)
					}
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, response)
			}))
			t.Cleanup(server.Close)
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			spec := rerankSpec(channel.OpenAICompatible, server.URL+"/v2")
			config, failure := runtime.manager.configForAttempt(spec)
			if failure != nil {
				t.Fatal(failure)
			}
			lease, err := runtime.manager.pool.acquire(t.Context(), config)
			if err != nil {
				t.Fatal(err)
			}
			defer lease.Release()
			owner := lease.runtime.(*Runtime)
			prepared, failure := owner.prepare(spec, false)
			if failure != nil {
				t.Fatal(failure)
			}
			target, err := runtime.registry.ResolveExecutionTarget(channel.OpenAICompatible, spec.TargetConfig)
			if err != nil {
				t.Fatal(err)
			}
			spec.Operation = execution.OperationProbe
			spec.ProbeMaxOutputTokens = 0
			prepared, failure = prepareRerank(spec, target, prepared.provider, prepared.directKey, prepared.secrets)
			if failure != nil {
				t.Fatal(failure)
			}
			result := owner.executePassthrough(t.Context(), spec, prepared)
			normalizeProbeAttemptResult(spec, &result, true)
			valid := validDataProbeResponse(protocol.Rerank, []byte(response))
			if result.ProbeAnswerPresent != valid || (result.Error == nil) != valid || result.StatusCode != 200 {
				t.Fatalf("rank result=%+v", result)
			}
		})
	}
}
