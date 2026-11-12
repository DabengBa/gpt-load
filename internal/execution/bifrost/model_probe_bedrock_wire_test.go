package bifrost

import (
	"context"
	"encoding/json"
	"net/http"
	"reflect"
	"testing"

	"github.com/maximhq/bifrost/core/providers/bedrock"
	"github.com/maximhq/bifrost/core/schemas"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestConvertedBedrockPreparedModelProbeMatchesOrdinaryConversion(t *testing.T) {
	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.AWSBedrock, protocol.OpenAICompletions, execution.OperationProbe, "", "", nil)
	spec.TargetConfig = json.RawMessage(`{"region":"us-east-1"}`)
	spec.Credential = execution.NewCredentialSnapshot(11, 1, 1, []byte(`{"api_key":"`+testAPIKey+`"}`))
	spec.ClientModel, spec.UpstreamModel = "probe-client", "anthropic.claude-3-haiku"
	spec.ProbePreparedBody = []byte(`{"model":"anthropic.claude-3-haiku","messages":[{"role":"user","content":"What is 2 + 2? Please answer briefly."}],"temperature":0.3}`)
	spec.ConfiguredParameters = []string{"temperature", "max_tokens"}
	spec = freezeTestAttempt(spec)
	prepared, failure := runtime.prepare(spec, false)
	if failure != nil {
		t.Fatalf("prepare() failure = %+v error=%+v", failure, failure.Error)
	}
	if prepared.mode != channel.RouteConverted || prepared.request == nil {
		t.Fatalf("prepared = %#v", prepared)
	}
	ordinary := spec.Clone()
	ordinary.Operation = execution.OperationChatCompletion
	ordinary.Method, ordinary.Path = http.MethodPost, "/v1/chat/completions"
	ordinary.Body, ordinary.ProbePreparedBody = ordinary.ProbePreparedBody, nil
	ordinary = freezeTestAttempt(ordinary)
	ordinaryPrepared, failure := runtime.prepare(ordinary, false)
	if failure != nil || ordinaryPrepared.request == nil {
		t.Fatalf("ordinary prepare = %#v failure = %#v", ordinaryPrepared, failure)
	}
	if prepared.request.Params == nil || prepared.request.Params.MaxCompletionTokens != nil {
		t.Fatalf("probe conversion input must preserve removed max_tokens: %#v", prepared.request.Params)
	}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()
	wire, err := bedrock.ToBedrockChatCompletionRequest(ctx, prepared.request)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	ordinaryResult, err := bedrock.ToBedrockChatCompletionRequest(ctx, ordinaryPrepared.request)
	if err != nil {
		t.Fatal(err)
	}
	ordinaryRaw, err := json.Marshal(ordinaryResult)
	if err != nil {
		t.Fatal(err)
	}
	var ordinaryBody map[string]any
	if err := json.Unmarshal(ordinaryRaw, &ordinaryBody); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(body, ordinaryBody) {
		t.Fatalf("probe SDK serialization %s differs from ordinary %s", raw, ordinaryRaw)
	}
	config, _ := body["inferenceConfig"].(map[string]any)
	if config["temperature"] != 0.3 {
		t.Errorf("converted SDK output missing temperature: %s", raw)
	}
	if config["maxTokens"] != float64(4096) {
		t.Errorf("converted SDK default maxTokens = %v, want 4096", config["maxTokens"])
	}
}
