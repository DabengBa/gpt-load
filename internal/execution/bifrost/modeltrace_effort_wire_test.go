package bifrost

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestModelTraceEffortNativeWire(t *testing.T) {
	cases := []struct {
		name                   string
		channel                channel.ID
		protocol               protocol.Protocol
		model, field, response string
		want                   any
	}{
		{"responses", channel.OpenAI, protocol.OpenAIResponses, "gpt-5.4", "reasoning.effort", openAIResponsesConvertedFixture, "low"},
		{"responses pro", channel.OpenAI, protocol.OpenAIResponses, "gpt-5-pro", "reasoning.effort", openAIResponsesConvertedFixture, "high"},
		{"responses pro snapshot", channel.OpenAI, protocol.OpenAIResponses, "gpt-5-pro-2025-10-06", "reasoning.effort", openAIResponsesConvertedFixture, "high"},
		{"responses unknown alias", channel.OpenAI, protocol.OpenAIResponses, "my-gpt-5-pro-alias", "reasoning.effort", openAIResponsesConvertedFixture, "low"},
		{"anthropic", channel.Anthropic, protocol.Anthropic, "claude-sonnet-4-6", "output_config.effort", anthropicResponsesConvertedFixture, "low"},
		{"anthropic opus", channel.Anthropic, protocol.Anthropic, "claude-opus-4-5-20251101", "output_config.effort", anthropicResponsesConvertedFixture, "low"},
		{"anthropic haiku", channel.Anthropic, protocol.Anthropic, "claude-haiku-4-5-20251001", "output_config.effort", anthropicResponsesConvertedFixture, "anthropic budget"},
		{"anthropic sonnet 4.5", channel.Anthropic, protocol.Anthropic, "claude-sonnet-4-5-20250929", "output_config.effort", anthropicResponsesConvertedFixture, "anthropic budget"},
		{"gemini level", channel.Gemini, protocol.Gemini, "gemini-3-flash", "generationConfig.thinkingConfig.thinkingLevel", geminiResponsesConvertedFixture, "low"},
		{"gemini budget", channel.Gemini, protocol.Gemini, "gemini-2.5-flash", "generationConfig.thinkingConfig.thinkingBudget", geminiResponsesConvertedFixture, "budget"},
		{"gemini pro budget", channel.Gemini, protocol.Gemini, "gemini-2.5-pro", "generationConfig.thinkingConfig.thinkingBudget", geminiResponsesConvertedFixture, "budget"},
		{"gemini without thinking", channel.Gemini, protocol.Gemini, "gemini-2.0-flash", "generationConfig.thinkingConfig", geminiResponsesConvertedFixture, "none"},
	}
	for _, tc := range cases {
		for _, op := range []execution.Operation{execution.OperationModelTrace, execution.OperationProbe} {
			t.Run(tc.name+"/"+string(op), func(t *testing.T) {
				wire := make(chan []byte, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					body, _ := io.ReadAll(r.Body)
					wire <- body
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, tc.response)
				}))
				defer server.Close()
				options := testRuntimeOptions{allowPrivateNetwork: true}
				switch tc.channel {
				case channel.OpenAI:
					options.openAIBaseURL = server.URL
				case channel.Anthropic:
					options.anthropicBaseURL = server.URL
				case channel.Gemini:
					options.geminiBaseURL = server.URL + "/v1beta"
				}
				runtime := newProtocolTestRuntime(t, options)
				spec := utilitySpec(tc.channel, tc.protocol, op, "", "", nil)
				spec.ClientModel, spec.UpstreamModel = tc.model, tc.model
				spec.PromptText, spec.ProbeMaxOutputTokens = "challenge prompt", 2048
				result := runtime.Execute(t.Context(), freezeTestAttempt(spec))
				if result.Error != nil || result.StatusCode != http.StatusOK {
					t.Fatalf("execute: %+v", result)
				}
				select {
				case body := <-wire:
					value := gjson.GetBytes(body, tc.field)
					if op == execution.OperationProbe {
						if value.Exists() || gjson.GetBytes(body, "generationConfig.thinkingConfig").Exists() || gjson.GetBytes(body, "thinking").Exists() {
							t.Fatalf("ordinary probe has effort: %s", body)
						}
						if tc.protocol == protocol.Gemini && gjson.GetBytes(body, "generationConfig.maxOutputTokens").Int() != 2048 {
							t.Fatalf("ordinary probe output limit changed: %s", body)
						}
						return
					}
					if !strings.Contains(string(body), "challenge prompt") {
						t.Errorf("challenge prompt missing: %s", body)
					}
					if tc.want == "none" {
						if value.Exists() {
							t.Fatalf("model without thinking: %s", body)
						}
					} else if tc.want == "budget" {
						if !value.Exists() || value.Int() <= 0 || gjson.GetBytes(body, "generationConfig.maxOutputTokens").Int()-value.Int() < 2048 || gjson.GetBytes(body, "generationConfig.thinkingConfig.thinkingLevel").Exists() {
							t.Fatalf("Gemini budget mapping: %s", body)
						}
					} else if tc.want == "anthropic budget" {
						budget := gjson.GetBytes(body, "thinking.budget_tokens").Int()
						if value.Exists() || gjson.GetBytes(body, "thinking.type").String() != "enabled" || budget != 1177 || budget >= gjson.GetBytes(body, "max_tokens").Int() {
							t.Fatalf("Anthropic budget mapping: %s", body)
						}
					} else if value.String() != tc.want {
						t.Fatalf("field %s = %s, want %v; body=%s", tc.field, value.Raw, tc.want, body)
					}
				default:
					t.Fatal("no provider request")
				}
			})
		}
	}
}

func TestModelTraceGatewayAnthropicLegacyEffort(t *testing.T) {
	for _, model := range []string{"claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929"} {
		t.Run(model, func(t *testing.T) {
			wire := make(chan []byte, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				wire <- body
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, anthropicResponsesConvertedFixture)
			}))
			defer server.Close()
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			spec := utilitySpec(channel.GPTLoad, protocol.Anthropic, execution.OperationModelTrace, "", "", nil)
			spec.TargetConfig = json.RawMessage(`{"base_url":"` + server.URL + `"}`)
			spec.ClientModel, spec.UpstreamModel = model, model
			spec.PromptText, spec.ProbeMaxOutputTokens = "challenge prompt", 2048
			result := runtime.Execute(t.Context(), freezeTestAttempt(spec))
			if result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("execute: %+v", result)
			}
			select {
			case body := <-wire:
				if gjson.GetBytes(body, "output_config.effort").Exists() || gjson.GetBytes(body, "thinking.type").String() != "enabled" || gjson.GetBytes(body, "thinking.budget_tokens").Int() != 1177 {
					t.Fatalf("gateway Anthropic budget mapping: %s", body)
				}
			default:
				t.Fatal("no provider request")
			}
		})
	}
}

func TestModelTraceAnthropicInsufficientOutputBudget(t *testing.T) {
	spec := execution.AttemptSpec{Operation: execution.OperationModelTrace, UpstreamModel: "claude-haiku-4-5-20251001"}
	body, err := applyModelTraceEffort(spec, protocol.Anthropic, []byte(`{"max_tokens":1024}`))
	if err != nil || gjson.GetBytes(body, "thinking").Exists() || gjson.GetBytes(body, "output_config.effort").Exists() {
		t.Fatalf("insufficient budget must omit reasoning controls: body=%s err=%v", body, err)
	}
}

func TestModelTraceReasoningCapabilityOverrides(t *testing.T) {
	yes, no := true, false
	schemas.SetCapabilityResolver(func(provider schemas.ModelProvider, model string) *schemas.ModelCapabilities {
		switch {
		case provider == schemas.Anthropic && model == "custom-effort":
			return &schemas.ModelCapabilities{SupportsNativeEffort: &yes}
		case provider == schemas.Anthropic && model == "claude-sonnet-4-6":
			return &schemas.ModelCapabilities{SupportsNativeEffort: &no}
		case provider == schemas.Vertex && model == "gemini-2.0-flash":
			return &schemas.ModelCapabilities{SupportsReasoning: &yes}
		case provider == schemas.Gemini && model == "gemini-3-flash":
			return &schemas.ModelCapabilities{SupportsReasoning: &no}
		case provider == schemas.OpenAI && model == "gpt-5-pro":
			return &schemas.ModelCapabilities{ReasoningEffortLevels: []string{"low", "high"}}
		case provider == schemas.OpenAI && model == "custom-known":
			return &schemas.ModelCapabilities{ReasoningEffortLevels: []string{"high"}}
		case provider == schemas.Gemini && model == "gemini-2.5-flash":
			return &schemas.ModelCapabilities{MaxOutputTokens: schemas.Ptr(4096)}
		}
		return nil
	})
	t.Cleanup(func() { schemas.SetCapabilityResolver(nil) })
	for _, tc := range []struct {
		name               string
		channel            channel.ID
		protocol           protocol.Protocol
		model, body, field string
		want               bool
	}{
		{"anthropic override enables", channel.Anthropic, protocol.Anthropic, "custom-effort", `{"max_tokens":2048}`, "output_config.effort", true},
		{"anthropic override disables", channel.Anthropic, protocol.Anthropic, "claude-sonnet-4-6", `{"max_tokens":2048}`, "output_config.effort", false},
		{"vertex override enables", channel.GoogleVertex, protocol.Gemini, "gemini-2.0-flash", `{"generationConfig":{"maxOutputTokens":2048}}`, "generationConfig.thinkingConfig", true},
		{"gemini override disables", channel.Gemini, protocol.Gemini, "gemini-3-flash", `{"generationConfig":{"maxOutputTokens":2048}}`, "generationConfig.thinkingConfig", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spec := execution.AttemptSpec{Operation: execution.OperationModelTrace, ChannelID: string(tc.channel), UpstreamModel: tc.model}
			body, err := applyModelTraceEffort(spec, tc.protocol, []byte(tc.body))
			if err != nil || gjson.GetBytes(body, tc.field).Exists() != tc.want {
				t.Fatalf("capability override: body=%s err=%v", body, err)
			}
		})
	}
	for _, tc := range []struct{ model, want string }{{"gpt-5-pro", "low"}, {"custom-known", "high"}} {
		spec := execution.AttemptSpec{Operation: execution.OperationModelTrace, ChannelID: string(channel.OpenAI), UpstreamModel: tc.model}
		body, err := applyModelTraceEffort(spec, protocol.OpenAIResponses, []byte(`{}`))
		if err != nil || gjson.GetBytes(body, "reasoning.effort").String() != tc.want {
			t.Errorf("Responses capability override: model=%s body=%s err=%v", tc.model, body, err)
		}
	}
	spec := execution.AttemptSpec{Operation: execution.OperationModelTrace, ChannelID: string(channel.Gemini), UpstreamModel: "gemini-2.5-flash"}
	body, err := applyModelTraceEffort(spec, protocol.Gemini, []byte(`{"generationConfig":{"maxOutputTokens":2048}}`))
	if err != nil || gjson.GetBytes(body, "generationConfig.thinkingConfig").Exists() || gjson.GetBytes(body, "generationConfig.maxOutputTokens").Int() != 2048 {
		t.Errorf("known output cap cannot fit full budget: body=%s err=%v", body, err)
	}
}

func TestModelTraceGatewayAndCompatibleWire(t *testing.T) {
	cases := []struct {
		name                         string
		channel                      channel.ID
		protocol                     protocol.Protocol
		model, field, response, want string
	}{
		{"compatible chat", channel.OpenAICompatible, protocol.OpenAICompletions, "model", "reasoning_effort", `{"id":"chat_1","object":"chat.completion","created":1,"model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, "low"},
		{"gateway responses", channel.GPTLoad, protocol.OpenAIResponses, "gpt-5.4", "reasoning.effort", openAIResponsesConvertedFixture, "low"},
		{"gateway responses pro", channel.GPTLoad, protocol.OpenAIResponses, "gpt-5-pro", "reasoning.effort", openAIResponsesConvertedFixture, "high"},
		{"gateway responses unknown alias", channel.GPTLoad, protocol.OpenAIResponses, "my-gpt-5-pro-alias", "reasoning.effort", openAIResponsesConvertedFixture, "low"},
		{"gateway anthropic", channel.GPTLoad, protocol.Anthropic, "claude-sonnet-4-6", "output_config.effort", anthropicResponsesConvertedFixture, "low"},
		{"gateway gemini", channel.GPTLoad, protocol.Gemini, "gemini-3-flash", "generationConfig.thinkingConfig.thinkingLevel", geminiResponsesConvertedFixture, "low"},
		{"gateway gemini flash budget", channel.GPTLoad, protocol.Gemini, "gemini-2.5-flash", "generationConfig.thinkingConfig.thinkingBudget", geminiResponsesConvertedFixture, "budget"},
		{"gateway gemini pro budget", channel.GPTLoad, protocol.Gemini, "gemini-2.5-pro", "generationConfig.thinkingConfig.thinkingBudget", geminiResponsesConvertedFixture, "budget"},
		{"gateway gemini without thinking", channel.GPTLoad, protocol.Gemini, "gemini-2.0-flash", "generationConfig.thinkingConfig", geminiResponsesConvertedFixture, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			wire := make(chan []byte, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				wire <- body
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.response)
			}))
			defer server.Close()
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			spec := utilitySpec(tc.channel, tc.protocol, execution.OperationModelTrace, "", "", nil)
			spec.TargetConfig = json.RawMessage(`{"base_url":"` + server.URL + `"}`)
			spec.ClientModel, spec.UpstreamModel = tc.model, tc.model
			spec.PromptText, spec.ProbeMaxOutputTokens = "challenge prompt", 2048
			result := runtime.Execute(t.Context(), freezeTestAttempt(spec))
			if result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("execute: %+v", result)
			}
			select {
			case body := <-wire:
				if got := gjson.GetBytes(body, tc.field); tc.want == "budget" {
					if got.Int() <= 0 || gjson.GetBytes(body, "generationConfig.maxOutputTokens").Int()-got.Int() < 2048 {
						t.Errorf("gateway budget leaves insufficient answer tokens: %s", body)
					}
				} else if got.String() != tc.want || (tc.want == "" && got.Exists()) {
					t.Errorf("%s = %q; body=%s", tc.field, got, body)
				}
				if !strings.Contains(string(body), "challenge prompt") {
					t.Errorf("prompt missing: %s", body)
				}
			default:
				t.Fatal("no provider request")
			}
		})
	}
}

func TestModelTraceGatewayOrdinaryProbeUnchanged(t *testing.T) {
	for _, tc := range []struct {
		protocol        protocol.Protocol
		model, response string
	}{
		{protocol.OpenAIResponses, "gpt-5-pro", openAIResponsesConvertedFixture},
		{protocol.Gemini, "gemini-2.5-pro", geminiResponsesConvertedFixture},
	} {
		t.Run(string(tc.protocol), func(t *testing.T) {
			wire := make(chan []byte, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				wire <- body
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.response)
			}))
			defer server.Close()
			runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
			spec := utilitySpec(channel.GPTLoad, tc.protocol, execution.OperationProbe, "", "", nil)
			spec.TargetConfig = json.RawMessage(`{"base_url":"` + server.URL + `"}`)
			spec.ClientModel, spec.UpstreamModel = tc.model, tc.model
			spec.ProbeMaxOutputTokens = 2048
			result := runtime.Execute(t.Context(), freezeTestAttempt(spec))
			if result.Error != nil || result.StatusCode != http.StatusOK {
				t.Fatalf("execute: %+v", result)
			}
			select {
			case body := <-wire:
				if gjson.GetBytes(body, "reasoning.effort").Exists() || gjson.GetBytes(body, "generationConfig.thinkingConfig").Exists() {
					t.Fatalf("ordinary gateway probe has reasoning controls: %s", body)
				}
				if tc.protocol == protocol.Gemini && gjson.GetBytes(body, "generationConfig.maxOutputTokens").Int() != 2048 {
					t.Fatalf("ordinary gateway probe budget changed: %s", body)
				}
			default:
				t.Fatal("no provider request")
			}
		})
	}
}

func TestModelTraceTypedResponsesWire(t *testing.T) {
	for _, model := range []string{"openai/gpt-5.4", "openai/gpt-5-pro"} {
		for _, op := range []execution.Operation{execution.OperationModelTrace, execution.OperationProbe} {
			t.Run(model+"/"+string(op), func(t *testing.T) {
				wire := make(chan []byte, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					body, _ := io.ReadAll(r.Body)
					wire <- body
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, openAIResponsesConvertedFixture)
				}))
				defer server.Close()
				runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
				runtime.baseURLs[channel.OpenRouter] = server.URL
				spec := utilitySpec(channel.OpenRouter, protocol.OpenAIResponses, op, "", "", nil)
				spec.ClientModel, spec.UpstreamModel = model, model
				spec.PromptText, spec.ProbeMaxOutputTokens = "challenge prompt", 2048
				result := runtime.Execute(t.Context(), freezeTestAttempt(spec))
				if result.Error != nil || result.StatusCode != http.StatusOK {
					t.Fatalf("execute: %+v", result)
				}
				select {
				case body := <-wire:
					got := gjson.GetBytes(body, "reasoning.effort")
					want := "low"
					if model == "openai/gpt-5-pro" {
						want = "high"
					}
					if op == execution.OperationModelTrace && got.String() != want {
						t.Errorf("challenge body: %s", body)
					}
					if op == execution.OperationProbe && got.Exists() {
						t.Errorf("ordinary probe body: %s", body)
					}
				default:
					t.Fatal("no provider request")
				}
			})
		}
	}
}

func TestModelTraceVertexPreparationEffort(t *testing.T) {
	runtime := newProtocolTestRuntime(t, testRuntimeOptions{allowPrivateNetwork: true})
	spec := utilitySpec(channel.GoogleVertex, protocol.Gemini, execution.OperationModelTrace, "", "", nil)
	spec.TargetConfig = json.RawMessage(`{"location":"us-central1"}`)
	spec.Credential = execution.NewCredentialSnapshot(12, 1, 1, []byte(`{"service_account_json":"{\"type\":\"service_account\",\"project_id\":\"probe-project\",\"client_email\":\"svc@example.iam.gserviceaccount.com\",\"private_key\":\"secret\"}"}`))
	spec.ClientModel, spec.UpstreamModel = "client", "gemini-2.5-flash"
	spec.PromptText, spec.ProbeMaxOutputTokens = "challenge prompt", 2048
	prepared, failure := runtime.prepare(freezeTestAttempt(spec), false)
	if failure != nil || prepared.passthrough == nil {
		t.Fatalf("prepare: failure=%+v prepared=%+v", failure, prepared)
	}
	body := prepared.passthrough.Body
	if v := gjson.GetBytes(body, "generationConfig.thinkingConfig.thinkingBudget"); !v.Exists() || v.Int() <= 0 || gjson.GetBytes(body, "generationConfig.maxOutputTokens").Int()-v.Int() < 2048 {
		t.Fatalf("Vertex budget: %s", body)
	}
	if gjson.GetBytes(body, "generationConfig.thinkingConfig.thinkingLevel").Exists() || !strings.Contains(string(body), "challenge prompt") {
		t.Fatalf("Vertex body: %s", body)
	}
	spec.ClientModel, spec.UpstreamModel = "client", "gemini-2.0-flash"
	prepared, failure = runtime.prepare(freezeTestAttempt(spec), false)
	if failure != nil || prepared.passthrough == nil || gjson.GetBytes(prepared.passthrough.Body, "generationConfig.thinkingConfig").Exists() {
		t.Fatalf("Vertex without thinking: failure=%+v prepared=%+v", failure, prepared)
	}
}
