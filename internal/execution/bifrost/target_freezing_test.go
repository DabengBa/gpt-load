package bifrost

import (
	"encoding/json"
	"strings"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
)

func TestRuntimeRejectsFrozenTargetConfigProviderAndRouteMismatches(t *testing.T) {
	registry := channel.NewRegistry()
	config := effectiveConfigForTest(t, registry, channel.OpenAI, nil)
	runtime, err := newConfiguredRuntime(t.Context(), runtimeOptions{allowPrivateNetwork: true}, registry, config)
	if err != nil {
		t.Fatalf("newConfiguredRuntime() error = %v", err)
	}
	defer runtime.Shutdown()

	base := openAIResponsesSpec(execution.OperationResponsesCreate, "POST", "/v1/responses")
	tests := []struct {
		name          string
		mutate        func(*execution.AttemptSpec)
		wantReason    string
		wantErrorCode string
	}{
		{
			name: "target config",
			mutate: func(spec *execution.AttemptSpec) {
				spec.TargetConfig = json.RawMessage(`{"base_url":"https://relay.example"}`)
			},
			wantReason: "execution target does not match provider runtime",
		},
		{
			name: "provider binding",
			mutate: func(spec *execution.AttemptSpec) {
				spec.ChannelID = string(channel.Gemini)
				spec.RouteMode = execution.RouteConverted
			},
			wantReason: "execution target does not match provider runtime",
		},
		{
			name:          "route mode",
			mutate:        func(spec *execution.AttemptSpec) { spec.RouteMode = execution.RouteConverted },
			wantReason:    "channel does not support the requested route",
			wantErrorCode: execution.ErrorCodeTargetConversionNotSupported,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := base.Clone()
			test.mutate(&spec)
			_, failure := runtime.prepare(spec, false)
			if failure == nil || failure.Error == nil || failure.DispatchState != execution.DispatchNotSent {
				t.Fatalf("prepare() failure = %+v", failure)
			}
			if !strings.Contains(failure.Error.Summary, test.wantReason) {
				t.Fatalf("failure summary = %q, want %q", failure.Error.Summary, test.wantReason)
			}
			if test.wantErrorCode != "" && failure.Error.Code != test.wantErrorCode {
				t.Fatalf("failure code = %q, want %q", failure.Error.Code, test.wantErrorCode)
			}
		})
	}
}
