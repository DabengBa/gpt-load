package cpa

import (
	"encoding/json"
	"strings"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
)

func TestAdapterRejectsFrozenTargetAndRouteMismatches(t *testing.T) {
	_, adapter, _, spec := newAntigravityImagesRuntime(t)
	base := spec.Clone()
	tests := []struct {
		name       string
		mutate     func(*execution.AttemptSpec)
		wantReason string
	}{
		{
			name: "target config",
			mutate: func(spec *execution.AttemptSpec) {
				spec.TargetConfig = json.RawMessage(`{"unknown":"value"}`)
			},
			wantReason: "resolve subscription target",
		},
		{
			name:       "route mode",
			mutate:     func(spec *execution.AttemptSpec) { spec.RouteMode = execution.RouteNative },
			wantReason: "route is not declared by the channel",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := base.Clone()
			test.mutate(&spec)
			_, err := adapter.validateSpec(spec)
			if err == nil || !strings.Contains(err.Error(), test.wantReason) {
				t.Fatalf("validateSpec() error = %v, want %q", err, test.wantReason)
			}
		})
	}
}

func TestAdapterProviderBindingComesFromOneImmutableChannelRegistry(t *testing.T) {
	registry := channel.NewRegistry()
	checked := 0
	for _, descriptor := range registry.List() {
		channelID := channel.ID(descriptor.ID)
		providerKind, ok := registry.ProviderKind(channelID)
		if !ok {
			t.Fatalf("ProviderKind(%q) returned no binding", channelID)
		}
		target, err := registry.ResolveExecutionTarget(channelID, nil)
		if err != nil {
			continue
		}
		checked++
		if target.ProviderKind != providerKind {
			t.Fatalf("channel %q provider binding = %q, target binding = %q", channelID, providerKind, target.ProviderKind)
		}
	}
	if checked == 0 {
		t.Fatal("no fixed channel target was checked")
	}
}
