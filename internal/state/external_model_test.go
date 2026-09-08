package state

import (
	"strings"
	"testing"
)

func TestExternalModelNamePrefersAliasThenUpstreamID(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		upstreamID string
		alias      string
		want       string
	}{
		{name: "alias wins", upstreamID: "provider-a", alias: "public", want: "public"},
		{name: "blank alias falls back to upstream id", upstreamID: "provider-a", alias: "  ", want: "provider-a"},
		{name: "values are trimmed", upstreamID: " provider-a ", alias: " public ", want: "public"},
		{name: "blank alias and id stay blank", upstreamID: "  ", alias: "", want: ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := ExternalModelName(test.upstreamID, test.alias); got != test.want {
				t.Fatalf("ExternalModelName(%q, %q) = %q, want %q", test.upstreamID, test.alias, got, test.want)
			}
		})
	}
}

func TestValidateModelRouteEntriesAcceptsDefaultsAndBoundaries(t *testing.T) {
	t.Parallel()
	maxWeight := MaxWeight
	zero := 0
	tests := []struct {
		name   string
		models []ModelConfig
	}{
		{
			name:   "unset weight and priority use defaults",
			models: []ModelConfig{{ID: "a", Alias: "x"}, {ID: "b"}},
		},
		{
			name: "boundary weights and priorities keep the external model routable",
			models: []ModelConfig{
				{ID: "a", Alias: "x", Weight: &maxWeight, Priority: intPointer(1)},
				{ID: "b", Alias: "x", Weight: &zero, Priority: intPointer(2)},
			},
		},
		{
			name: "same external name may map to different upstream models",
			models: []ModelConfig{
				{ID: "a", Alias: "x", Weight: intPointer(1)},
				{ID: "b", Alias: "x"},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := ValidateModelRouteEntries("group 7 (route)", test.models); err != nil {
				t.Fatalf("ValidateModelRouteEntries() error = %v", err)
			}
		})
	}
}

func TestValidateModelRouteEntriesRejectsInvalidEntries(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		models  []ModelConfig
		wantErr string
	}{
		{
			name:    "empty upstream id",
			models:  []ModelConfig{{ID: "  ", Alias: "x"}},
			wantErr: `group 7 (route) model entry 0: model id is required`,
		},
		{
			name:    "negative weight",
			models:  []ModelConfig{{ID: "a", Weight: intPointer(-1)}},
			wantErr: `group 7 (route) model "a": weight must be between 0 and 100`,
		},
		{
			name:    "weight above manual weight limit",
			models:  []ModelConfig{{ID: "a", Weight: intPointer(MaxWeight + 1)}},
			wantErr: `group 7 (route) model "a": weight must be between 0 and 100`,
		},
		{
			name:    "priority below one",
			models:  []ModelConfig{{ID: "a", Priority: intPointer(0)}},
			wantErr: `group 7 (route) model "a": priority must be at least 1`,
		},
		{
			name:    "duplicate external and upstream pair",
			models:  []ModelConfig{{ID: "a", Alias: "x"}, {ID: " a ", Alias: "x"}},
			wantErr: `group 7 (route) has duplicate route entry for external model "x" and upstream model "a"`,
		},
		{
			name: "external model with only zero weights is unroutable",
			models: []ModelConfig{
				{ID: "a", Alias: "x", Weight: intPointer(0)},
				{ID: "b", Alias: "x", Weight: intPointer(0)},
			},
			wantErr: `group 7 (route) external model "x" entry weights must sum to a positive value`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateModelRouteEntries("group 7 (route)", test.models)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("ValidateModelRouteEntries() error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}
