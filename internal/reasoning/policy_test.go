package reasoning

import "testing"

func TestResolveEffortPriority(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		entry  string
		group  string
		client string
		want   string
		source EffortSource
	}{
		{name: "entry over group", entry: "high", group: "low", client: "medium", want: "high", source: EffortSourceEntry},
		{name: "group over client", group: "low", client: "medium", want: "low", source: EffortSourceGroup},
		{name: "client fallback", client: "medium", want: "medium", source: EffortSourceClient},
		{name: "none is explicit", entry: "none", group: "high", want: "none", source: EffortSourceEntry},
		{name: "unconfigured", want: "", source: EffortSourceProviderDefault},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, source, err := ResolveEffort(test.entry, test.group, test.client)
			if err != nil || got != test.want || source != test.source {
				t.Fatalf("ResolveEffort() = %q, %q, %v; want %q, %q", got, source, err, test.want, test.source)
			}
		})
	}
}

func TestResolveEffortRejectsInvalidConfiguredValue(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		entry  string
		group  string
		client string
	}{
		{name: "invalid entry", entry: "invalid", group: "high", client: "low"},
		{name: "invalid group", group: "unsupported", client: "low"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, source, err := ResolveEffort(test.entry, test.group, test.client)
			if err == nil || got != "" || source != "" {
				t.Fatalf("ResolveEffort() = %q, %q, %v; want invalid configured value error", got, source, err)
			}
		})
	}
}

func TestResolveEffortKeepsInvalidClientValueUnvalidated(t *testing.T) {
	t.Parallel()
	got, source, err := ResolveEffort("", "", "provider-specific")
	if err != nil || got != "provider-specific" || source != EffortSourceClient {
		t.Fatalf("ResolveEffort() = %q, %q, %v; want original client value and no error", got, source, err)
	}
}
