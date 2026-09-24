package reasoning

import (
	"fmt"
	"strings"
)

// EffortSource identifies the winning source for a resolved reasoning effort.
type EffortSource string

const (
	EffortSourceEntry           EffortSource = "entry"
	EffortSourceClient          EffortSource = "client"
	EffortSourceProviderDefault EffortSource = "provider_default"
)

var validEfforts = map[string]struct{}{
	"none": {}, "minimal": {}, "low": {}, "medium": {}, "high": {}, "xhigh": {}, "max": {},
}

// ResolveEffort applies the entry > client > provider priority. Entry-owned
// policy values are validated instead of silently normalized to a different
// level. Client-owned values remain untouched so an unconfigured center
// preserves the protocol's existing validation and wire semantics.
func ResolveEffort(entry, client string) (string, EffortSource, error) {
	value := strings.ToLower(strings.TrimSpace(entry))
	if value != "" {
		if _, valid := validEfforts[value]; !valid {
			return "", "", fmt.Errorf("invalid reasoning effort from %s source", EffortSourceEntry)
		}
		return value, EffortSourceEntry, nil
	}
	if client != "" {
		return client, EffortSourceClient, nil
	}
	return "", EffortSourceProviderDefault, nil
}

// Config is the explicit reasoning configuration supplied by the client.
// Empty fields mean the client did not provide that part of the configuration.
type Config struct {
	Mode         string `json:"mode,omitempty"`
	Effort       string `json:"effort,omitempty"`
	BudgetTokens *int64 `json:"budget_tokens,omitempty"`
}

// Clone returns an independent reasoning configuration.
func (config Config) Clone() Config {
	clone := config
	if config.BudgetTokens != nil {
		value := *config.BudgetTokens
		clone.BudgetTokens = &value
	}
	return clone
}

func (config Config) Present() bool {
	return config.Mode != "" || config.Effort != "" || config.BudgetTokens != nil
}
