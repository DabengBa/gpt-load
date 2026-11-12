// Package prepare applies group request settings in the same order for each caller.
package prepare

import (
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/parameteroverride"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
)

// Request prepares one client-protocol body, preserving configured field ownership.
func Request(rules parameteroverride.Rules, client protocol.Protocol, operation execution.Operation, model string, body []byte, entryEffort, clientEffort string) ([]byte, []string, bool, bool, error) {
	body, applied, err := rules.Apply(client, operation, model, body)
	if err != nil {
		return nil, nil, false, false, err
	}
	configured := rules.ConfiguredFields(client, operation, model)
	effort, source, err := reasoning.ResolveEffort(entryEffort, clientEffort)
	if err != nil {
		return nil, nil, false, false, err
	}
	overridden := false
	if dialect.SupportsReasoningEffortOverride(client, operation) && effort != "" && source != reasoning.EffortSourceClient {
		var changed bool
		body, changed, err = dialect.SetReasoningEffort(body, effort, client)
		if err != nil {
			return nil, nil, false, false, err
		}
		applied = applied || changed
		overridden = source == reasoning.EffortSourceEntry
	}
	return body, configured, overridden, applied, nil
}
