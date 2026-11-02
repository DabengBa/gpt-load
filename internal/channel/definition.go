package channel

import (
	"encoding/json"
	"fmt"
	"gpt-load/internal/channel/spec"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"strings"
)

type definition struct {
	responsesWebsocket      execution.WebsocketCapabilities
	descriptor              Descriptor
	params                  objectSchema
	validateParams          func(json.RawMessage) (map[string]string, error)
	credentials             objectSchema
	validateCredential      func(map[string]string) error
	catalogProviderID       string
	providerKind            ProviderKind
	probeContract           spec.ProbeContract
	connection              spec.Connection
	endpointPolicy          spec.EndpointPolicy
	fixedBaseURL            string
	fixedTargetConfig       json.RawMessage
	modes                   map[protocol.Protocol]map[execution.Operation]RouteMode
	resolvers               map[routeKey]spec.RouteResolver
	responsesStoreHandlings map[routeKey]ResponsesStoreHandling
}

func (d definition) matches(query string) bool {
	values := []string{string(d.descriptor.ID), d.descriptor.Name, d.descriptor.Description}
	values = append(values, d.descriptor.SearchTerms...)
	for _, value := range values {
		if strings.Contains(strings.ToLower(value), query) {
			return true
		}
	}
	return false
}

func newRegistry(definitions []definition) (*Registry, error) {
	registry := &Registry{byID: make(map[ID]definition, len(definitions)), order: make([]ID, 0, len(definitions))}
	for _, definition := range definitions {
		id := definition.descriptor.ID
		if err := validateDefinition(definition); err != nil {
			return nil, err
		}
		if _, duplicate := registry.byID[id]; duplicate {
			return nil, fmt.Errorf("duplicate channel ID %q", id)
		}
		definition.descriptor = cloneDescriptor(definition.descriptor)
		definition.modes = cloneRouteModes(definition.modes)
		definition.resolvers = cloneRouteResolvers(definition.resolvers)
		definition.responsesStoreHandlings = cloneResponsesStoreHandlings(
			definition.responsesStoreHandlings,
		)
		definition.fixedTargetConfig = append(json.RawMessage(nil), definition.fixedTargetConfig...)
		registry.byID[id] = definition
		registry.order = append(registry.order, id)
	}
	return registry, nil
}

func validateDefinition(definition definition) error {
	id := string(definition.descriptor.ID)
	if id == "" || strings.Trim(id, "abcdefghijklmnopqrstuvwxyz0123456789_") != "" || strings.HasPrefix(id, "_") || strings.HasSuffix(id, "_") {
		return fmt.Errorf("invalid channel ID %q", id)
	}
	if strings.TrimSpace(definition.descriptor.Name) == "" {
		return fmt.Errorf("channel %q has no name", id)
	}
	if !definition.providerKind.Valid() {
		return fmt.Errorf("channel %q has invalid provider kind", id)
	}
	if definition.connection.CredentialInput == "" {
		return fmt.Errorf("channel %q has no credential input", id)
	}
	if len(definition.fixedTargetConfig) > 0 && !json.Valid(definition.fixedTargetConfig) {
		return fmt.Errorf("channel %q has invalid fixed target config", id)
	}
	for name, schema := range map[string]objectSchema{"params": definition.params, "credential": definition.credentials} {
		seen := make(map[string]struct{}, len(schema))
		for _, field := range schema {
			key := field.descriptor.Key
			if key == "" || field.normalize == nil || (field.descriptor.InputKind != InputText && field.descriptor.InputKind != InputURL && field.descriptor.InputKind != InputSecret) {
				return fmt.Errorf("channel %q has invalid %s field", id, name)
			}
			if _, duplicate := seen[key]; duplicate {
				return fmt.Errorf("channel %q has duplicate %s field %q", id, name, key)
			}
			seen[key] = struct{}{}
		}
	}
	if len(definition.modes) == 0 {
		return fmt.Errorf("channel %q has no client protocols", id)
	}
	for clientProtocol, modes := range definition.modes {
		if !clientProtocol.Valid() {
			return fmt.Errorf("channel %q has invalid client protocol %q", id, clientProtocol)
		}
		if len(modes) == 0 {
			return fmt.Errorf("channel %q has no operations for %q", id, clientProtocol)
		}
		for operation, mode := range modes {
			if !operation.Valid() || !mode.Valid() {
				return fmt.Errorf("channel %q has no valid route mode for %q/%q", id, clientProtocol, operation)
			}
		}
	}
	return nil
}

func (r *Registry) lookup(id ID) (definition, bool) {
	if r == nil {
		return definition{}, false
	}
	definition, ok := r.byID[id]
	return definition, ok
}
