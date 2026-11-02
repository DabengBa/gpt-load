package channel

import (
	"bytes"
	"encoding/json"
	"fmt"
	"gpt-load/internal/channel/spec"
	"strings"
)

// Registry is an immutable code-owned channel directory.
type Registry struct {
	byID  map[ID]definition
	order []ID
}

// CompileRegistry validates every built-in module and constructs the immutable
// runtime registry. Production startup propagates any invalid module as an
// initialization error instead of publishing a partial registry.
func CompileRegistry() (*Registry, error) {
	definitions, err := compileBuiltInModules(builtInModules())
	if err != nil {
		return nil, fmt.Errorf("compile channel modules: %w", err)
	}
	registry, err := newRegistry(definitions)
	if err != nil {
		return nil, fmt.Errorf("build channel registry: %w", err)
	}
	return registry, nil
}

// NewRegistry constructs the built-in, read-only channel registry for compact
// tests and helpers whose signature cannot return an error.
func NewRegistry() *Registry {
	registry, err := CompileRegistry()
	if err != nil {
		panic(err)
	}
	return registry
}

// Get returns an independent public descriptor.
func (r *Registry) Get(id ID) (Descriptor, bool) {
	definition, ok := r.lookup(id)
	if !ok {
		return Descriptor{}, false
	}
	return cloneDescriptor(definition.descriptor), true
}

// List returns every built-in preset in stable product order.
func (r *Registry) List() []Descriptor {
	if r == nil {
		return nil
	}
	result := make([]Descriptor, 0, len(r.order))
	for _, id := range r.order {
		result = append(result, cloneDescriptor(r.byID[id].descriptor))
	}
	return result
}

// Search returns case-insensitive ID/name/keyword matches in stable product order.
func (r *Registry) Search(query string) []Descriptor {
	if r == nil {
		return nil
	}
	normalized := strings.ToLower(strings.TrimSpace(query))
	result := make([]Descriptor, 0, len(r.order))
	for _, id := range r.order {
		definition := r.byID[id]
		if normalized != "" && !definition.matches(normalized) {
			continue
		}
		result = append(result, cloneDescriptor(definition.descriptor))
	}
	return result
}

// CatalogProviderID returns the exact Models.dev provider mapping for a known
// channel. A known compatible channel intentionally returns an empty mapping.
func (r *Registry) CatalogProviderID(id ID) (string, bool) {
	definition, ok := r.lookup(id)
	if !ok {
		return "", false
	}
	return definition.catalogProviderID, true
}

// ProviderKind returns the runtime provider family for a known channel.
// It is internal metadata and is not part of the serialized descriptor.
func (r *Registry) ProviderKind(id ID) (ProviderKind, bool) {
	definition, ok := r.lookup(id)
	if !ok {
		return "", false
	}
	return definition.providerKind, true
}

// SupportsOutboundProxy reports whether GPT-Load may inject a managed proxy
// into the channel's provider runtime.
func (r *Registry) SupportsOutboundProxy(id ID) bool {
	definition, ok := r.lookup(id)
	return ok && definition.providerKind.SupportsOutboundProxy()
}

// FixedBaseURL returns the code-owned endpoint for a fixed preset.
func (r *Registry) FixedBaseURL(id ID) (string, bool) {
	definition, ok := r.lookup(id)
	if !ok || definition.endpointPolicy != spec.EndpointFixedWithOverride || definition.fixedBaseURL == "" {
		return "", false
	}
	return definition.fixedBaseURL, true
}

// ValidateParams validates and normalizes the channel parameter object.
func (r *Registry) ValidateParams(id ID, raw json.RawMessage) (Params, error) {
	definition, ok := r.lookup(id)
	if !ok {
		return Params{}, &ValidationError{Field: "channel_id", Reason: "unknown channel"}
	}
	var values map[string]string
	var err error
	if definition.validateParams != nil {
		values, err = definition.validateParams(raw)
	} else {
		values, err = definition.params.validate("params", raw)
	}
	if err != nil {
		return Params{}, err
	}
	return Params{values: values}, nil
}

// ValidateCredential validates and normalizes one credential object.
func (r *Registry) ValidateCredential(id ID, raw json.RawMessage) (Credential, error) {
	definition, ok := r.lookup(id)
	if !ok {
		return Credential{}, &ValidationError{Field: "channel_id", Reason: "unknown channel"}
	}
	values, err := definition.credentials.validate("credential", raw)
	if err != nil {
		return Credential{}, err
	}
	if definition.validateCredential != nil {
		if err := definition.validateCredential(values); err != nil {
			return Credential{}, err
		}
	}
	return Credential{values: values}, nil
}

// Resolve validates params and produces a provider-neutral execution target.
func (r *Registry) Resolve(id ID, raw json.RawMessage) (ResolvedTarget, error) {
	definition, ok := r.lookup(id)
	if !ok {
		return ResolvedTarget{}, &ValidationError{Field: "channel_id", Reason: "unknown channel"}
	}
	params, err := r.ValidateParams(id, raw)
	if err != nil {
		return ResolvedTarget{}, err
	}
	targetConfig := resolvedTargetConfig(definition, params)
	return ResolvedTarget{
		ResponsesWebsocket: definition.responsesWebsocket,

		ChannelID:         id,
		ProviderKind:      definition.providerKind,
		TargetConfig:      append(json.RawMessage(nil), targetConfig...),
		CatalogProviderID: definition.catalogProviderID,
		probeContract:     definition.probeContract,
		modes:             cloneRouteModes(definition.modes),
		resolvers:         cloneRouteResolvers(definition.resolvers),
		responsesStoreHandlings: cloneResponsesStoreHandlings(
			definition.responsesStoreHandlings,
		),
	}, nil
}

// ResolveExecutionTarget validates a frozen runtime target produced by Resolve.
// Fixed presets must match their code-owned target exactly; configurable
// channels re-validate the target through their parameter schema.
func (r *Registry) ResolveExecutionTarget(id ID, raw json.RawMessage) (ResolvedTarget, error) {
	definition, ok := r.lookup(id)
	if !ok {
		return ResolvedTarget{}, &ValidationError{Field: "channel_id", Reason: "unknown channel"}
	}
	if len(definition.fixedTargetConfig) > 0 && bytes.Equal(bytes.TrimSpace(raw), definition.fixedTargetConfig) {
		return r.Resolve(id, nil)
	}
	if len(definition.fixedTargetConfig) > 0 {
		params, err := r.ValidateParams(id, raw)
		if err != nil {
			return ResolvedTarget{}, err
		}
		if _, overridden := params.Value("base_url"); !overridden {
			return ResolvedTarget{}, &ValidationError{Field: "target_config", Reason: "does not match channel preset"}
		}
		return r.Resolve(id, params.CanonicalJSON())
	}
	return r.Resolve(id, raw)
}

func resolvedTargetConfig(definition definition, params Params) json.RawMessage {
	if len(definition.fixedTargetConfig) > 0 {
		if baseURL, overridden := params.Value("base_url"); overridden {
			return canonicalJSON(map[string]string{"base_url": baseURL})
		}
		return append(json.RawMessage(nil), definition.fixedTargetConfig...)
	}
	if len(definition.params) > 0 {
		return params.CanonicalJSON()
	}
	return json.RawMessage(`{}`)
}
