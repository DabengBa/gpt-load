package channel

import (
	"gpt-load/internal/channel/spec"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func cloneDescriptor(source Descriptor) Descriptor {
	source.SearchTerms = append([]string{}, source.SearchTerms...)
	source.ParamFields = append([]FieldDescriptor{}, source.ParamFields...)
	source.CredentialFields = append([]FieldDescriptor{}, source.CredentialFields...)
	for index := range source.ParamFields {
		source.ParamFields[index].DefaultValue = cloneStringPointer(source.ParamFields[index].DefaultValue)
	}
	for index := range source.CredentialFields {
		source.CredentialFields[index].DefaultValue = cloneStringPointer(source.CredentialFields[index].DefaultValue)
	}
	source.Routes = append([]RouteDescriptor{}, source.Routes...)
	for index := range source.Routes {
		source.Routes[index].PossibleModes = append([]RouteMode{}, source.Routes[index].PossibleModes...)
	}
	source.ClientProtocols = append([]protocol.Protocol{}, source.ClientProtocols...)
	return source
}

func cloneStringPointer(source *string) *string {
	if source == nil {
		return nil
	}
	value := *source
	return &value
}

func cloneRouteModes(source map[protocol.Protocol]map[execution.Operation]RouteMode) map[protocol.Protocol]map[execution.Operation]RouteMode {
	clone := make(map[protocol.Protocol]map[execution.Operation]RouteMode, len(source))
	for clientProtocol, modes := range source {
		clonedModes := make(map[execution.Operation]RouteMode, len(modes))
		for operation, mode := range modes {
			clonedModes[operation] = mode
		}
		clone[clientProtocol] = clonedModes
	}
	return clone
}

func cloneRouteResolvers(source map[routeKey]spec.RouteResolver) map[routeKey]spec.RouteResolver {
	clone := make(map[routeKey]spec.RouteResolver, len(source))
	for key, resolver := range source {
		clone[key] = resolver
	}
	return clone
}

func cloneResponsesStoreHandlings(
	source map[routeKey]ResponsesStoreHandling,
) map[routeKey]ResponsesStoreHandling {
	clone := make(map[routeKey]ResponsesStoreHandling, len(source))
	for key, compatibility := range source {
		clone[key] = compatibility
	}
	return clone
}
