package channel

import (
	"encoding/json"
	"gpt-load/internal/channel/modules"
	"gpt-load/internal/channel/spec"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"slices"
)

// RouteMode is the neutral execution route decision exposed by a channel.
type RouteMode = execution.RouteMode

const (
	RouteNative    = execution.RouteNative
	RouteConverted = execution.RouteConverted
)

// ResponsesStoreHandling describes how one Responses Create route handles the
// client's store intent.
type ResponsesStoreHandling = spec.ResponsesStoreHandling

const (
	ResponsesStoreHandlingNone            = spec.ResponsesStoreHandlingNone
	ResponsesStoreHandlingUpstreamManaged = spec.ResponsesStoreHandlingUpstreamManaged
	ResponsesStoreHandlingStateless       = spec.ResponsesStoreHandlingStateless
)

// ProviderKind identifies the upstream API family selected by a code-owned
// channel preset. It is runtime metadata, not a persisted SDK identifier.
type ProviderKind = spec.ProviderKind

const (
	ProviderOpenAI               = spec.ProviderOpenAI
	ProviderAnthropic            = spec.ProviderAnthropic
	ProviderGemini               = spec.ProviderGemini
	ProviderMultiProtocolGateway = spec.ProviderMultiProtocolGateway
	ProviderOpenAICompatible     = spec.ProviderOpenAICompatible
	ProviderAzureOpenAI          = spec.ProviderAzureOpenAI
	ProviderAWSBedrock           = spec.ProviderAWSBedrock
	ProviderGoogleVertex         = spec.ProviderGoogleVertex
	ProviderDeepSeek             = spec.ProviderDeepSeek
	ProviderOpenRouter           = spec.ProviderOpenRouter
	ProviderGroq                 = spec.ProviderGroq
	ProviderXAI                  = spec.ProviderXAI
)

// ResolvedTarget is the provider-neutral execution target derived from a channel preset.
type ResolvedTarget struct {
	ResponsesWebsocket execution.WebsocketCapabilities `json:"-"`

	ChannelID         ID              `json:"channel_id"`
	ProviderKind      ProviderKind    `json:"-"`
	TargetConfig      json.RawMessage `json:"-"`
	CatalogProviderID string          `json:"-"`

	probeContract           spec.ProbeContract
	modes                   map[protocol.Protocol]map[execution.Operation]RouteMode
	resolvers               map[routeKey]spec.RouteResolver
	responsesStoreHandlings map[routeKey]ResponsesStoreHandling
}

type routeKey struct {
	clientProtocol protocol.Protocol
	operation      execution.Operation
}

// Mode returns how an operation reaches this channel.
func (t ResolvedTarget) Mode(
	clientProtocol protocol.Protocol,
	operation execution.Operation,
) (RouteMode, bool) {
	byOperation, ok := t.modes[clientProtocol]
	if !ok {
		return "", false
	}
	mode, ok := byOperation[operation]
	return mode, ok
}

// ResponsesStoreHandling returns the explicit store handling for one route.
func (t ResolvedTarget) ResponsesStoreHandling(
	clientProtocol protocol.Protocol,
	operation execution.Operation,
) ResponsesStoreHandling {
	return t.responsesStoreHandlings[routeKey{
		clientProtocol: clientProtocol,
		operation:      operation,
	}]
}

// ModeForModel returns the route mode after applying model-specific channel behavior.
func (t ResolvedTarget) ModeForModel(
	clientProtocol protocol.Protocol,
	operation execution.Operation,
	upstreamModel string,
) (RouteMode, bool) {
	mode, ok := t.Mode(clientProtocol, operation)
	if !ok {
		return "", false
	}
	if resolver := t.resolvers[routeKey{clientProtocol: clientProtocol, operation: operation}]; resolver != nil {
		resolved := resolver(upstreamModel, mode)
		if !resolved.Valid() {
			return "", false
		}
		return resolved, true
	}
	return mode, true
}

// ProbeProtocol returns the single generative protocol declared by the channel.
// It is derived from the code-owned probe contract rather than from protocol
// ordering or provider kind.
func (t ResolvedTarget) ProbeProtocol() (protocol.Protocol, bool) {
	contract, ok := t.ProbeContract()
	if !ok {
		return "", false
	}
	return contract.Protocol, true
}

// ProbeRoute resolves the declared probe protocol and the model-dependent mode
// for one upstream model. Every probe target uses one protocol and one route.
func (t ResolvedTarget) ProbeRoute(upstreamModel string) (protocol.Protocol, RouteMode, bool) {
	contract, ok := t.ProbeContract()
	if !ok {
		return "", "", false
	}
	mode, ok := t.ModeForModel(contract.Protocol, execution.OperationProbe, upstreamModel)
	if !ok {
		return "", "", false
	}
	return contract.Protocol, mode, true
}

// ModelTraceRoute resolves the declared route of one diagnostic challenge.
// Fingerprint challenges only use generative protocols: an Embeddings, Rerank,
// or Images route cannot produce the fingerprint text the challenge asks for.
func (t ResolvedTarget) ModelTraceRoute(upstreamModel string) (protocol.Protocol, RouteMode, bool) {
	if !t.probeContract.Valid() || !t.probeContract.Protocol.SupportsGeneratedText() {
		return "", "", false
	}
	mode, ok := t.ModeForModel(t.probeContract.Protocol, execution.OperationModelTrace, upstreamModel)
	if !ok {
		return "", "", false
	}
	return t.probeContract.Protocol, mode, true
}

// PreferredRoute selects one declared route for a utility operation,
// preferring native mode and then canonical protocol order.
func (t ResolvedTarget) PreferredRoute(
	operation execution.Operation,
	upstreamModel string,
) (protocol.Protocol, RouteMode, bool) {
	for _, clientProtocol := range protocol.DataPlaneProtocols() {
		if mode, ok := t.ModeForModel(clientProtocol, operation, upstreamModel); ok && mode == RouteNative {
			return clientProtocol, mode, true
		}
	}
	for _, clientProtocol := range protocol.DataPlaneProtocols() {
		if mode, ok := t.ModeForModel(clientProtocol, operation, upstreamModel); ok {
			return clientProtocol, mode, true
		}
	}
	return "", "", false
}

// PreferredProtocol selects the protocol of the preferred utility route.
func (t ResolvedTarget) PreferredProtocol(
	operation execution.Operation,
	upstreamModel string,
) (protocol.Protocol, bool) {
	clientProtocol, _, ok := t.PreferredRoute(operation, upstreamModel)
	return clientProtocol, ok
}

// ProbeContract returns the code-owned manual probe contract. API-key channels
// declare exactly one probe protocol.
func (t ResolvedTarget) ProbeContract() (spec.ProbeContract, bool) {
	if !t.probeContract.Valid() {
		return spec.ProbeContract{}, false
	}
	return t.probeContract, true
}

// NormalizeVertexGeminiModel returns the Vertex resource ID for a Gemini,
// Gemma, or numeric custom endpoint that can preserve the native Gemini wire format.
func NormalizeVertexGeminiModel(model string) (string, bool) {
	return modules.NormalizeVertexGeminiModel(model)
}

// Operations returns the stable operation set declared for a client protocol.
func (t ResolvedTarget) Operations(clientProtocol protocol.Protocol) []execution.Operation {
	byOperation := t.modes[clientProtocol]
	operations := make([]execution.Operation, 0, len(byOperation))
	for operation := range byOperation {
		operations = append(operations, operation)
	}
	slices.Sort(operations)
	return operations
}

// SupportsResponsesLifecycle reports whether stored Responses resources can
// be retrieved and managed through this target.
func (t ResolvedTarget) SupportsResponsesLifecycle() bool {
	for _, operation := range []execution.Operation{
		execution.OperationResponsesRetrieve,
		execution.OperationResponsesDelete,
		execution.OperationResponsesCancel,
		execution.OperationResponsesInputItems,
	} {
		mode, ok := t.Mode(protocol.OpenAIResponses, operation)
		if !ok || mode != RouteNative {
			return false
		}
	}
	return true
}
