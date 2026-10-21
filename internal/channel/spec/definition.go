// Package spec defines the code-owned contract shared by channel modules and
// the channel compiler.
package spec

import (
	"encoding/json"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

// ID is a stable product channel identifier.
type ID string

const (
	OpenAI           ID = "openai"
	Anthropic        ID = "anthropic"
	Gemini           ID = "gemini"
	AzureOpenAI      ID = "azure_openai"
	AWSBedrock       ID = "aws_bedrock"
	GoogleVertex     ID = "google_vertex"
	GPTLoad          ID = "gpt_load"
	NewAPI           ID = "newapi"
	CLIProxyAPI      ID = "cliproxyapi"
	Sub2API          ID = "sub2api"
	OpenAICompatible ID = "openai_compatible"
	DeepSeek         ID = "deepseek"
	MoonshotAI       ID = "moonshotai"
	SiliconFlow      ID = "siliconflow"
	ZhipuAI          ID = "zhipuai"
	Alibaba          ID = "alibaba"
	Volcengine       ID = "volcengine"
	OpenRouter       ID = "openrouter"
	Groq             ID = "groq"
	XAI              ID = "xai"
	Cerebras         ID = "cerebras"
	Mistral          ID = "mistral"
	Nebius           ID = "nebius"
	Parasail         ID = "parasail"
	Wafer            ID = "wafer"
	HuggingFace      ID = "huggingface"
	Cohere           ID = "cohere"
	OpenCodeGo       ID = "opencode_go"
	OpenCodeZen      ID = "opencode_zen"
)

// ProviderKind is the single dispatch key used to resolve a ProviderAdapter.
type ProviderKind string

const (
	ProviderOpenAI               ProviderKind = "openai"
	ProviderAnthropic            ProviderKind = "anthropic"
	ProviderGemini               ProviderKind = "gemini"
	ProviderMultiProtocolGateway ProviderKind = "multi_protocol_gateway"
	ProviderOpenAICompatible     ProviderKind = "openai_compatible"
	ProviderAzureOpenAI          ProviderKind = "azure_openai"
	ProviderAWSBedrock           ProviderKind = "aws_bedrock"
	ProviderGoogleVertex         ProviderKind = "google_vertex"
	ProviderDeepSeek             ProviderKind = "deepseek"
	ProviderOpenRouter           ProviderKind = "openrouter"
	ProviderGroq                 ProviderKind = "groq"
	ProviderXAI                  ProviderKind = "xai"
)

// Valid reports whether the provider adapter key is recognized.
func (kind ProviderKind) Valid() bool {
	switch kind {
	case ProviderOpenAI,
		ProviderAnthropic,
		ProviderGemini,
		ProviderMultiProtocolGateway,
		ProviderOpenAICompatible,
		ProviderAzureOpenAI,
		ProviderAWSBedrock,
		ProviderGoogleVertex,
		ProviderDeepSeek,
		ProviderOpenRouter,
		ProviderGroq,
		ProviderXAI:
		return true
	default:
		return false
	}
}

// SupportsOutboundProxy reports whether GPT-Load may inject its managed proxy
// configuration into this provider. Provider SDK-native environment handling
// remains outside this capability contract.
func (kind ProviderKind) SupportsOutboundProxy() bool {
	if !kind.Valid() {
		return false
	}
	switch kind {
	case ProviderAzureOpenAI, ProviderAWSBedrock, ProviderGoogleVertex:
		return false
	default:
		return true
	}
}

// InputKind describes how the management UI collects one field.
type InputKind string

const (
	InputText   InputKind = "text"
	InputURL    InputKind = "url"
	InputSecret InputKind = "secret"
)

// ValueNormalizer canonicalizes one field without retaining its input.
type ValueNormalizer func(string) (string, error)

// Field is one code-owned parameter or credential field.
type Field struct {
	Key        string
	Label      string
	InputKind  InputKind
	Required   bool
	Sensitive  bool
	Default    string
	Normalizer ValueNormalizer
}

// ConnectionType describes credential acquisition and lifecycle only. It is
// never an execution dispatch key.
type ConnectionType string

const (
	ConnectionAPIKey ConnectionType = "api_key"
)

// Connection is the single connection contract for a channel.
type Connection struct {
	Type            ConnectionType
	CredentialInput string
}

// EndpointPolicy declares where non-secret target configuration comes from.
type EndpointPolicy string

const (
	EndpointSDKDefault        EndpointPolicy = "sdk_default"
	EndpointFixedWithOverride EndpointPolicy = "fixed_with_override"
	EndpointRequiredBaseURL   EndpointPolicy = "required_base_url"
	EndpointCloudParams       EndpointPolicy = "cloud_params"
	EndpointNone              EndpointPolicy = "none"
)

// ProbeContract is the single, code-owned probe contract for one provider.
// Every API-key channel must declare exactly one probe protocol and its
// minimum output token budget.
type ProbeContract struct {
	// Protocol is the single client protocol used for probe requests.
	// It must be a generative data-plane protocol supported by the channel
	// definition (Chat Completions, Responses, Anthropic, or Gemini).
	// Embeddings, Rerank, and Images are not probeable.
	Protocol protocol.Protocol

	// MinOutputTokens is the minimum output token budget for the probe
	// request. The actual value sent respects provider-specific minimums
	// (e.g., 3 for OpenAI Chat, 16 for native OpenAI Responses).
	MinOutputTokens int
}

// Valid reports whether the probe contract is well-formed for an API-key channel.
func (c ProbeContract) Valid() bool {
	if c.Protocol == protocol.OpenAIEmbeddings || c.Protocol == protocol.GeminiEmbeddings || c.Protocol == protocol.Rerank {
		return c.MinOutputTokens == 0
	}
	return c.Protocol.SupportsGeneratedText() && c.MinOutputTokens > 0
}

// ProviderBinding identifies the adapter and its non-secret target policy.
type ProviderBinding struct {
	ProviderKind      ProviderKind
	CatalogProviderID string
	EndpointPolicy    EndpointPolicy
	FixedBaseURL      string
	ProbeContract     ProbeContract
}

// ExtensionID is a strongly typed code-owned extension binding.
type ExtensionID string

// RouteResolverID selects a model-aware route resolver.
type RouteResolverID ExtensionID

// ParamsNormalizerID selects an object-level parameter normalizer.
type ParamsNormalizerID ExtensionID

// CredentialValidatorID selects an object-level credential validator.
type CredentialValidatorID ExtensionID

// ResponsesStoreHandling declares how one Responses Create route handles the
// client's store intent without implying complete Responses lifecycle support.
type ResponsesStoreHandling string

const (
	ResponsesStoreHandlingNone            ResponsesStoreHandling = ""
	ResponsesStoreHandlingUpstreamManaged ResponsesStoreHandling = "upstream_managed"
	ResponsesStoreHandlingStateless       ResponsesStoreHandling = "stateless"
)

// Valid reports whether the Responses store handling value is recognized.
func (handling ResponsesStoreHandling) Valid() bool {
	return handling == ResponsesStoreHandlingNone ||
		handling == ResponsesStoreHandlingUpstreamManaged ||
		handling == ResponsesStoreHandlingStateless
}

// Route is one explicit channel execution capability.
type Route struct {
	ClientProtocol         protocol.Protocol
	Operation              execution.Operation
	Mode                   execution.RouteMode
	ResponsesStoreHandling ResponsesStoreHandling
	Resolver               RouteResolverID
	PossibleModes          []execution.RouteMode
}

// NewRoute constructs one explicit static route declaration.
func NewRoute(
	clientProtocol protocol.Protocol,
	operation execution.Operation,
	mode execution.RouteMode,
) Route {
	return Route{ClientProtocol: clientProtocol, Operation: operation, Mode: mode}
}

// NewResponsesCreateRoute constructs one Responses Create route with an
// explicit store-handling contract.
func NewResponsesCreateRoute(
	mode execution.RouteMode,
	handling ResponsesStoreHandling,
) Route {
	return Route{
		ClientProtocol:         protocol.OpenAIResponses,
		Operation:              execution.OperationResponsesCreate,
		Mode:                   mode,
		ResponsesStoreHandling: handling,
	}
}

// Definition is the complete, code-owned declaration for one channel. Every
// built-in module declares its schema, provider, routes, and bindings inline;
// shared helpers may normalize a field or construct one Route, but never hide
// a complete channel definition, schema, or route set.
type Definition struct {
	ResponsesWebsocket  execution.WebsocketCapabilities
	ID                  ID
	Name                string
	Mark                string
	Icon                string
	SearchTerms         []string
	Description         string
	Connection          Connection
	Params              []Field
	ParamsNormalizer    ParamsNormalizerID
	Credentials         []Field
	CredentialValidator CredentialValidatorID
	Provider            ProviderBinding
	Routes              []Route
}

// ParamsNormalizer applies object-level defaults or cross-field rules.
type ParamsNormalizer func(json.RawMessage) (map[string]string, error)

// CredentialValidator validates a canonical credential object.
type CredentialValidator func(map[string]string) error

// RouteResolver applies one declared model-dependent mode policy.
type RouteResolver func(upstreamModel string, defaultMode execution.RouteMode) execution.RouteMode

// Extensions are implementations owned by the same file as their channel.
type Extensions struct {
	ParamsNormalizers    map[ParamsNormalizerID]ParamsNormalizer
	CredentialValidators map[CredentialValidatorID]CredentialValidator
	RouteResolvers       map[RouteResolverID]RouteResolver
}

// Module keeps a channel declaration and its optional custom code together.
type Module struct {
	Definition Definition
	Extensions Extensions
}
