// Package channel defines the code-owned directory of supported upstream channels.
package channel

import (
	"gpt-load/internal/channel/spec"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

// ID is a stable product channel identifier.
type ID = spec.ID

const (
	OpenAI           = spec.OpenAI
	Anthropic        = spec.Anthropic
	Gemini           = spec.Gemini
	AzureOpenAI      = spec.AzureOpenAI
	AWSBedrock       = spec.AWSBedrock
	GoogleVertex     = spec.GoogleVertex
	GPTLoad          = spec.GPTLoad
	NewAPI           = spec.NewAPI
	CLIProxyAPI      = spec.CLIProxyAPI
	Sub2API          = spec.Sub2API
	OpenAICompatible = spec.OpenAICompatible
	DeepSeek         = spec.DeepSeek
	MoonshotAI       = spec.MoonshotAI
	SiliconFlow      = spec.SiliconFlow
	ZhipuAI          = spec.ZhipuAI
	Alibaba          = spec.Alibaba
	Volcengine       = spec.Volcengine
	OpenRouter       = spec.OpenRouter
	Groq             = spec.Groq
	XAI              = spec.XAI
	Cerebras         = spec.Cerebras
	Mistral          = spec.Mistral
	Nebius           = spec.Nebius
	Parasail         = spec.Parasail
	Wafer            = spec.Wafer
	HuggingFace      = spec.HuggingFace
	Cohere           = spec.Cohere
	OpenCodeGo       = spec.OpenCodeGo
	OpenCodeZen      = spec.OpenCodeZen
)

// InputKind describes how a field is collected without exposing its value.
type InputKind = spec.InputKind

const (
	InputText   = spec.InputText
	InputURL    = spec.InputURL
	InputSecret = spec.InputSecret
)

// FieldDescriptor is the public, value-free schema for one channel field.
type FieldDescriptor struct {
	Key          string    `json:"key"`
	Label        string    `json:"label"`
	InputKind    InputKind `json:"input_kind"`
	Required     bool      `json:"required"`
	Sensitive    bool      `json:"sensitive"`
	DefaultValue *string   `json:"default_value"`
}

// ConnectionDescriptor describes the single code-owned credential flow used
// by a channel.
type ConnectionDescriptor struct {
	CredentialInput string `json:"credential_input"`
}

// CapabilityDescriptor is the safe public projection of optional channel behavior.
type CapabilityDescriptor struct {
	ModelDiscovery bool `json:"model_discovery"`
	OutboundProxy  bool `json:"outbound_proxy"`
}

// RouteDescriptor is the safe, compiled route projection exposed to the UI.
type RouteDescriptor struct {
	ClientProtocol protocol.Protocol   `json:"client_protocol"`
	Operation      execution.Operation `json:"operation"`
	RouteMode      RouteMode           `json:"route_mode"`
	ModelDependent bool                `json:"model_dependent"`
	PossibleModes  []RouteMode         `json:"possible_modes,omitempty"`
}

// Descriptor contains only information safe to expose to the management UI.
type Descriptor struct {
	ID   ID     `json:"channel_id"`
	Name string `json:"name"`
	Mark string `json:"mark"`
	// Icon names a frontend-owned icon resource. The compiler defaults an empty
	// module value to the channel ID; the frontend falls back to Mark when the
	// resolved resource is not bundled.
	Icon             string               `json:"icon"`
	SearchTerms      []string             `json:"search_terms"`
	Description      string               `json:"description"`
	ParamFields      []FieldDescriptor    `json:"param_fields"`
	CredentialFields []FieldDescriptor    `json:"credential_fields"`
	Connection       ConnectionDescriptor `json:"connection"`
	Capabilities     CapabilityDescriptor `json:"capabilities"`
	Routes           []RouteDescriptor    `json:"routes"`
	ClientProtocols  []protocol.Protocol  `json:"client_protocols"`
}
