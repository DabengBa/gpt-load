package catalog

import (
	"bytes"
	"encoding/json"
	"strings"
)

func parseModelMetadata(fields map[string]json.RawMessage) (ModelMetadata, error) {
	stringFields := [...]struct {
		name string
		set  func(*ModelMetadata, string)
	}{
		{name: "description", set: func(metadata *ModelMetadata, value string) { metadata.Description = value }},
		{name: "family", set: func(metadata *ModelMetadata, value string) { metadata.Family = value }},
		{name: "knowledge", set: func(metadata *ModelMetadata, value string) { metadata.Knowledge = value }},
		{name: "release_date", set: func(metadata *ModelMetadata, value string) { metadata.ReleaseDate = value }},
		{name: "last_updated", set: func(metadata *ModelMetadata, value string) { metadata.LastUpdated = value }},
		{name: "status", set: func(metadata *ModelMetadata, value string) { metadata.Status = value }},
	}
	var metadata ModelMetadata
	for _, field := range stringFields {
		value, err := decodeStringField(fields, field.name)
		if err != nil {
			return ModelMetadata{}, err
		}
		field.set(&metadata, strings.TrimSpace(value))
	}

	boolFields := [...]struct {
		name string
		set  func(*ModelMetadata, *bool)
	}{
		{name: "attachment", set: func(metadata *ModelMetadata, value *bool) { metadata.Capabilities.Attachment = value }},
		{name: "reasoning", set: func(metadata *ModelMetadata, value *bool) { metadata.Capabilities.Reasoning = value }},
		{name: "tool_call", set: func(metadata *ModelMetadata, value *bool) { metadata.Capabilities.ToolCall = value }},
		{name: "structured_output", set: func(metadata *ModelMetadata, value *bool) { metadata.Capabilities.StructuredOutput = value }},
		{name: "temperature", set: func(metadata *ModelMetadata, value *bool) { metadata.Capabilities.Temperature = value }},
		{name: "open_weights", set: func(metadata *ModelMetadata, value *bool) { metadata.OpenWeights = value }},
	}
	for _, field := range boolFields {
		value, err := decodeBoolField(fields, field.name)
		if err != nil {
			return ModelMetadata{}, err
		}
		field.set(&metadata, value)
	}

	modalities, err := parseModelModalities(fields["modalities"])
	if err != nil {
		return ModelMetadata{}, err
	}
	limits, err := parseModelLimits(fields["limit"])
	if err != nil {
		return ModelMetadata{}, err
	}
	metadata.Modalities = modalities
	metadata.Limits = limits
	return metadata, nil
}

func parseModelModalities(raw json.RawMessage) (ModelModalities, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return ModelModalities{}, nil
	}
	fields, err := decodeCanonicalObject(raw, "model modalities", modalityCanonicalFields)
	if err != nil {
		return ModelModalities{}, err
	}
	input, err := decodeStringSliceField(fields, "input")
	if err != nil {
		return ModelModalities{}, err
	}
	output, err := decodeStringSliceField(fields, "output")
	if err != nil {
		return ModelModalities{}, err
	}
	return ModelModalities{Input: input, Output: output}, nil
}

func parseModelLimits(raw json.RawMessage) (ModelLimits, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return ModelLimits{}, nil
	}
	fields, err := decodeCanonicalObject(raw, "model limits", limitCanonicalFields)
	if err != nil {
		return ModelLimits{}, err
	}
	contextLimit, err := decodeNonNegativeInt64Field(fields, "context")
	if err != nil {
		return ModelLimits{}, err
	}
	inputLimit, err := decodeNonNegativeInt64Field(fields, "input")
	if err != nil {
		return ModelLimits{}, err
	}
	outputLimit, err := decodeNonNegativeInt64Field(fields, "output")
	if err != nil {
		return ModelLimits{}, err
	}
	return ModelLimits{Context: contextLimit, Input: inputLimit, Output: outputLimit}, nil
}
