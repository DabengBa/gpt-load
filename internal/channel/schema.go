package channel

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ValidationError describes one invalid field without embedding its value.
type ValidationError struct {
	Field  string
	Reason string
}

func (e *ValidationError) Error() string { return fmt.Sprintf("%s: %s", e.Field, e.Reason) }

type normalizer func(string) (string, error)

type fieldSpec struct {
	descriptor   FieldDescriptor
	defaultValue string
	normalize    normalizer
}

type objectSchema []fieldSpec

func (s objectSchema) descriptors() []FieldDescriptor {
	descriptors := make([]FieldDescriptor, len(s))
	for index, field := range s {
		descriptors[index] = field.descriptor
	}
	return descriptors
}

func (s objectSchema) validate(prefix string, raw json.RawMessage) (map[string]string, error) {
	object, err := decodeStrictObject(raw)
	if err != nil {
		return nil, &ValidationError{Field: prefix, Reason: err.Error()}
	}
	byKey := make(map[string]fieldSpec, len(s))
	for _, field := range s {
		byKey[field.descriptor.Key] = field
	}
	for key := range object {
		if _, ok := byKey[key]; !ok {
			return nil, &ValidationError{Field: prefix + "." + key, Reason: "unknown field"}
		}
	}
	values := make(map[string]string, len(object))
	for _, field := range s {
		rawValue, exists := object[field.descriptor.Key]
		if !exists {
			if field.defaultValue != "" {
				values[field.descriptor.Key] = field.defaultValue
				continue
			}
			if field.descriptor.Required {
				return nil, &ValidationError{Field: prefix + "." + field.descriptor.Key, Reason: "required"}
			}
			continue
		}
		var value string
		if err := json.Unmarshal(rawValue, &value); err != nil {
			return nil, &ValidationError{Field: prefix + "." + field.descriptor.Key, Reason: "must be a string"}
		}
		if strings.TrimSpace(value) == "" && field.defaultValue != "" {
			values[field.descriptor.Key] = field.defaultValue
			continue
		}
		normalized, err := field.normalize(value)
		if err != nil {
			return nil, &ValidationError{Field: prefix + "." + field.descriptor.Key, Reason: err.Error()}
		}
		values[field.descriptor.Key] = normalized
	}
	return values, nil
}
