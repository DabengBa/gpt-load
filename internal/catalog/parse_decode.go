package catalog

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

func decodeUniqueObject(decoder *json.Decoder, kind string) (map[string]json.RawMessage, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, fmt.Errorf("decode %s object: %w", kind, err)
	}
	delimiter, ok := token.(json.Delim)
	if !ok || delimiter != '{' {
		return nil, fmt.Errorf("%s catalog value must be an object", kind)
	}
	values := make(map[string]json.RawMessage)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, fmt.Errorf("decode %s key: %w", kind, err)
		}
		key, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("%s object key must be a string", kind)
		}
		if _, exists := values[key]; exists {
			return nil, fmt.Errorf("duplicate %s key %q", kind, key)
		}
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return nil, fmt.Errorf("decode %s %q: %w", kind, key, err)
		}
		values[key] = append(json.RawMessage(nil), raw...)
	}
	if _, err := decoder.Token(); err != nil {
		return nil, fmt.Errorf("close %s object: %w", kind, err)
	}
	return values, nil
}

func decodeCanonicalObject(raw []byte, kind string, canonicalFields []string) (map[string]json.RawMessage, error) {
	if err := rejectDuplicateObjectFields(raw); err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	fields, err := decodeUniqueObject(decoder, kind)
	if err != nil {
		return nil, err
	}
	if err := requireJSONEOF(decoder); err != nil {
		return nil, err
	}
	for field := range fields {
		for _, canonical := range canonicalFields {
			if field != canonical && strings.EqualFold(field, canonical) {
				return nil, fmt.Errorf("%s field %q is a non-canonical alias of %q", kind, field, canonical)
			}
		}
	}
	return fields, nil
}

func decodeStringField(fields map[string]json.RawMessage, name string) (string, error) {
	raw, exists := fields[name]
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", nil
	}
	var value string
	if err := decodeOneJSON(raw, &value); err != nil {
		return "", fmt.Errorf("decode %s field: %w", name, err)
	}
	return value, nil
}

func decodeBoolField(fields map[string]json.RawMessage, name string) (*bool, error) {
	raw, exists := fields[name]
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var value bool
	if err := decodeOneJSON(raw, &value); err != nil {
		return nil, fmt.Errorf("decode %s field: %w", name, err)
	}
	return &value, nil
}

func decodeStringSliceField(fields map[string]json.RawMessage, name string) ([]string, error) {
	raw, exists := fields[name]
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var values []string
	if err := decodeOneJSON(raw, &values); err != nil {
		return nil, fmt.Errorf("decode %s field: %w", name, err)
	}
	result := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		normalized := strings.TrimSpace(value)
		if normalized == "" {
			return nil, fmt.Errorf("%s modality must not be empty", name)
		}
		if _, duplicate := seen[normalized]; duplicate {
			return nil, fmt.Errorf("duplicate %s modality %q", name, normalized)
		}
		seen[normalized] = struct{}{}
		result = append(result, normalized)
	}
	return result, nil
}

func decodeNonNegativeInt64Field(fields map[string]json.RawMessage, name string) (*int64, error) {
	value, err := decodeNumberField(fields, name)
	if err != nil || value == nil {
		return nil, err
	}
	parsed, err := strconv.ParseInt(value.String(), 10, 64)
	if err != nil || parsed < 0 {
		return nil, fmt.Errorf("%s limit must be a non-negative integer", name)
	}
	return &parsed, nil
}

func decodeNumberField(fields map[string]json.RawMessage, name string) (*json.Number, error) {
	raw, exists := fields[name]
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var decoded any
	if err := decodeOneJSON(raw, &decoded); err != nil {
		return nil, fmt.Errorf("decode %s field: %w", name, err)
	}
	value, ok := decoded.(json.Number)
	if !ok {
		return nil, fmt.Errorf("decode %s field: JSON number is required", name)
	}
	return &value, nil
}

func decodeOneJSON(raw []byte, destination any) error {
	if err := rejectDuplicateObjectFields(raw); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	return requireJSONEOF(decoder)
}

func rejectDuplicateObjectFields(raw []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := scanUniqueJSONValue(decoder); err != nil {
		return err
	}
	return requireJSONEOF(decoder)
}

func scanUniqueJSONValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, isDelimiter := token.(json.Delim)
	if !isDelimiter {
		return nil
	}
	switch delimiter {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return fmt.Errorf("JSON object key must be a string")
			}
			if _, exists := seen[key]; exists {
				return fmt.Errorf("duplicate JSON object key %q", key)
			}
			seen[key] = struct{}{}
			if err := scanUniqueJSONValue(decoder); err != nil {
				return err
			}
		}
		_, err = decoder.Token()
		return err
	case '[':
		for decoder.More() {
			if err := scanUniqueJSONValue(decoder); err != nil {
				return err
			}
		}
		_, err = decoder.Token()
		return err
	default:
		return fmt.Errorf("unexpected JSON delimiter %q", delimiter)
	}
}

func requireJSONEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("trailing JSON value is not allowed")
		}
		return fmt.Errorf("decode trailing JSON: %w", err)
	}
	return nil
}
