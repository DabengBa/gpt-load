package channel

import (
	"encoding/json"
)

// Params is one validated, canonical channel parameter object.
type Params struct {
	values map[string]string
}

// Value returns one normalized parameter without exposing the backing map.
func (p Params) Value(key string) (string, bool) {
	value, ok := p.values[key]
	return value, ok
}

// CanonicalJSON returns an independent canonical JSON object.
func (p Params) CanonicalJSON() json.RawMessage { return canonicalJSON(p.values) }

// Credential is one validated credential object. Its fields are never marshaled implicitly.
type Credential struct {
	values map[string]string
}

// Value returns one normalized credential field without exposing the backing map.
func (c Credential) Value(key string) (string, bool) {
	value, ok := c.values[key]
	return value, ok
}

// CanonicalJSON explicitly returns the credential object for encryption or execution.
func (c Credential) CanonicalJSON() json.RawMessage { return canonicalJSON(c.values) }

// MarshalJSON prevents accidental serialization of credential names or values.
func (c Credential) MarshalJSON() ([]byte, error) { return []byte(`{}`), nil }
