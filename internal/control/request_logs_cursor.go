package control

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"

	"gpt-load/internal/requestlog"
)

func decodeRequestLogCursor(encoded string) (*requestlog.Cursor, error) {
	raw, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("decode request log cursor base64: %w", err)
	}
	if encoded != base64.RawURLEncoding.EncodeToString(raw) {
		return nil, fmt.Errorf("decode request log cursor base64: non-canonical encoding")
	}
	payload, err := decodeRequestLogCursorPayload(raw)
	if err != nil {
		return nil, err
	}
	if payload.Version != requestLogCursorV2 {
		return nil, fmt.Errorf("unsupported request log cursor version")
	}
	if err := validateSafeMilliseconds(payload.CompletedAtMS); err != nil {
		return nil, fmt.Errorf("invalid request log cursor completed_at_ms: %w", err)
	}
	if !canonicalLowercaseUUIDv4.MatchString(payload.RequestID) {
		return nil, fmt.Errorf("invalid request log cursor request_id")
	}
	return &requestlog.Cursor{
		CompletedAtMS: payload.CompletedAtMS,
		RequestID:     payload.RequestID,
	}, nil
}

func decodeRequestLogCursorPayload(raw []byte) (requestLogCursorPayload, error) {
	if err := rejectDuplicateRequestLogCursorFields(raw); err != nil {
		return requestLogCursorPayload{}, err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var payload requestLogCursorPayload
	if err := decoder.Decode(&payload); err != nil {
		return requestLogCursorPayload{}, fmt.Errorf("decode request log cursor JSON: %w", err)
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errorsIsEOF(err) {
		if err == nil {
			return requestLogCursorPayload{}, fmt.Errorf("decode request log cursor JSON: multiple values")
		}
		return requestLogCursorPayload{}, fmt.Errorf("decode request log cursor JSON: %w", err)
	}
	return payload, nil
}

func rejectDuplicateRequestLogCursorFields(raw []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	token, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("decode request log cursor JSON: %w", err)
	}
	if delimiter, ok := token.(json.Delim); !ok || delimiter != '{' {
		return fmt.Errorf("request log cursor must be a JSON object")
	}
	seen := make(map[string]struct{}, 3)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return fmt.Errorf("decode request log cursor field: %w", err)
		}
		key, ok := token.(string)
		if !ok {
			return fmt.Errorf("request log cursor field name is invalid")
		}
		if _, exists := seen[key]; exists {
			return fmt.Errorf("duplicate request log cursor field %q", key)
		}
		seen[key] = struct{}{}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return fmt.Errorf("decode request log cursor field %q: %w", key, err)
		}
	}
	if _, err := decoder.Token(); err != nil {
		return fmt.Errorf("decode request log cursor JSON: %w", err)
	}
	return nil
}

func errorsIsEOF(err error) bool {
	return err == io.EOF
}

func encodeRequestLogCursor(cursor requestlog.Cursor) (string, error) {
	if err := validateSafeMilliseconds(cursor.CompletedAtMS); err != nil {
		return "", fmt.Errorf("encode request log cursor: invalid completed_at_ms: %w", err)
	}
	if !canonicalLowercaseUUIDv4.MatchString(cursor.RequestID) {
		return "", fmt.Errorf("encode request log cursor: invalid request ID")
	}
	payload := requestLogCursorPayload{
		Version:       requestLogCursorV2,
		CompletedAtMS: cursor.CompletedAtMS,
		RequestID:     cursor.RequestID,
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("encode request log cursor JSON: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(raw), nil
}
