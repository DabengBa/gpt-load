package agent

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"

	"gpt-load/internal/requestlog"
)

var canonicalLowercaseUUIDv4 = regexp.MustCompile(
	`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`,
)

const maxSafeInteger = int64(9_007_199_254_740_991)

var errUnsafeCanonicalUint = errors.New(
	"agent unsigned integer is outside the JSON safe integer range",
)

func parseCanonicalSafeMilliseconds(value string) (int64, error) {
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil || strconv.FormatInt(parsed, 10) != value {
		return 0, fmt.Errorf("agent timestamp must be canonical base-10")
	}
	if err := validateSafeMilliseconds(parsed); err != nil {
		return 0, err
	}
	return parsed, nil
}

func validateSafeMilliseconds(value int64) error {
	if value < 0 || value > maxSafeInteger {
		return fmt.Errorf("agent timestamp is outside the JSON safe integer range")
	}
	return nil
}

func parseCanonicalSafeUint(value string) (uint64, error) {
	parsed, err := strconv.ParseUint(value, 10, 64)
	if err != nil || strconv.FormatUint(parsed, 10) != value {
		return 0, fmt.Errorf("agent unsigned integer must be canonical base-10")
	}
	if parsed > uint64(maxSafeInteger) {
		return 0, errUnsafeCanonicalUint
	}
	return parsed, nil
}

func parseCanonicalSafePlatformUint(value string) (uint, error) {
	parsed, err := parseCanonicalSafeUint(value)
	if err != nil {
		return 0, err
	}
	if parsed > math.MaxUint {
		return 0, errUnsafeCanonicalUint
	}
	return uint(parsed), nil
}

type requestCursorPayload struct {
	Version       int    `json:"v"`
	CompletedAtMS int64  `json:"completed_at_ms"`
	RequestID     string `json:"request_id"`
}

func encodeRequestCursor(cursor requestlog.Cursor) (string, error) {
	if err := validateSafeMilliseconds(cursor.CompletedAtMS); err != nil {
		return "", fmt.Errorf("encode agent cursor: invalid completed_at_ms: %w", err)
	}
	if !canonicalLowercaseUUIDv4.MatchString(cursor.RequestID) {
		return "", fmt.Errorf("encode agent cursor: invalid request ID")
	}
	raw, err := json.Marshal(requestCursorPayload{
		Version:       SchemaVersion,
		CompletedAtMS: cursor.CompletedAtMS,
		RequestID:     cursor.RequestID,
	})
	if err != nil {
		return "", fmt.Errorf("encode agent cursor: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(raw), nil
}

func decodeRequestCursor(encoded string) (*requestlog.Cursor, error) {
	raw, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("decode agent cursor base64: %w", err)
	}
	if encoded != base64.RawURLEncoding.EncodeToString(raw) {
		return nil, fmt.Errorf("decode agent cursor base64: non-canonical encoding")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var payload requestCursorPayload
	if err := decoder.Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode agent cursor JSON: %w", err)
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, fmt.Errorf("decode agent cursor JSON: multiple values")
		}
		return nil, fmt.Errorf("decode agent cursor JSON: %w", err)
	}
	if payload.Version != SchemaVersion {
		return nil, fmt.Errorf("unsupported agent cursor version")
	}
	if err := validateSafeMilliseconds(payload.CompletedAtMS); err != nil {
		return nil, fmt.Errorf("invalid agent cursor completed_at_ms: %w", err)
	}
	if !canonicalLowercaseUUIDv4.MatchString(payload.RequestID) {
		return nil, fmt.Errorf("invalid agent cursor request_id")
	}
	return &requestlog.Cursor{
		CompletedAtMS: payload.CompletedAtMS,
		RequestID:     payload.RequestID,
	}, nil
}

func optionalRequestLogString(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func optionalRequestLogID(value uint) *uint {
	if value == 0 {
		return nil
	}
	return &value
}
