package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"gpt-load/internal/platform/redact"
)

type credentialLiteralReplacers struct {
	redact   *strings.Replacer
	residual *strings.Replacer
}

func newCredentialLiteralReplacers(secrets []string) (credentialLiteralReplacers, bool) {
	sorted := append([]string(nil), secrets...)
	sort.SliceStable(sorted, func(left, right int) bool {
		return len(sorted[left]) > len(sorted[right])
	})

	redactionPairs := make([]string, 0, len(sorted)*2)
	residualPairs := make([]string, 0, len(sorted)*2)
	seen := make(map[string]struct{}, len(sorted))
	for _, secret := range sorted {
		if secret == "" {
			continue
		}
		if _, exists := seen[secret]; exists {
			continue
		}
		seen[secret] = struct{}{}
		redactionPairs = append(redactionPairs, secret, redact.Placeholder)
		residualPairs = append(residualPairs, secret, "")
	}
	if len(redactionPairs) == 0 {
		return credentialLiteralReplacers{}, false
	}
	return credentialLiteralReplacers{
		redact:   strings.NewReplacer(redactionPairs...),
		residual: strings.NewReplacer(residualPairs...),
	}, true
}

func redactCredentialLiterals(
	body []byte,
	secrets []string,
	limit int64,
) ([]byte, bool, bool) {
	replacers, exists := newCredentialLiteralReplacers(secrets)
	if !exists {
		return bytes.Clone(body), false, int64(len(body)) <= limit
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return redactNonJSONCredentialLiterals(body, replacers, limit)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return redactNonJSONCredentialLiterals(body, replacers, limit)
	}

	rewritten, changed, residual, ok := rewriteJSONCredentialLiterals(value, replacers, limit)
	if !ok || residual {
		return nil, residual, ok
	}
	if !changed {
		return bytes.Clone(body), false, true
	}
	encoded, err := json.Marshal(rewritten)
	if err != nil || int64(len(encoded)) > limit {
		return nil, false, false
	}
	return encoded, false, true
}

func redactNonJSONCredentialLiterals(
	body []byte,
	replacers credentialLiteralReplacers,
	limit int64,
) ([]byte, bool, bool) {
	rewritten, ok := replaceCredentialStringBounded(string(body), replacers.redact, limit)
	if !ok {
		return nil, false, false
	}
	if credentialLiteralRemains(rewritten, replacers.residual) {
		return nil, true, true
	}
	return []byte(rewritten), false, true
}

func rewriteJSONCredentialLiterals(
	value any,
	replacers credentialLiteralReplacers,
	limit int64,
) (any, bool, bool, bool) {
	switch typed := value.(type) {
	case string:
		rewritten, ok := replaceCredentialStringBounded(typed, replacers.redact, limit)
		if !ok {
			return nil, false, false, false
		}
		if credentialLiteralRemains(rewritten, replacers.residual) {
			return nil, false, true, true
		}
		return rewritten, rewritten != typed, false, true
	case json.Number:
		if credentialLiteralRemains(typed.String(), replacers.residual) {
			return nil, false, true, true
		}
		return value, false, false, true
	case bool:
		if credentialLiteralRemains(fmt.Sprintf("%t", typed), replacers.residual) {
			return nil, false, true, true
		}
		return value, false, false, true
	case nil:
		if credentialLiteralRemains("null", replacers.residual) {
			return nil, false, true, true
		}
		return nil, false, false, true
	case []any:
		changed := false
		for index, item := range typed {
			rewritten, itemChanged, residual, ok := rewriteJSONCredentialLiterals(
				item,
				replacers,
				limit,
			)
			if !ok || residual {
				return nil, false, residual, ok
			}
			typed[index] = rewritten
			changed = changed || itemChanged
		}
		return typed, changed, false, true
	case map[string]any:
		rewrittenObject := make(map[string]any, len(typed))
		changed := false
		for key, item := range typed {
			rewrittenKey, ok := replaceCredentialStringBounded(key, replacers.redact, limit)
			if !ok {
				return nil, false, false, false
			}
			if credentialLiteralRemains(rewrittenKey, replacers.residual) {
				return nil, false, true, true
			}
			if _, exists := rewrittenObject[rewrittenKey]; exists {
				return nil, false, false, false
			}
			rewrittenItem, itemChanged, residual, ok := rewriteJSONCredentialLiterals(
				item,
				replacers,
				limit,
			)
			if !ok || residual {
				return nil, false, residual, ok
			}
			rewrittenObject[rewrittenKey] = rewrittenItem
			changed = changed || rewrittenKey != key || itemChanged
		}
		return rewrittenObject, changed, false, true
	default:
		return nil, false, false, false
	}
}

func replaceCredentialStringBounded(
	source string,
	replacer *strings.Replacer,
	limit int64,
) (string, bool) {
	writer := &boundedCredentialBuffer{limit: limit}
	if _, err := replacer.WriteString(writer, source); err != nil {
		return "", false
	}
	return writer.buffer.String(), true
}

func credentialLiteralRemains(source string, residual *strings.Replacer) bool {
	return residual.Replace(source) != source
}

func credentialLiteralsRemain(body []byte, secrets []string) bool {
	replacers, exists := newCredentialLiteralReplacers(secrets)
	if !exists {
		return false
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return credentialLiteralRemains(string(body), replacers.residual)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return credentialLiteralRemains(string(body), replacers.residual)
	}
	return jsonCredentialLiteralsRemain(value, replacers.residual)
}
