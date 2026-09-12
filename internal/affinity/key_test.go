package affinity

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

type testHasher struct{}

func (testHasher) Hash(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

const testSignalValue = "stable-signal"

func TestDeriveKeyScopesStableSignal(t *testing.T) {
	t.Parallel()

	base := DeriveKey(
		testHasher{}, 7, protocol.OpenAIResponses, "gpt-4o",
		execution.OperationResponsesCreate, SignalPromptPrefix, []byte(testSignalValue),
	)
	if !base.Valid() {
		t.Fatal("DeriveKey() returned an empty key")
	}
	tests := []struct {
		name      string
		accessKey uint
		protocol  protocol.Protocol
		model     string
		operation execution.Operation
		signal    SignalType
		value     string
	}{
		{name: "access key", accessKey: 8, protocol: protocol.OpenAIResponses, model: "gpt-4o", operation: execution.OperationResponsesCreate, signal: SignalPromptPrefix, value: testSignalValue},
		{name: "protocol", accessKey: 7, protocol: protocol.OpenAICompletions, model: "gpt-4o", operation: execution.OperationResponsesCreate, signal: SignalPromptPrefix, value: testSignalValue},
		{name: "client model", accessKey: 7, protocol: protocol.OpenAIResponses, model: "gpt-5", operation: execution.OperationResponsesCreate, signal: SignalPromptPrefix, value: testSignalValue},
		{name: "operation", accessKey: 7, protocol: protocol.OpenAIResponses, model: "gpt-4o", operation: execution.OperationResponsesCompact, signal: SignalPromptPrefix, value: testSignalValue},
		{name: "signal type", accessKey: 7, protocol: protocol.OpenAIResponses, model: "gpt-4o", operation: execution.OperationResponsesCreate, signal: SignalPromptCacheKey, value: testSignalValue},
		{name: "signal value", accessKey: 7, protocol: protocol.OpenAIResponses, model: "gpt-4o", operation: execution.OperationResponsesCreate, signal: SignalPromptPrefix, value: "other-signal"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := DeriveKey(
				testHasher{}, test.accessKey, test.protocol, test.model,
				test.operation, test.signal, []byte(test.value),
			)
			if !got.Valid() || got == base {
				t.Fatalf("DeriveKey() = %q, want non-empty key different from base", got)
			}
		})
	}
	duplicate := DeriveKey(
		testHasher{}, 7, protocol.OpenAIResponses, "gpt-4o",
		execution.OperationResponsesCreate, SignalPromptPrefix, []byte(testSignalValue),
	)
	if duplicate != base {
		t.Fatalf("duplicate DeriveKey() = %q, want %q", duplicate, base)
	}
}

func TestDeriveKeyTreatsMissingClientModelAsEmptyField(t *testing.T) {
	t.Parallel()

	missing := DeriveKey(
		testHasher{}, 7, protocol.OpenAIResponses, "",
		execution.OperationResponsesCreate, SignalPromptCacheKey, []byte("queue"),
	)
	if !missing.Valid() {
		t.Fatal("DeriveKey() with empty client model returned an empty key")
	}
	named := DeriveKey(
		testHasher{}, 7, protocol.OpenAIResponses, "gpt-4o",
		execution.OperationResponsesCreate, SignalPromptCacheKey, []byte("queue"),
	)
	if named == missing {
		t.Fatal("DeriveKey() ignores the client model")
	}
	if repeated := DeriveKey(
		testHasher{}, 7, protocol.OpenAIResponses, "",
		execution.OperationResponsesCreate, SignalPromptCacheKey, []byte("queue"),
	); repeated != missing {
		t.Fatalf("repeated DeriveKey() = %q, want %q", repeated, missing)
	}
}

func TestDeriveKeyIsolatesExplicitSignalFromPrefixSignal(t *testing.T) {
	t.Parallel()

	prefix := DeriveKey(
		testHasher{}, 7, protocol.OpenAIResponses, "gpt-4o",
		execution.OperationResponsesCreate, SignalPromptPrefix, []byte("shared-value"),
	)
	explicit := DeriveKey(
		testHasher{}, 7, protocol.OpenAIResponses, "gpt-4o",
		execution.OperationResponsesCreate, SignalPromptCacheKey, []byte("shared-value"),
	)
	if !prefix.Valid() || !explicit.Valid() || prefix == explicit {
		t.Fatalf("keys = %q / %q, want distinct non-empty namespaces", prefix, explicit)
	}
}

func TestDeriveKeyRejectsIncompleteScope(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		hasher    Hasher
		accessKey uint
		protocol  protocol.Protocol
		signal    SignalType
		value     []byte
	}{
		{name: "nil hasher", accessKey: 1, protocol: protocol.OpenAIResponses, signal: SignalPromptPrefix, value: []byte("prefix")},
		{name: "zero access key", hasher: testHasher{}, protocol: protocol.OpenAIResponses, signal: SignalPromptPrefix, value: []byte("prefix")},
		{name: "invalid protocol", hasher: testHasher{}, accessKey: 1, protocol: protocol.Protocol("invalid"), signal: SignalPromptPrefix, value: []byte("prefix")},
		{name: "unknown signal type", hasher: testHasher{}, accessKey: 1, protocol: protocol.OpenAIResponses, signal: SignalType("invalid"), value: []byte("prefix")},
		{name: "empty signal type", hasher: testHasher{}, accessKey: 1, protocol: protocol.OpenAIResponses, value: []byte("prefix")},
		{name: "empty signal value", hasher: testHasher{}, accessKey: 1, protocol: protocol.OpenAIResponses, signal: SignalPromptPrefix},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := DeriveKey(
				test.hasher, test.accessKey, test.protocol, "", execution.OperationResponsesCreate,
				test.signal, test.value,
			); got.Valid() {
				t.Fatalf("DeriveKey() = %q, want empty key", got)
			}
		})
	}
}

func TestDeriveKeyUsesUnambiguousFieldBoundaries(t *testing.T) {
	t.Parallel()

	left := DeriveKey(
		testHasher{}, 1, protocol.OpenAIResponses, "gpt-4o",
		execution.OperationResponsesCreate, SignalPromptPrefix, []byte("ab"),
	)
	right := DeriveKey(
		testHasher{}, 1, protocol.OpenAIResponses, "gpt-4o",
		execution.OperationResponsesCreate, SignalPromptPrefix, []byte("a"),
	)
	if !left.Valid() || !right.Valid() || left == right {
		t.Fatalf("keys = %q / %q, want distinct non-empty values", left, right)
	}
	shiftedModel := DeriveKey(
		testHasher{}, 1, protocol.OpenAIResponses, "gpt-4oX",
		execution.OperationResponsesCreate, SignalPromptPrefix, []byte("a"),
	)
	if shiftedModel == left {
		t.Fatal("DeriveKey() collides when the client model boundary shifts")
	}
}
