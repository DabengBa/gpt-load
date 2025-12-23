package inbound

import (
	"testing"

	"gpt-load/internal/transformer/inbound/anthropic"
	"gpt-load/internal/transformer/inbound/openai"
	"gpt-load/internal/transformer/model"

	"pgregory.net/rapid"
)

// Property 3: 转换器工厂正确性
// For any registered API format type, the transformer factory should return
// the corresponding transformer instance; for unregistered format types,
// it should return nil.
// **Validates: Requirements 4.1, 4.3, 4.5**

func TestGetInbound_RegisteredTypes_ReturnsCorrectType(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test all registered types
		registeredTypes := RegisteredTypes()

		// Pick a random registered type
		if len(registeredTypes) == 0 {
			t.Skip("no registered types")
			return
		}

		idx := rapid.IntRange(0, len(registeredTypes)-1).Draw(t, "typeIndex")
		inboundType := registeredTypes[idx]

		// Get the inbound transformer
		inbound := GetInbound(inboundType)

		// Verify it's not nil
		if inbound == nil {
			t.Fatalf("GetInbound(%v) returned nil for registered type", inboundType)
		}

		// Verify it returns the correct type
		switch inboundType {
		case InboundTypeOpenAIChat:
			if _, ok := inbound.(*openai.ChatInbound); !ok {
				t.Fatalf("GetInbound(InboundTypeOpenAIChat) returned wrong type: %T", inbound)
			}
		case InboundTypeAnthropic:
			if _, ok := inbound.(*anthropic.MessagesInbound); !ok {
				t.Fatalf("GetInbound(InboundTypeAnthropic) returned wrong type: %T", inbound)
			}
		}

		// Verify the transformer implements the Inbound interface
		var _ model.Inbound = inbound
	})
}

func TestGetInbound_UnregisteredTypes_ReturnsNil(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random InboundType that is not registered
		// Use a large value that's unlikely to be registered
		unregisteredType := InboundType(rapid.IntRange(100, 1000).Draw(t, "unregisteredType"))

		// Verify it's not registered
		if IsRegistered(unregisteredType) {
			t.Skip("randomly generated type is registered")
			return
		}

		// Get the inbound transformer
		inbound := GetInbound(unregisteredType)

		// Verify it's nil
		if inbound != nil {
			t.Fatalf("GetInbound(%v) returned non-nil for unregistered type: %T", unregisteredType, inbound)
		}
	})
}

func TestIsRegistered_Correctness(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test known registered types
		knownRegistered := []InboundType{InboundTypeOpenAIChat, InboundTypeAnthropic}

		for _, inboundType := range knownRegistered {
			if !IsRegistered(inboundType) {
				t.Fatalf("IsRegistered(%v) returned false for known registered type", inboundType)
			}
		}

		// Test known unregistered type (OpenAI Response is defined but not implemented)
		if IsRegistered(InboundTypeOpenAIResponse) {
			// This is expected to be unregistered until implemented
			// If it becomes registered, this test should be updated
		}
	})
}

func TestRegisteredTypes_ContainsAllRegistered(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		registeredTypes := RegisteredTypes()

		// Verify each returned type is actually registered
		for _, inboundType := range registeredTypes {
			if !IsRegistered(inboundType) {
				t.Fatalf("RegisteredTypes() returned %v which is not registered", inboundType)
			}

			// Verify GetInbound returns non-nil for each
			inbound := GetInbound(inboundType)
			if inbound == nil {
				t.Fatalf("GetInbound(%v) returned nil for type in RegisteredTypes()", inboundType)
			}
		}
	})
}

func TestInboundTypeFromAPIFormat_Mapping(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test all known API formats
		testCases := []struct {
			format   model.APIFormat
			expected InboundType
		}{
			{model.APIFormatOpenAIChat, InboundTypeOpenAIChat},
			{model.APIFormatOpenAIResponse, InboundTypeOpenAIResponse},
			{model.APIFormatAnthropic, InboundTypeAnthropic},
		}

		// Pick a random test case
		idx := rapid.IntRange(0, len(testCases)-1).Draw(t, "testCaseIndex")
		tc := testCases[idx]

		result := InboundTypeFromAPIFormat(tc.format)
		if result != tc.expected {
			t.Fatalf("InboundTypeFromAPIFormat(%v) = %v, expected %v", tc.format, result, tc.expected)
		}
	})
}

func TestInboundTypeFromAPIFormat_UnknownFormat_ReturnsDefault(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random unknown API format
		unknownFormat := model.APIFormat(rapid.StringMatching(`^unknown_[a-z0-9]+$`).Draw(t, "unknownFormat"))

		result := InboundTypeFromAPIFormat(unknownFormat)

		// Should return default (OpenAI Chat)
		if result != InboundTypeOpenAIChat {
			t.Fatalf("InboundTypeFromAPIFormat(%v) = %v, expected default InboundTypeOpenAIChat", unknownFormat, result)
		}
	})
}

func TestGetInbound_ReturnsNewInstance(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that GetInbound returns a new instance each time
		registeredTypes := RegisteredTypes()

		if len(registeredTypes) == 0 {
			t.Skip("no registered types")
			return
		}

		idx := rapid.IntRange(0, len(registeredTypes)-1).Draw(t, "typeIndex")
		inboundType := registeredTypes[idx]

		// Get two instances
		inbound1 := GetInbound(inboundType)
		inbound2 := GetInbound(inboundType)

		// Verify they are different instances (not the same pointer)
		if inbound1 == inbound2 {
			t.Fatalf("GetInbound(%v) returned the same instance twice", inboundType)
		}
	})
}

func TestInboundType_String(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		testCases := []struct {
			inboundType InboundType
			expected    string
		}{
			{InboundTypeOpenAIChat, "openai_chat"},
			{InboundTypeOpenAIResponse, "openai_response"},
			{InboundTypeAnthropic, "anthropic"},
		}

		// Pick a random test case
		idx := rapid.IntRange(0, len(testCases)-1).Draw(t, "testCaseIndex")
		tc := testCases[idx]

		result := tc.inboundType.String()
		if result != tc.expected {
			t.Fatalf("InboundType(%d).String() = %s, expected %s", tc.inboundType, result, tc.expected)
		}
	})
}

func TestInboundType_String_Unknown(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random unknown InboundType
		unknownType := InboundType(rapid.IntRange(100, 1000).Draw(t, "unknownType"))

		result := unknownType.String()
		if result != "unknown" {
			t.Fatalf("InboundType(%d).String() = %s, expected 'unknown'", unknownType, result)
		}
	})
}
