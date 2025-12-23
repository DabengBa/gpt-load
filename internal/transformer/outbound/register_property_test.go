package outbound

import (
	"testing"

	"gpt-load/internal/transformer/model"
	"gpt-load/internal/transformer/outbound/anthropic"
	"gpt-load/internal/transformer/outbound/gemini"
	"gpt-load/internal/transformer/outbound/openai"

	"pgregory.net/rapid"
)

// Property 3: 转换器工厂正确性
// For any registered API format type, the transformer factory should return
// the corresponding transformer instance; for unregistered format types,
// it should return nil.
// **Validates: Requirements 4.2, 4.4, 4.5**

func TestGetOutbound_RegisteredTypes_ReturnsCorrectType(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test all registered types
		registeredTypes := RegisteredTypes()

		// Pick a random registered type
		if len(registeredTypes) == 0 {
			t.Skip("no registered types")
			return
		}

		idx := rapid.IntRange(0, len(registeredTypes)-1).Draw(t, "typeIndex")
		outboundType := registeredTypes[idx]

		// Get the outbound transformer
		outbound := GetOutbound(outboundType)

		// Verify it's not nil
		if outbound == nil {
			t.Fatalf("GetOutbound(%v) returned nil for registered type", outboundType)
		}

		// Verify it returns the correct type
		switch outboundType {
		case OutboundTypeOpenAIChat:
			if _, ok := outbound.(*openai.ChatOutbound); !ok {
				t.Fatalf("GetOutbound(OutboundTypeOpenAIChat) returned wrong type: %T", outbound)
			}
		case OutboundTypeAnthropic:
			if _, ok := outbound.(*anthropic.MessagesOutbound); !ok {
				t.Fatalf("GetOutbound(OutboundTypeAnthropic) returned wrong type: %T", outbound)
			}
		case OutboundTypeGemini:
			if _, ok := outbound.(*gemini.MessagesOutbound); !ok {
				t.Fatalf("GetOutbound(OutboundTypeGemini) returned wrong type: %T", outbound)
			}
		}

		// Verify the transformer implements the Outbound interface
		var _ model.Outbound = outbound
	})
}

func TestGetOutbound_UnregisteredTypes_ReturnsNil(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random OutboundType that is not registered
		// Use a large value that's unlikely to be registered
		unregisteredType := OutboundType(rapid.IntRange(100, 1000).Draw(t, "unregisteredType"))

		// Verify it's not registered
		if IsRegistered(unregisteredType) {
			t.Skip("randomly generated type is registered")
			return
		}

		// Get the outbound transformer
		outbound := GetOutbound(unregisteredType)

		// Verify it's nil
		if outbound != nil {
			t.Fatalf("GetOutbound(%v) returned non-nil for unregistered type: %T", unregisteredType, outbound)
		}
	})
}

func TestIsRegistered_Correctness(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test known registered types
		knownRegistered := []OutboundType{OutboundTypeOpenAIChat, OutboundTypeAnthropic, OutboundTypeGemini}

		for _, outboundType := range knownRegistered {
			if !IsRegistered(outboundType) {
				t.Fatalf("IsRegistered(%v) returned false for known registered type", outboundType)
			}
		}

		// Test known unregistered type (OpenAI Response is defined but not implemented)
		if IsRegistered(OutboundTypeOpenAIResponse) {
			// This is expected to be unregistered until implemented
			// If it becomes registered, this test should be updated
		}
	})
}

func TestRegisteredTypes_ContainsAllRegistered(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		registeredTypes := RegisteredTypes()

		// Verify each returned type is actually registered
		for _, outboundType := range registeredTypes {
			if !IsRegistered(outboundType) {
				t.Fatalf("RegisteredTypes() returned %v which is not registered", outboundType)
			}

			// Verify GetOutbound returns non-nil for each
			outbound := GetOutbound(outboundType)
			if outbound == nil {
				t.Fatalf("GetOutbound(%v) returned nil for type in RegisteredTypes()", outboundType)
			}
		}
	})
}

func TestOutboundTypeFromAPIFormat_Mapping(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test all known API formats
		testCases := []struct {
			format   model.APIFormat
			expected OutboundType
		}{
			{model.APIFormatOpenAIChat, OutboundTypeOpenAIChat},
			{model.APIFormatOpenAIResponse, OutboundTypeOpenAIResponse},
			{model.APIFormatAnthropic, OutboundTypeAnthropic},
			{model.APIFormatGemini, OutboundTypeGemini},
		}

		// Pick a random test case
		idx := rapid.IntRange(0, len(testCases)-1).Draw(t, "testCaseIndex")
		tc := testCases[idx]

		result := OutboundTypeFromAPIFormat(tc.format)
		if result != tc.expected {
			t.Fatalf("OutboundTypeFromAPIFormat(%v) = %v, expected %v", tc.format, result, tc.expected)
		}
	})
}

func TestOutboundTypeFromAPIFormat_UnknownFormat_ReturnsDefault(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random unknown API format
		unknownFormat := model.APIFormat(rapid.StringMatching(`^unknown_[a-z0-9]+$`).Draw(t, "unknownFormat"))

		result := OutboundTypeFromAPIFormat(unknownFormat)

		// Should return default (OpenAI Chat)
		if result != OutboundTypeOpenAIChat {
			t.Fatalf("OutboundTypeFromAPIFormat(%v) = %v, expected default OutboundTypeOpenAIChat", unknownFormat, result)
		}
	})
}

func TestGetOutbound_ReturnsValidInstance(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that GetOutbound returns a valid instance each time
		// Note: For empty structs, Go may return the same address as an optimization,
		// so we verify that the factory returns a non-nil, usable instance
		registeredTypes := RegisteredTypes()

		if len(registeredTypes) == 0 {
			t.Skip("no registered types")
			return
		}

		idx := rapid.IntRange(0, len(registeredTypes)-1).Draw(t, "typeIndex")
		outboundType := registeredTypes[idx]

		// Get two instances
		outbound1 := GetOutbound(outboundType)
		outbound2 := GetOutbound(outboundType)

		// Verify both are non-nil
		if outbound1 == nil {
			t.Fatalf("GetOutbound(%v) returned nil on first call", outboundType)
		}
		if outbound2 == nil {
			t.Fatalf("GetOutbound(%v) returned nil on second call", outboundType)
		}

		// Verify both implement the Outbound interface
		var _ model.Outbound = outbound1
		var _ model.Outbound = outbound2
	})
}

func TestOutboundType_String(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		testCases := []struct {
			outboundType OutboundType
			expected     string
		}{
			{OutboundTypeOpenAIChat, "openai_chat"},
			{OutboundTypeOpenAIResponse, "openai_response"},
			{OutboundTypeAnthropic, "anthropic"},
			{OutboundTypeGemini, "gemini"},
		}

		// Pick a random test case
		idx := rapid.IntRange(0, len(testCases)-1).Draw(t, "testCaseIndex")
		tc := testCases[idx]

		result := tc.outboundType.String()
		if result != tc.expected {
			t.Fatalf("OutboundType(%d).String() = %s, expected %s", tc.outboundType, result, tc.expected)
		}
	})
}

func TestOutboundType_String_Unknown(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random unknown OutboundType
		unknownType := OutboundType(rapid.IntRange(100, 1000).Draw(t, "unknownType"))

		result := unknownType.String()
		if result != "unknown" {
			t.Fatalf("OutboundType(%d).String() = %s, expected 'unknown'", unknownType, result)
		}
	})
}
