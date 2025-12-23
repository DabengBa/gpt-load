package models

import (
	"testing"

	"pgregory.net/rapid"
)

// Property 4: API 格式到出站转换器映射
// For any Group configuration, based on its api_format field (or the default mapping from channel_type),
// the system should select the correct outbound transformer type.
// **Validates: Requirements 5.2, 5.3, 5.4, 5.5**

func TestGroup_GetAPIFormat_ExplicitFormat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that when APIFormat is explicitly set, it is returned directly
		validFormats := []string{
			APIFormatOpenAIChat,
			APIFormatOpenAIResponse,
			APIFormatAnthropic,
			APIFormatGemini,
		}

		// Pick a random valid format
		idx := rapid.IntRange(0, len(validFormats)-1).Draw(t, "formatIndex")
		expectedFormat := validFormats[idx]

		// Create a group with explicit APIFormat
		group := &Group{
			APIFormat:   expectedFormat,
			ChannelType: "some_channel", // Should be ignored when APIFormat is set
		}

		result := group.GetAPIFormat()
		if result != expectedFormat {
			t.Fatalf("GetAPIFormat() = %s, expected %s when APIFormat is explicitly set", result, expectedFormat)
		}
	})
}

func TestGroup_GetAPIFormat_OpenAIChannelType(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that when APIFormat is empty and ChannelType is "openai",
		// GetAPIFormat returns "openai_chat"
		group := &Group{
			APIFormat:   "", // Empty - should infer from ChannelType
			ChannelType: "openai",
		}

		result := group.GetAPIFormat()
		if result != APIFormatOpenAIChat {
			t.Fatalf("GetAPIFormat() = %s, expected %s for ChannelType 'openai'", result, APIFormatOpenAIChat)
		}
	})
}

func TestGroup_GetAPIFormat_AnthropicChannelType(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that when APIFormat is empty and ChannelType is "anthropic",
		// GetAPIFormat returns "anthropic"
		group := &Group{
			APIFormat:   "", // Empty - should infer from ChannelType
			ChannelType: "anthropic",
		}

		result := group.GetAPIFormat()
		if result != APIFormatAnthropic {
			t.Fatalf("GetAPIFormat() = %s, expected %s for ChannelType 'anthropic'", result, APIFormatAnthropic)
		}
	})
}

func TestGroup_GetAPIFormat_GeminiChannelType(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that when APIFormat is empty and ChannelType is "gemini",
		// GetAPIFormat returns "gemini"
		group := &Group{
			APIFormat:   "", // Empty - should infer from ChannelType
			ChannelType: "gemini",
		}

		result := group.GetAPIFormat()
		if result != APIFormatGemini {
			t.Fatalf("GetAPIFormat() = %s, expected %s for ChannelType 'gemini'", result, APIFormatGemini)
		}
	})
}

func TestGroup_GetAPIFormat_UnknownChannelType_DefaultsToOpenAI(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that when APIFormat is empty and ChannelType is unknown,
		// GetAPIFormat returns "openai_chat" as default
		unknownChannelTypes := []string{
			"unknown",
			"custom",
			"azure",
			"other",
			"",
		}

		idx := rapid.IntRange(0, len(unknownChannelTypes)-1).Draw(t, "channelTypeIndex")
		channelType := unknownChannelTypes[idx]

		group := &Group{
			APIFormat:   "", // Empty - should infer from ChannelType
			ChannelType: channelType,
		}

		result := group.GetAPIFormat()
		if result != APIFormatOpenAIChat {
			t.Fatalf("GetAPIFormat() = %s, expected %s for unknown ChannelType '%s'", result, APIFormatOpenAIChat, channelType)
		}
	})
}

func TestGroup_GetAPIFormat_ExplicitOverridesChannelType(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that explicit APIFormat always overrides ChannelType inference
		channelTypes := []string{"openai", "anthropic", "gemini", "unknown"}
		apiFormats := []string{
			APIFormatOpenAIChat,
			APIFormatOpenAIResponse,
			APIFormatAnthropic,
			APIFormatGemini,
		}

		// Pick random channel type and API format
		ctIdx := rapid.IntRange(0, len(channelTypes)-1).Draw(t, "channelTypeIndex")
		afIdx := rapid.IntRange(0, len(apiFormats)-1).Draw(t, "apiFormatIndex")

		channelType := channelTypes[ctIdx]
		apiFormat := apiFormats[afIdx]

		group := &Group{
			APIFormat:   apiFormat,
			ChannelType: channelType,
		}

		result := group.GetAPIFormat()
		if result != apiFormat {
			t.Fatalf("GetAPIFormat() = %s, expected %s (explicit APIFormat should override ChannelType '%s')", result, apiFormat, channelType)
		}
	})
}

func TestGroup_GetAPIFormat_BackwardCompatibility(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test backward compatibility: existing groups without APIFormat field
		// should still work correctly based on ChannelType
		testCases := []struct {
			channelType    string
			expectedFormat string
		}{
			{"openai", APIFormatOpenAIChat},
			{"anthropic", APIFormatAnthropic},
			{"gemini", APIFormatGemini},
		}

		idx := rapid.IntRange(0, len(testCases)-1).Draw(t, "testCaseIndex")
		tc := testCases[idx]

		// Simulate an existing group without APIFormat set
		group := &Group{
			ChannelType: tc.channelType,
			// APIFormat is zero value (empty string)
		}

		result := group.GetAPIFormat()
		if result != tc.expectedFormat {
			t.Fatalf("GetAPIFormat() = %s, expected %s for backward compatible ChannelType '%s'", result, tc.expectedFormat, tc.channelType)
		}
	})
}

func TestGroup_GetAPIFormat_AllValidFormatsReturnCorrectly(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Test that all valid API formats are returned correctly when set explicitly
		allFormats := []string{
			APIFormatOpenAIChat,
			APIFormatOpenAIResponse,
			APIFormatAnthropic,
			APIFormatGemini,
		}

		for _, format := range allFormats {
			group := &Group{
				APIFormat: format,
			}

			result := group.GetAPIFormat()
			if result != format {
				t.Fatalf("GetAPIFormat() = %s, expected %s", result, format)
			}
		}
	})
}
