package bifrost

import (
	"strings"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"gpt-load/internal/channel"
)

func TestSDKProviderTimeoutDoesNotPreemptAcceptedRequestTimeouts(t *testing.T) {
	config := providerConfig("https://api.openai.com", false, "openai", false)
	got := config.NetworkConfig.DefaultRequestTimeoutInSeconds
	want := int64((1<<63)-1) / int64(time.Second)
	if int64(got) != want {
		t.Fatalf("SDK provider timeout = %d seconds, want backstop %d", got, want)
	}
}

func TestProviderConfigEnablesAnthropicPromptCaching(t *testing.T) {
	config := providerConfig("https://api.anthropic.com", false, schemas.Anthropic, false)

	if config.PromptCache == nil {
		t.Fatal("Anthropic provider config must enable prompt-cache injection")
	}
	if !config.PromptCache.AutoInject {
		t.Fatal("Anthropic provider config must auto-inject a stable cache breakpoint")
	}
}

func TestAnthropicPromptCachingMarksLatestToolResultWithoutMutatingInput(t *testing.T) {
	config := providerConfig("https://api.anthropic.com", false, schemas.Anthropic, false)
	text := "stable instructions"
	role := schemas.ResponsesInputMessageRoleDeveloper
	toolOutputType := schemas.ResponsesMessageTypeFunctionCallOutput
	callID, result := "call_1", "complete tool output"
	input := []schemas.ResponsesMessage{
		{Role: &role, Content: &schemas.ResponsesMessageContent{ContentStr: &text}},
		{
			Type: &toolOutputType,
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID: &callID,
				Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: &result},
			},
		},
	}

	marked := providerUtils.InjectLastResponsesCacheBreakpoint(config.PromptCache, input)
	if marked[1].CacheControl == nil {
		t.Fatal("the complete tool result must carry the automatic cache marker")
	}
	if marked[0].Content.ContentBlocks != nil {
		t.Fatal("the breakpoint must move from the stable prefix to the latest tool result")
	}
	if input[1].CacheControl != nil {
		t.Fatal("cache injection must not mutate the shared request")
	}
}

func TestSDKDefaultBaseURLComesFromProviderInitialization(t *testing.T) {
	tests := []struct {
		kind channel.ProviderKind
		want string
	}{
		{kind: channel.ProviderOpenAI, want: "https://api.openai.com"},
		{kind: channel.ProviderAnthropic, want: "https://api.anthropic.com"},
		{kind: channel.ProviderGemini, want: "https://generativelanguage.googleapis.com/v1beta"},
		{kind: channel.ProviderDeepSeek, want: "https://api.deepseek.com"},
		{kind: channel.ProviderOpenRouter, want: "https://openrouter.ai/api"},
		{kind: channel.ProviderGroq, want: "https://api.groq.com/openai"},
		{kind: channel.ProviderXAI, want: "https://api.x.ai"},
	}

	for _, test := range tests {
		t.Run(string(test.kind), func(t *testing.T) {
			got, unique, err := sdkDefaultBaseURL(test.kind)
			if err != nil || !unique || got != test.want {
				t.Fatalf("sdkDefaultBaseURL() = %q, %t, %v; want %q, true, nil", got, unique, err, test.want)
			}
		})
	}
}

func TestSDKDefaultBaseURLReportsOnlyUniqueSDKDefaults(t *testing.T) {
	manager := &RuntimeManager{registry: channel.NewRegistry()}
	for _, channelID := range []channel.ID{
		channel.AzureOpenAI,
		channel.AWSBedrock,
		channel.GoogleVertex,
		channel.OpenAICompatible,
		channel.MoonshotAI,
		channel.SiliconFlow,
		channel.ZhipuAI,
		channel.Alibaba,
		channel.Volcengine,
	} {
		t.Run(string(channelID), func(t *testing.T) {
			got, unique, err := manager.DefaultBaseURL(channelID)
			if err != nil || unique || got != "" {
				t.Fatalf("DefaultBaseURL() = %q, %t, %v; want empty, false, nil", got, unique, err)
			}
		})
	}

	openAI, unique, err := manager.DefaultBaseURL(channel.OpenAI)
	if err != nil || !unique || openAI != "https://api.openai.com" || strings.HasSuffix(openAI, "/v1") {
		t.Fatalf("OpenAI DefaultBaseURL() = %q, %t, %v", openAI, unique, err)
	}
}
