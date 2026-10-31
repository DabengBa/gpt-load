package bifrost

import (
	"encoding/json"

	providerutils "github.com/maximhq/bifrost/core/providers/utils"
)

const anthropicMinimumReasoningMaxTokens = 1024

// normalizeAnthropicNativeReasoning keeps an entry-owned effort compatible
// with an already enabled manual-thinking request. Native passthrough does
// not run Bifrost's typed request builder, so the budget must be updated here.
// Requests without a usable max_tokens are deliberately left for Anthropic to
// validate; this function never invents an output limit.
func normalizeAnthropicNativeReasoning(body []byte, effort string) ([]byte, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, err
	}
	var thinking struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(root["thinking"], &thinking); err != nil || thinking.Type != "enabled" {
		return body, nil
	}
	var maxTokens int
	if err := json.Unmarshal(root["max_tokens"], &maxTokens); err != nil || maxTokens <= anthropicMinimumReasoningMaxTokens {
		return body, nil
	}
	budget, err := providerutils.GetBudgetTokensFromReasoningEffort(effort, anthropicMinimumReasoningMaxTokens, maxTokens)
	if err != nil || budget < anthropicMinimumReasoningMaxTokens || budget >= maxTokens {
		return body, nil
	}
	var thinkingObject map[string]json.RawMessage
	if err := json.Unmarshal(root["thinking"], &thinkingObject); err != nil {
		return body, nil
	}
	encoded, err := json.Marshal(budget)
	if err != nil {
		return nil, err
	}
	thinkingObject["budget_tokens"] = encoded
	root["thinking"], err = json.Marshal(thinkingObject)
	if err != nil {
		return nil, err
	}
	return json.Marshal(root)
}

func effortFromAnthropicBody(body []byte) string {
	var root struct {
		OutputConfig struct {
			Effort string `json:"effort"`
		} `json:"output_config"`
	}
	if json.Unmarshal(body, &root) != nil {
		return ""
	}
	return root.OutputConfig.Effort
}
