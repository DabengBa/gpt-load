package modules

// probeMinOutputTokens is the code-owned output budget every API-key channel
// declares for a manual probe. The realistic probe question needs a few tokens
// to answer; three is the observed upstream floor ("max_tokens must be greater
// than 2") and sixteen stays above provider-specific minimums such as native
// OpenAI Responses without wasting tokens.
const probeMinOutputTokens = 16
