package modules

// probeMinOutputTokens is the code-owned output budget every API-key channel
// declares for a manual probe. A thinking-capable model can spend the whole
// budget on internal reasoning before emitting any visible text, so the budget
// sits well above the small provider minimums (the observed Chat floor
// "max_tokens must be greater than 2" and native OpenAI Responses). 128 keeps
// the probe question answerable without wasting tokens.
const probeMinOutputTokens = 128
