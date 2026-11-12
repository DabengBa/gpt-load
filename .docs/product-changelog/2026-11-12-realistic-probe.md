# Actual Request Testing

- **Change:** Saved model rows and the dispatch center can open an editable
  request workspace for Chat Completions, Responses, Anthropic Messages and
  Gemini. It observes streaming completion and links to actual request attempts.
  Fixed Responses diagnostics now send a user-message input list.
- **Why:** Fixed prompts, forced non-streaming and diagnostic route selection
  could disagree with a client's protocol, parameter rules and delivery path.
- **Scope:** Normal data-plane requests with a transient supplied access key;
  normal routing, usage and health effects apply. Existing diagnostic and
  recovery controls remain. No tool execution or browser-persisted credentials;
  opted-in raw debug capture can record the request headers.
- **Current owners:** [product behavior](../db/features/realistic-probe.md) and
  [technical implementation](../tech/realistic-probe.md).
- **Evidence and delivery:** `web/scripts/realistic-probe.test.ts`,
  `web/e2e/astryx-realistic-probe.spec.ts`, and the Responses native/multiprotocol
  wire tests under `internal/execution/bifrost/`. Local implementation on
  `feat/realistic-probe`; not deployed.
