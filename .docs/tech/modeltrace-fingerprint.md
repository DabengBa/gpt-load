---
description: "How fingerprint challenges use lm-detector, real provider routes, frozen request pricing, and durable request logs."
kind: technical
topic: modeltrace-fingerprint
relations:
  related:
    - tech/model-test-alias.md
    - tech/frozen-model-pricing.md
    - tech/usage-accounting.md
code:
  paths:
    - internal/control/modeltrace_probe.go
    - internal/control/modeltrace_retry.go
    - internal/control/modeltrace_context_check.go
    - internal/control/modeltrace_client.go
    - internal/channel/compiler.go
    - internal/control/modeltrace_detector.go
    - web/src/frontends/astryx/features/models/ModelFingerprintDialog.tsx
---

# ModelTrace Fingerprint Detection

## Scope

- **Owner:** the control-plane fingerprint run, its real model challenge
  executions, request-log observations, and local lm-detector analysis.
- **Authoritative for:** challenge routing, usage and pricing treatment,
  attribution boundaries and failure behavior.
- **Excludes:** ordinary single-route model testing
  ([Provider-Specific Model Test Aliases](model-test-alias.md)), general frozen
  pricing ([Frozen Model Pricing](frozen-model-pricing.md)), and usage report
  aggregation ([Usage Accounting](usage-accounting.md)).

## Responsibility

An administrator can request an attribution run for one saved group/model row.
GPT-Load generates the challenge set locally, executes each challenge
through the model's compiled generated-text route, records each dispatched
challenge as a request, and analyzes usable answers with lm-detector.
The result reports available candidate scores and per-challenge outcomes; it is a
closed-set attribution over the bundled reference bank, not proof of the
upstream model's identity.

## Architecture And Constraints

- The control API is `POST /api/modeltrace-probe` with `{group_id, model}`.
  Route resolution uses the saved model's ModelTrace route, compiled alongside
  the health-probe route. Unsupported or unsaved targets fail before dispatch.
- A result with retryable challenges includes a `run_id`. The administrator can
  resend just one failed or detector-rejected challenge with
  `POST /api/modeltrace-probe/retry` and `{group_id, model, run_id, challenge_id}`.
  Retry state is kept in bounded process memory for up to 30 minutes after the
  latest result. If the state is gone or the original route or credential
  changed, start a new detection.
- One schedulable credential is selected for the run. Challenges are executed
  through the ordinary executor and protocol conversion path using that
  credential; challenge prompts and responses are not sent to the browser's
  analysis client.
- The SDK and its matching reference data are bundled with GPT-Load. The data
  is loaded locally on the first fingerprint run; no separate analysis service
  or runtime download is required.
- The Hostinger release entry point checks upstream before every release.
  `scripts/update-lm-detector.py` resolves one upstream commit and updates the
  SDK and both embedded data files together. Relevant changes are validated and
  merged via a `dev` PR before deployment; unrelated changes are skipped and a
  license change requires manual review. See the [release pipeline](release-pipeline.md).
  The legacy scheduled workflow is not relied on: GitHub schedules require the
  workflow on the default branch. A Dependabot PR for the Go SDK alone must not
  be merged without updating the matching data.
- The run has a ten-minute deadline, duplicate runs for the same group/model
  are rejected, and at most four runs execute concurrently per process.
- Attribution is diagnostic only. It does not trigger health recovery,
  cooldown, circuit breaking, retries, routing, or affinity changes.

## Core Implementation

- `internal/control/modeltrace_probe.go` resolves the target, acquires the
  credential, runs the challenges, retains each challenge result, constructs
  request events, and sends usable outputs for analysis.
- Every dispatched challenge has operation `modeltrace` and its own request
  ID/log entry, including failed executions. A failure before dispatch has no
  request log. Request-log persistence remains best effort and does not replace
  the API result.
- A manual retry sends the original challenge prompt once using the same route
  and credential, with a newly frozen price table and its own request log. The
  latest attempt replaces just that challenge in the result and updates the
  attribution; the other challenges and the context check are preserved.
- The run also sends one standalone, user-only context check in parallel on the
  same route and credential. Its request is logged and priced as `modeltrace`,
  but its response never enters the fingerprint analysis. The prompt text is
  six tokens in both `o200k_base` and `cl100k_base`. `context_check` compares
  this estimate with the provider's total reported input tokens, including
  cache reads/writes and excluding output. A positive difference greater than
  24 tokens flags suspected extra context. Failed requests or requests without
  complete, usable input usage return `unavailable` independently of attribution.
  The threshold allows for routine framing and tokenizer differences; larger
  legitimate overhead can still trigger it, and falsified usage can hide changes.
- ModelTrace challenges request low reasoning effort by default; ordinary
  probes and user requests do not inherit this setting. Responses uses
  `reasoning.effort` and Chat uses `reasoning_effort`. Responses consults SDK
  effort capabilities, with a narrow high-only fallback for the exact
  `gpt-5-pro` and `gpt-5-pro-2025-10-06` IDs (and their `openai/` OpenRouter
  IDs); a capability record takes precedence. Native Anthropic and
  Anthropic-protocol gateway requests use `output_config.effort` for models
  accepting it; otherwise they use the SDK's low-effort thinking budget mapping
  (for example Haiku 4.5 and Sonnet 4.5). If the output limit cannot fit the
  minimum thinking budget, reasoning controls are omitted. Native Gemini and
  Vertex omit thinking controls for models without reasoning support; otherwise
  the existing model-aware Gemini normalization uses a thinking level for
  newer models (which may be clamped) or a derived budget for older models.
  For a positive Gemini thinking budget, ModelTrace increases `maxOutputTokens`
  by that budget to retain the original answer allowance. If a known model
  output cap cannot fit the total, it omits the explicit thinking control and
  retains the original output limit instead. Typed provider conversions use the
  SDK's own mapping.
- Low effort is a request, not a guarantee: model and provider support varies,
  compatible endpoints may reject or ignore the parameter, and SDK conversions
  may omit or translate it (including on Bedrock). Anthropic and Gemini
  capability checks rely on the configured upstream model name and available
  SDK capabilities; opaque gateway aliases and unknown models can be
  misclassified, and the SDK's budget fallback does not prove that every older
  Anthropic model accepts thinking. Gemini normalization also relies on the
  model name; when no output cap is known, the enlarged limit may still be
  rejected by the actual upstream. Gateway model names need not identify their
  backing model, and unknown Responses aliases retain low unless capabilities
  say otherwise. Omitting a thinking control leaves the provider's default in
  effect, not necessarily disabled. There is no retry without reasoning controls
  on rejection; such a challenge can fail.
- The run freezes the published model-price table before dispatching its
  challenge batch. Reported valid usage is retained even when the attempt
  failed; absent or invalid usage is marked missing rather than treated as
  free. Pricing uses the frozen table and standard pricing mode, consistent
  with [Frozen Model Pricing](frozen-model-pricing.md).
- `internal/control/modeltrace_detector.go` stages the embedded reference data
  in a private temporary directory for SDK loading, then removes the directory.
  `THIRD_PARTY_NOTICES.md` identifies the SDK and matching data snapshot.
- The management UI displays candidates, challenge status, duration, status
  code, and links to individual request logs. It reports context fidelity
  separately with the expected and reported input counts, delta, and log link.
  It identifies closed-set
  attribution limitations and keeps executed challenge details visible when
  analysis fails.

## Operations And Failure Boundaries

- Fingerprint detection is enabled by default; set `MODELTRACE_ENABLED=false`
  to disable it before any challenge is dispatched.
- Detector loading failure, unsupported target, or
  missing schedulable credential fails before any Provider challenge is sent.
- Once execution begins, each challenge succeeds or fails independently. A
  usable answer may be analyzed even when other challenges fail.
- No usable outputs or an analysis error returns the executed challenge
  records with a distinct attribution failure. An
  analysis failure does not erase Provider request logs.
- Provider challenge requests are real usage and may incur Provider charges.
  Their usage and estimated cost are included in ordinary request history and
  usage aggregation; unlike control-plane `probe` operations, `modeltrace` is
  not excluded from usage aggregation.
- The SDK reports `scored`, `partial`, `unscorable`, or `not_model`; a
  probability may be unavailable. The reported probability is over the bundled reference bank. An unlisted
  model can still be assigned to its nearest candidate, so the result must not
  be interpreted as a definitive identity check.

## Verification

The API contract and failure handling are covered by
`internal/control/modeltrace_probe_test.go`; request-log and usage projection
are covered by `internal/requestlog/modeltrace_log_test.go`; local detector
loading and SDK result states are covered by `internal/control/modeltrace_local_test.go`.
