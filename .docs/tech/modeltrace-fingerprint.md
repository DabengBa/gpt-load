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
- One schedulable credential is selected for the run. Challenges are executed
  through the ordinary executor and protocol conversion path using that
  credential; challenge prompts and responses are not sent to the browser's
  analysis client.
- The SDK and its matching reference data are bundled with GPT-Load. The data
  is loaded locally on the first fingerprint run; no separate analysis service
  or runtime download is required.
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
- The run freezes the published model-price table before dispatching its
  challenge batch. Reported valid usage is retained even when the attempt
  failed; absent or invalid usage is marked missing rather than treated as
  free. Pricing uses the frozen table and standard pricing mode, consistent
  with [Frozen Model Pricing](frozen-model-pricing.md).
- `internal/control/modeltrace_detector.go` stages the embedded reference data
  in a private temporary directory for SDK loading, then removes the directory.
  `THIRD_PARTY_NOTICES.md` identifies the SDK and matching data snapshot.
- The management UI displays candidates, challenge status, duration, status
  code, and links to individual request logs. It identifies closed-set
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
