---
description: "How authenticated ModelTrace attribution challenges use real provider routes, frozen request pricing, durable request logs, and an independently deployed analysis service."
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
    - modeltrace/server.py
    - web/src/frontends/astryx/features/models/ModelFingerprintDialog.tsx
---

# ModelTrace Fingerprint Detection

## Scope

- **Owner:** the control-plane fingerprint run, its real model challenge
  executions and request-log observations, and the independently deployed
  ModelTrace analysis service.
- **Authoritative for:** challenge routing, usage and pricing treatment,
  attribution boundaries, service configuration, and failure behavior.
- **Excludes:** ordinary single-route model testing
  ([Provider-Specific Model Test Aliases](model-test-alias.md)), general frozen
  pricing ([Frozen Model Pricing](frozen-model-pricing.md)), and usage report
  aggregation ([Usage Accounting](usage-accounting.md)).

## Responsibility

An administrator can request an attribution run for one saved group/model row.
GPT-Load obtains the challenge set from ModelTrace, executes each challenge
through the model's compiled generated-text route, records each dispatched
challenge as a request, and submits usable answers to ModelTrace for analysis.
The result reports candidate probabilities and per-challenge outcomes; it is a
closed-set attribution over the service's reference bank, not proof of the
upstream model's identity.

## Architecture And Constraints

- The control API is `POST /api/modeltrace-probe` with `{group_id, model}`.
  Route resolution uses the saved model's ModelTrace route, compiled alongside
  the health-probe route. Unsupported or unsaved targets fail before dispatch.
- One schedulable credential is selected for the run. Challenges are executed
  through the ordinary executor and protocol conversion path using that
  credential; challenge prompts and responses are not sent to the browser's
  analysis client.
- ModelTrace is a separate HTTP service. GPT-Load sends it a bearer token for
  challenge retrieval and analysis; it receives no Provider credential. Its
  service description returned to the UI contains only the configured base
  URL and enabled state.
- The run has a six-minute deadline, duplicate runs for the same group/model
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
- `modeltrace/server.py` exposes authenticated challenge and analysis endpoints
  plus health checks. The vendored fingerprint implementation and reference
  bank are identified in `modeltrace/vendor/modeltrace/VENDOR.md`; the service
  is built and deployed independently with `modeltrace/Dockerfile` and
  `modeltrace/docker-compose.yml`.
- The management UI displays candidates, challenge status, duration, status
  code, and links to individual request logs. It identifies closed-set
  attribution limitations and keeps executed challenge details visible when
  analysis fails.

## Operations And Failure Boundaries

Configure `MODELTRACE_URL` and `MODELTRACE_TOKEN` in GPT-Load. The standalone
service uses `MODELTRACE_TOKEN` for its API bearer authentication. The token
must remain server-side; it is not returned by the control API. The service
can be updated independently from GPT-Load.

- Invalid configuration, unavailable analysis service, unsupported target, or
  missing schedulable credential fails before any Provider challenge is sent.
- Once execution begins, each challenge succeeds or fails independently. A
  usable answer may be analyzed even when other challenges fail.
- No usable outputs, rejected outputs, or analysis transport failure returns
  the executed challenge records with a distinct attribution failure. An
  analysis failure does not erase Provider request logs.
- Provider challenge requests are real usage and may incur Provider charges.
  Their usage and estimated cost are included in ordinary request history and
  usage aggregation; unlike control-plane `probe` operations, `modeltrace` is
  not excluded from usage aggregation.
- The reported probability is over the bundled reference bank. An unlisted
  model can still be assigned to its nearest candidate, so the result must not
  be interpreted as a definitive identity check.

## Verification

The API contract and failure handling are covered by
`internal/control/modeltrace_probe_test.go`; request-log and usage projection
are covered by `internal/requestlog/modeltrace_log_test.go`; the standalone
service contract is covered by `modeltrace/tests/test_service.py`. The PR's
reported validation also includes frontend type-check, lint, production build,
ICU and fingerprint-contract checks, and Playwright interaction/preview tests.
Production deployment and a real paid-provider end-to-end run were not part of
that validation.
