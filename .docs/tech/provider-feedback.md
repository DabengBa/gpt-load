---
description: "Classify Provider attempt feedback and describe retry decisions, budgets, and their safety boundaries."
kind: technical
topic: provider-feedback
relations:
  related:
    - tech/usage-accounting.md
    - tech/affinity-binding.md
    - tech/credential-diagnostics.md
    - db/features/usage-timing-metrics.md
code:
  paths:
    - internal/control/request_logs_mapping.go
    - internal/health/feedback.go
    - internal/health/execution_judge.go
    - internal/gateway/feedback.go
    - internal/gateway/execution_forward.go
    - internal/gateway/attempts.go
    - internal/gateway/stream_observation.go
    - internal/gateway/buffered_stream_forward.go
    - internal/execution/bifrost/converted.go
    - internal/requestlog/worker.go
    - internal/requestlog/query.go
    - internal/storage/migrations/0024_provider_feedback.go
    - web/src/frontends/astryx/features/logs/LogsTable.tsx
---
# Provider Feedback

## Scope

- **Owner:** provider-attempt feedback classification, measurement, persistence,
  request-log projection, usage attempt aggregates, and retry decisions.
- **Authoritative for:** eligibility, thresholds, status and reason invariants,
  performance-fault effects, migration behavior, and the distinction between
  feedback and request success/failure.
- **Excludes:** general request-log retention and usage aggregation rules owned
  by [Usage Accounting](usage-accounting.md), provider raw communication
  capture, and product-facing monitor copy.

## Responsibility

The feedback subsystem gives each eligible Provider attempt a bounded
performance assessment. It records the assessment and observed measurements in
request logs, exposes the final attempt feedback in list and detail responses,
and counts normal, slow, and faulty attempts in the existing attempt usage
aggregates.

Feedback is an observation of the Provider attempt, not a replacement for the
request result. A successful response with poor performance remains available to
the client; the performance fault separately contributes to credential health
failure handling so later selection can avoid the credential according to the
existing consecutive-failure and recovery policy.

The credential record also carries the Provider's redacted message for its most
recent real failure. That message is diagnostic only and is owned by
[Credential Failure Diagnostics](credential-diagnostics.md); it must not feed
back into the classification described here.

## Architecture And Constraints

Only a streamed attempt that was dispatched to the Provider, completed without
an execution error, and was judged as an upstream success is eligible for
performance classification. Provider failures are classified as faulty when
they are attributable to the upstream attempt, including upstream transport,
timeout, incomplete, protocol, and termination failures. Client cancellation,
downstream writes, server shutdown, conversion, and internal failures are not
Provider feedback failures.

The measurement clock observes the first Provider payload/event rather than
semantic first text. For unbuffered responses it subtracts time spent blocked on
downstream writes from the first-response and generation measurements. Buffered
responses do not include downstream release time. WebSocket turns use the same
per-turn measurement boundary.

Feedback is intentionally independent from ordinary request status and failure
classification. An unassessed attempt has empty status and reason. A normal
attempt has no reason. Slow is only paired with `output_rate_slow`; faulty is
paired with `upstream_failure`, `first_response_slow`, or
`output_rate_faulty`.

## Core Implementation

- `internal/health/feedback.go:ClassifyFeedback` validates measurements and
  applies the classification rules. First response over 30,000 ms is faulty;
  output below 10 tokens/s is faulty; output from 10 up to but excluding 20
  tokens/s is slow; output at 20 tokens/s or higher is normal. When both a slow
  first response and faulty output rate exist, `first_response_slow` is the
  reason. Missing or ineligible measurements remain unassessed.
- `internal/gateway/feedback.go` collects measurements, identifies upstream
  failure evidence, and applies the performance-fault health effect. A
  successful performance fault uses the existing credential-failure path with
  `feedback.performance_fault` and no retry for the current request.
- `internal/requestlog/mapper.go` bounds feedback status, reason, first-response
  milliseconds, and output rate before persistence. `internal/requestlog/query.go`
  projects feedback for every attempt and separately derives the final attempt
  feedback for list and detail records.
- `internal/control/request_logs_mapping.go` exposes feedback status, reason,
  first-response measurement, and output rate. The Astryx logs view consumes the
  classification as presentation only: the final attempt's Provider
  first-response time and every displayed duration carry the `slow` or `faulty`
  tone, while no surface renders the status or reason as a label. Feedback
  therefore never shares a field with request status or failure category.
  The list displays the final attempt's Provider first-response time and attempt
  duration (`final_attempt_duration_ms`), excluding earlier failed attempts.
  Output rate uses that attempt's measured `provider_tokens_per_second` rather
  than recomputing from request-level timing. Displayed durations use seconds
  with one decimal place. Request-level timing and all attempts remain recorded
  and available in the detail view; missing final-attempt timing displays `—`.

## Related Product Semantics And Binding Points

- The logs view is the product-facing consumer; this document owns the
  implementation contract behind its timing tone and attempt details.
- Whether a request result learns an affinity binding, and which upstream route
  that binding records, is owned by [Affinity Binding](affinity-binding.md);
  this document owns the feedback status that its final-success gate consumes.
- Product-facing usage timing semantics remain owned by [Usage Duration And
  First-Response Metrics](../db/features/usage-timing-metrics.md).
- Code binding: `internal/health/feedback.go:ClassifyFeedback`
- Code binding: `internal/gateway/feedback.go:providerFeedbackForAttempt`
- Code binding: `internal/requestlog/mapper.go:validateAttemptFeedback`
- Code binding: `internal/storage/migrations/0024_provider_feedback.go:Up0024`

## Persistence, Aggregation, And Failure Boundaries

Migration `0024_provider_feedback` adds bounded provider measurements and
status/reason columns to `request_log_attempts`. It also adds normal, slow, and
faulty attempt counts to both attempt aggregation journals and durable attempt
stats. New columns default to empty or zero values. The migration validates
SQLite tables, constraints, allowed status/reason pairs, non-negative values,
and the invariant that feedback counts do not exceed total attempts. It does
not infer feedback for historical rows.

The request-log worker records one feedback count for each persisted attempt
whose status is normal, slow, or faulty. Unassessed attempts contribute to the
total attempt count but none of the three feedback counts. Journal insertion,
application, and recovery remain idempotent, so retries of log processing do
not double-count feedback. Query aggregation sums the three counts with checked
integer arithmetic and rejects invalid persisted values.

Provider output rate is calculated as `output_tokens * 1000 / generation_ms`.
The rate is absent when output tokens or generation duration are unavailable or
non-positive. First-response and rate measurements may therefore be present
without a classification, and a request remains valid when feedback is
unassessed. This keeps missing observations distinct from normal performance.

Regression coverage is provided by the health classification tests, gateway
provider-feedback tests, request-log mapping/query tests, migration tests, and
the monitor provider-feedback Playwright spec.

## Same-provider retry budget

The gateway receives the resolved `first_provider_attempts` setting and uses it
to limit actual attempts that reuse the first executed Selection (same group,
credential, and upstream model), including that first execution. The setting
defaults to 1 and values below 1 are rejected. Every same-provider replay and
candidate switch consumes the same request-wide `retry_count` budget, so the
configured same-provider limit never creates additional attempts beyond the
global budget.

Non-stream empty completions and upstream host errors enter the ordinary HTTP
candidate retry loop; they do not receive a separate retry budget. Empty
completion remains an ambiguous attempt with no health effect. Ordinary upstream
host errors retain the group-skip effect; local failures and an ineligible
committed buffered response do not apply that effect. The normal committed-response,
buffered-replay eligibility, operation replay-safety, and WebSocket turn
boundaries still apply. Failed attempts remain individually visible in request
logs even when a later attempt determines the final request result.

### Retry To Obtain A Usable Upstream Answer

Internal retry and candidate rotation are intentional: within the existing
candidate selection, health/admission rules, request-wide retry budget, and
buffered replay boundary, the Gateway should keep trying eligible upstreams so
that the client can receive a usable upstream answer. Do not remove an existing
retry classification solely to reduce retries or because the earlier upstream
may have processed the request; evaluate that change against this delivery
intent and the existing replay boundary.

In particular, a complete Responses `response.incomplete` with
`incomplete_details.reason = "max_output_tokens"` remains an unsuccessful
buffered attempt eligible for the existing retry/rotation policy. It must not
be released immediately just because it is a syntactically complete upstream
event. If another eligible candidate succeeds, release that candidate's
response. If candidates or budget are exhausted, preserve the current buffered
failure behavior; do not add cross-attempt response storage or an incomplete
response fallback without a separately approved behavior change.

The converted Responses streaming adapter also retains its existing internal
replay classification for HTTP 429 `rate_limit_exceeded`, allowing the Gateway
to try another candidate. This is specific to the existing classification; it
does not authorize adding replay eligibility to every HTTP 429 code, including
usage-limit or model-capacity responses, without evidence and a policy decision.

These retries remain subject to the current request-wide budget, candidate
eligibility and health effects. A heartbeat without released payload does not
by itself stop an eligible buffered retry. Once upstream payload is released
to the client, another candidate must not be replayed; cancellation or the
request-wide deadline also stops further attempts. Reported usage from failed
attempts remains billable under [Frozen Model Pricing](frozen-model-pricing.md).
The intended paths are covered by
`internal/gateway/provider_incomplete_integration_test.go`,
`internal/gateway/http_rate_limit_integration_test.go`, and
`internal/execution/bifrost/http_rate_limit_replay_test.go`.

The gateway detects `upstream_empty_completion` only for a completed non-stream
OpenAI Chat Completions 2xx response with a complete JSON object, at least one
assistant choice without consumable payload, an absent, empty or `stop` finish
reason for every choice, and complete usage with positive output tokens. An
explicit `content_filter` finish reason is not an empty completion. The gateway
marks this diagnosis internally; an upstream error code alone cannot assert it.
If no later attempt succeeds, the existing deferred-attempt fallback determines
the final response; an empty completion's original response is returned only if
that attempt is selected by the fallback. See
`internal/gateway/retry_rotation_http_integration_test.go` for candidate order,
budget exhaustion, and attempt log coverage.

Before a same-provider replay, current credential and route-entry health plus
group skip state are checked. Cooldown, blacklist, breaker, and replay-safety
effects remain authoritative; a changed credential reference ends the replay
phase. A credential-refresh retry also uses the global budget and ends the
same-provider replay phase. Buffered streaming follows the same accounting:
when a buffered authentication failure requires credential refresh, the refresh
retry is reserved before the next replay decision, so the rejected credential
cannot consume the remaining budget through an ordinary first-provider replay.
WebSocket turns retain their separate unbound/unsent replay boundary.

Independent proof is provided by `internal/gateway/retry_budget_test.go`,
`internal/gateway/buffered_stream_auth_test.go`,
`internal/state/runtime_settings_test.go`, and
`internal/control/settings_http_test.go`. The first set covers replay order,
rotation, and the shared budget; the buffered-stream cases cover refresh,
fallback, and one/two-attempt exhaustion; the settings tests cover the default,
positive values, lower-bound rejection, persistence, and HTTP round trip.
