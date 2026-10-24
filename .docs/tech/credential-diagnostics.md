---
description: "Own the redacted upstream error message shown on a credential card, which attempts qualify as its source, and why adding it cannot change retry, cooldown, blacklist, or billing decisions."
kind: technical
topic: credential-diagnostics
relations:
  requires:
    - tech/provider-feedback.md
  related:
    - tech/debug-communications.md
    - tech/provider-registry.md
code:
  paths:
    - internal/requestlog/credential_activity.go
    - internal/control/credentials.go
    - internal/control/credential_observations.go
    - internal/gateway/websocket_turn.go
    - internal/health/execution_judge.go
    - internal/platform/redact/error_message.go
    - web/src/shared/control/resources/credentials.ts
---
# Credential Failure Diagnostics

## Scope

- **Owner:** the `last_failure_summary` projection on a credential record and
  the redacted-message extraction and health-classification rules that keep it
  diagnostic-only.
- **Authoritative for:** which attempt supplies the summary, how the message is
  extracted and sanitized, and which decisions must remain unchanged.
- **Excludes:** failure classification and credential health effects
  ([Provider Feedback](provider-feedback.md)), raw capture
  ([Debug Communication Capture](debug-communications.md)), and the credential
  CRUD contract in `docs/single-credential-management.md`.

## Responsibility

A credential card shows why its most recent real attempt failed. The record
carries the Provider's own message next to the existing structured failure
category, so an operator reads the upstream reason instead of inferring it from
a category alone. The summary is additive: it explains a decision the system
already made and never participates in making it.

## Architecture And Constraints

The summary comes from `request_log_attempts.error_summary`, the same redacted
message that attempt-level request logs already store. No new store, column, or
capture path is introduced.

Every API credential id on the page is queried in one batched read. The record
retains the failure summary and nothing else from the activity projection: the
last-used timestamp and the 24-hour success/failure counts the query also
returns are not projected onto the credential record.

The structured evidence fields and the message are separated on purpose. A
structured provider event identifies its failure shape on its own, so the
human-facing message is a fallback signal only for evidence that carries
neither a type nor a code.

## Core Implementation

- `internal/requestlog/credential_activity.go:queryCredentialLatestActivity`
  selects, per credential, the `error_summary` of the newest attempt ordered by
  completion time, request id, and sequence. It excludes locally-dispatched
  attempts, `probe` operations, `ok`, and `downstream_cancel`
  attempts, so an expected or self-inflicted row cannot masquerade as an
  upstream failure.
- The control-plane `internal/control/credential_observations.go:enrichCredentialActivities` projection
  assigns only `LastFailureSummary` to every credential item.
- `internal/control/credentials.go` exposes it as the optional
  `last_failure_summary` field; `web/src/shared/control/resources/credentials.ts`
  projects it as a string and leaves it absent when the Provider omitted it.
- `internal/platform/redact/error_message.go:ExtractErrorMessage` also
  reads the Responses failure envelope `response.error.message`, which is the
  shape a `response.failed` event uses.
- `internal/gateway/websocket_turn.go` builds the WebSocket error summary
  through `summarizeErrorBody` with the resolved credential and header-rule
  secrets, so the redaction path is the same one streaming HTTP uses.
- `internal/health/execution_judge.go:classifyExecutionEvidence` classifies a
  `response.failed` event from its type and code without appending the summary
  to the marker text.

## Related Product Semantics And Binding Points

- Failure categories themselves, retry policy, and credential health effects
  remain owned by [Provider Feedback](provider-feedback.md).
- Product-facing credential surfaces are described in the Astryx credential
  feature components; this document owns only the field and its provenance.
- Code binding: `internal/requestlog/credential_activity.go:QueryCredentialActivity`
- Code binding: `internal/control/credentials.go`
- Code binding: `internal/platform/redact/error_message.go:ExtractErrorMessage`

## Operations, Calculation, And Failure Boundaries

- **Selected attempt:** newest qualifying attempt for that credential, with no
  24-hour window on the summary itself.
- **Query window:** the activity read still requires one, so
  `enrichCredentialActivities` asks for the last 24 hours
  (`credentialDailyUsageWindow`). `QueryCredentialActivity` rejects a window that
  is empty, longer than 24 hours, or holds more than 100 credential ids.
  `enrichCredentialActivities` returns without assigning anything when the read
  fails, so a failed lookup leaves an empty summary.
- **Sanitization:** the message passes the existing request-log path —
  `redact.ExtractErrorMessage` for the Provider field, then
  `sanitizeErrorSummary` (invalid UTF-8 repair, redaction of known secrets, and
  whitespace normalization). It is bounded like every other summary, so a long
  Provider message is truncated rather than rejected.
- **Degradation:** a missing message leaves the field absent and the UI renders
  the failure category alone. A query failure leaves the field empty and does not
  fail the credential read.
- **Decision isolation:** the summary must not change health classification,
  retry or fallback choice, cooldown, blacklist, or billing. Appending it to the
  marker text for `response.failed` evidence is precisely what would let a
  Provider wording change a decision, so it is excluded there.

Regression coverage: `internal/requestlog/credential_activity_test.go` and
`internal/requestlog/model_probe_log_test.go` pin the activity query and probe
isolation, `internal/control/credential_detail_summary_test.go` pins the API
credential detail summary and nonfatal activity lookup failures,
`internal/platform/redact/redactor_test.go` pins the Responses envelope message,
and `internal/gateway/request_log_test.go` pins `summarizeErrorBody`.
