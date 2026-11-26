---
description: Keep internal retry and candidate rotation because the delivery goal is a usable upstream answer
kind: adr
topic: provider-retry-usable-answer
id: ADR-0006
status: active
supersedes: null
created_at: 2026-11-11
affects_symbols:
  - internal/state/runtime_settings.go#SettingFirstProviderAttempts
---
# ADR-0006: Retry and rotate candidates to obtain a usable upstream answer

## Context

Retry can repeat work that an earlier upstream may already have processed, and
it consumes the request-wide budget. Removing classifications would reduce
attempts but can leave the client without a usable answer.

## Decision

Internal retry and candidate rotation are intentional within existing
candidate-selection, health/admission, request-wide budget, and buffered replay
boundaries. Do not remove an existing retry classification solely to reduce
retries or because an earlier upstream may have processed the request.

A complete Responses `response.incomplete` with
`incomplete_details.reason = "max_output_tokens"` remains an unsuccessful
buffered attempt eligible for retry/rotation. A syntactically complete upstream
event is not a reason to release it early. The existing internal replay
classification for HTTP 429 `rate_limit_exceeded` is retained, but it must not
be generalized to every HTTP 429 code without evidence and a policy decision.

## Rejected

- Treating any syntactically complete event as final: this releases unusable
  output and defeats rotation.
- Replaying another candidate after payload has been released to the client:
  this violates the buffered replay boundary and can mix responses.
- Storing cross-attempt responses or adding an incomplete-response fallback
  without a separate behavior decision.

## Constraint

Retries remain subject to the request-wide budget, candidate eligibility and
health effects. Once upstream payload is released, another candidate must not
be replayed; cancellation or the request-wide deadline also stops attempts.
