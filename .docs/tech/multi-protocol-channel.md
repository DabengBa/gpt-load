---
description: "Multi-protocol gateway quirk: version-looking base paths are preserved verbatim and can produce doubled operation paths."
kind: technical
topic: multi-protocol-channel
id: QUIRK-005
status: active
external_service: self-hosted-api-gateways
verified_date: 2026-11-16
paths:
  - internal/channel/modules/multi_protocol.go
enforcement: fresh-audit
relations:
  related:
    - db/pages/groups.md
    - db/features/realistic-probe.md
code:
  paths:
    - internal/channel/modules/multi_protocol.go
---
# Multi-Protocol Channel Quirk

## Symptom

A version-looking path in the configured base URL is NOT stripped or rejected:
`/v1` produces `/v1/v1/chat/completions` and `/team/v1beta` produces
`/team/v1beta/v1beta/...`. These can still be valid deployment prefixes, so
silently normalizing them would break real deployments.

## Invariant Action

- Configure the gateway root or deployment prefix, NOT an API version path;
  the adapter appends the operation protocol path to the configured base.
- The import form warns about a version-looking path rather than blocking it.
- Declared route operations — not a general promise that every gateway
  supports every protocol — define GPT-Load's boundary. A probe only
  establishes the tested operation; the default probe is OpenAI Chat
  Completions with minimum output budget 128.

## Boundary

General probe preparation, recovery, and limits are owned by
`db/features/realistic-probe.md`. Route declaration tables are excluded —
code is the source of truth.
