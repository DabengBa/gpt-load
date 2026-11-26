---
description: "Reasoning-history replay quirk: strict Responses route rejection rules and conversion-shape requirements across protocols."
kind: technical
topic: protocol-conversion
id: QUIRK-002
status: active
external_service: openai-responses-api
verified_date: 2026-11-21
paths:
  - third_party/bifrost-core/**
  - internal/execution/bifrost/**
enforcement: fresh-audit
code:
  paths:
    - third_party/bifrost-core/schemas/mux.go
    - third_party/bifrost-core/schemas/responses.go
    - internal/codexrecovery/classify.go
    - internal/codexrecovery/sanitize.go
---
# Reasoning Replay Quirk

## Symptom

The strict Responses route rejects reasoning items carrying `role` or missing
`summary`. Converted history that does not match the required shape triggers
"unsupported `role` or missing `summary`" rejections on replay. Local fixtures
do not prove live-provider behavior.

## Invariant Action

- Converted reasoning items MUST omit `role` and always carry a `summary`
  array (`summary: []` when only raw reasoning exists; raw text stays in
  `content` as `reasoning_text`).
- The outer Anthropic assistant `role` MUST NOT be transferred into a reasoning
  item. Visible thinking keeps `content.reasoning_text` with its signature;
  redacted data stays in `encrypted_content`.
- Target-specific changes require target behavior evidence. Never delete
  reasoning globally: tool-call turns on some destinations require reasoning.
- Propagate adapter errors instead of silently dropping reasoning. Explicit
  history-rejection recovery is request-local, consumes the existing attempt
  budget, and MUST NOT turn continuation into cross-account failover. Once any
  event reaches the client, the turn MUST NOT be replayed.

## Boundary

The converter never rewrites client-stored history. Serialization tables are
excluded — code is the source of truth.
