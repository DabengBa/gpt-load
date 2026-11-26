---
description: "Mislabeled completed-JSON quirk: providers labeling non-streaming Responses output as an event stream and the narrow relabel rule."
kind: technical
topic: responses-unary-compatibility
id: QUIRK-003
status: active
external_service: openai-responses-api
verified_date: 2026-11-16
paths:
  - internal/execution/bifrost/**
enforcement: fresh-audit
relations:
  related:
    - tech/modeltrace-fingerprint.md
code:
  paths:
    - internal/execution/bifrost/responses_test.go
---
# Responses Unary Compatibility Quirk

## Symptom

A provider can return a completed Responses JSON body labeled
`Content-Type: text/event-stream` for a non-streaming request.

## Invariant Action

- Relabel the content type to `application/json` ONLY when the decoded body is
  a structurally valid completed response (`object` is `response`, `status` is
  `completed`, `output` is a valid typed items array, `error` is absent or
  null) and both client and upstream protocol are native OpenAI Responses.
- Malformed, incomplete, error-bearing, actual-SSE, non-success, or undecodable
  bodies MUST NOT be relabeled; normal downstream handling and safety checks
  continue unchanged.
- Probe and ModelTrace accept a valid completed JSON body under an SSE label;
  anything else follows the existing terminal-event normalization path.

## Boundary

Model-alias rewriting may still run after the relabel. Local fixtures are not
live-provider proof. Feature tables are excluded — code is the source of truth.
