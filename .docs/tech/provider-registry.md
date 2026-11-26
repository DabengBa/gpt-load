---
description: "OpenRouter catalog merge and Embeddings wire-authority quirk: published model filtering and why the raw body stays authoritative."
kind: technical
topic: model-discovery
id: QUIRK-001
status: active
external_service: openrouter-models-catalog
verified_date: 2026-10-20
paths:
  - internal/execution/bifrost/**
enforcement: fresh-audit
relations:
  related:
    - tech/reasoning-replay.md
code:
  paths:
    - internal/execution/bifrost/list_models.go
    - internal/execution/bifrost/embeddings.go
---
# Provider Registry Quirks

## Symptom

The Bifrost SDK merges the OpenRouter decisions catalog
(`/v1/models?output_modalities=decisions`) into the same model list, while
GPT-Load has no native decisions endpoint. Merged decision entries decode
without an architecture, so chat/embeddable capability cannot be judged. For
Embeddings, the typed marker is only SDK preflight input: values Go cannot
represent (for example token IDs beyond Go integer range) would be lost if the
typed value were sent.

## Invariant Action

- Drop OpenRouter entries that declare only unservable output modalities or
  carry no architecture before publishing the sanitized catalog; mixed
  `["decisions","text"]` entries stay. Non-OpenRouter providers publish the
  full catalog unchanged.
- The typed Embeddings marker MUST never become the request payload: send the
  sanitized raw body so Go-unrepresentable choices pass through verbatim.
- Treat the raw response body as the upstream authority; typed response data is
  cleared before projection and gateway handling.
- Catalog narrowing only shrinks the published list. It MUST NOT change
  routing, health, or stored model selection; groups may still configure
  models outside the list.

## Boundary

A probe or fixture certifies the tested operation only; neither establishes
compatibility with every external gateway or provider. Catalog rule tables are
excluded — code is the source of truth.
