---
description: "Historical request-log facet evidence invariants: failure semantics, non-normalization of stale values, and snapshot boundaries."
kind: technical
topic: request-log-filtering
id: INV-002
status: active
paths:
  - internal/requestlog/**
  - internal/control/request_logs.go
  - web/src/shared/control/resources/request-logs.ts
enforcement: fresh-audit
code:
  paths:
    - internal/requestlog/query_facets.go
    - internal/requestlog/query_list.go
    - internal/control/request_logs.go
---
# Request-Log Facet Invariants

## Invariants

- WHEN a facet query fails, the web client SHALL surface the option error and MUST NOT substitute current group configuration as historical evidence.
- Historical client-model values that fail route validation SHALL remain unchanged in the underlying logs and API result; the system MUST NOT normalize them into a different exact-match filter.
- The client model selector SHALL offer only historical values accepted unchanged by the existing route text validator (nonempty, trimmed, no Unicode control characters, at most 255 UTF-8 bytes).
- Facet dimensions SHALL use all other applied list filters but exclude their own dimension; historical group names come from matching historical snapshots, not the current configuration.
- The two dimension queries execute separately; the system MUST NOT assume a shared snapshot across concurrent writes.

## Boundary

Facet failures never fall back to live configuration; stale or invalid
historical values are preserved verbatim. Endpoint shapes and filter
parameter tables are excluded — code is the source of truth.
