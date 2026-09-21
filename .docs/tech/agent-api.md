---
description: "The Agent control-plane API contract for independent credentials, redacted diagnostics, and approved model-route changes."
kind: technical
topic: agent-control-plane
code:
  paths:
    - internal/agent
    - internal/control/agent_credentials.go
    - internal/control/proposals.go
    - internal/storage/migrations/0020_agent_credentials.go
    - internal/storage/migrations/0021_agent_change_proposals.go
---
# Agent Control-Plane API

## Scope

- **Owner:** `internal/agent` and the control-plane proposal/credential boundary.
- **Authoritative for:** current HTTP resource shape, authentication scopes, redaction rules, proposal lifecycle, and durable operation behavior.
- **Excludes:** the web UI, ordinary provider/data-plane authentication, and MCP transport details owned by [Agent Operations MCP](mcp-agent-ops.md).

## Responsibility

The Agent API provides a separately authenticated machine surface over existing GPT-Load facts. It supports redacted diagnostics and a narrow, approved change path for model-route `weight` and `priority`; it does not create a second request-log, usage, or capture store.

## Architecture And Constraints

Agent credentials are independent of administrator `AUTH_KEY` and data-plane access keys. The credential store persists only an HMAC-SHA-256 digest, returns plaintext once at creation, and checks disabled or expired status on every request. Grantable scopes are `diagnostics:read`, `changes:propose`, and `changes:apply`.

Read projections are built from request logs, attempts, route snapshots, health counters, usage aggregates, and debug-capture metadata. Evidence is metadata-only: authorization material, cookies, API keys, complete headers, request/response bodies, and raw capture downloads remain outside the Agent surface. Capture results distinguish captured, not retained, and unavailable.

Proposal creation records the current snapshot revision and explicit integer target/current values without mutating routes. Administrator approval records the runtime epoch. Apply rechecks approval, epoch, snapshot revision, and expected values before committing the route patch, proposal binding, and control operation together. A bound operation is replayed or recovered before later approval/value checks; incomplete recovery remains observable as `CONTROL_OPERATION_INCOMPLETE`.

## Core Implementation

- `internal/agent/http_routes.go` declares the versioned Agent route module and scope boundaries.
- `internal/agent/auth.go` and `internal/agent/credentials.go` authenticate Bearer credentials and resolve the caller on each request.
- `internal/agent/requests.go`, `projection.go`, `health.go`, `routes.go`, and `usage.go` expose bounded read projections.
- `internal/control/agent_credentials.go` owns administrator credential management and idempotent creation.
- `internal/control/proposals.go` owns proposal creation, administrator approval, apply, replay, and read-only operation projections.
- Migrations `0020_agent_credentials` and `0021_agent_change_proposals` install the credential ledger and unique proposal-to-operation binding.

## Related Product Semantics And Binding Points

- Product-facing administration of Agent access is represented by the existing control-plane UI boundary; this API document does not define a new UI page.
- Code binding: `internal/agent/http_routes.go:Module`
- Code binding: `internal/control/proposals.go:CreateChangeProposal`, `ApplyChangeProposal`

## Operations, Calculation, And Failure Boundaries

The API uses bounded pages and validates identifiers, timestamps, models, protocols, and operations before querying. Token and estimated nano-USD values are projected as decimal strings where declared by capabilities. Read paths use query-only dependencies and do not trigger runtime recovery or configuration publication. Proposal apply is serialized by the control write lock and durable operation stages; replay uses the stored canonical result and never returns the one-time credential secret.
