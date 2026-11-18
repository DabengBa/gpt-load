---
description: "The Agent control-plane API contract for independent credentials, redacted diagnostics, and scoped model-route maintenance."
kind: technical
topic: agent-control-plane
relations:
  related:
    - tech/usage-accounting.md
code:
  paths:
    - internal/agent
    - internal/control/agent_credentials.go
    - internal/control/agent_model_routes.go
    - internal/control/proposals.go
    - internal/storage/migrations/0020_agent_credentials.go
    - internal/storage/migrations/0021_agent_change_proposals.go
---
# Agent Control-Plane API

## Scope

- **Owner:** `internal/agent` and its control-plane credential and model-route services.
- **Authoritative for:** current HTTP resource shape, authentication scopes, redaction rules, model-route maintenance, proposal lifecycle, and durable operation behavior.
- **Excludes:** the web UI and ordinary provider/data-plane authentication.

## Responsibility

The Agent API provides redacted diagnostics and scoped model-route maintenance over existing GPT-Load facts; it does not create a second request-log, usage, or capture store. The existing proposal-based path for `weight` and `priority` remains available.

## Architecture And Constraints

Agent credentials are independent of administrator `AUTH_KEY` and data-plane access keys. The credential store persists only an HMAC-SHA-256 digest, returns plaintext once at creation, and checks disabled or expired status on every request. Grantable scopes are `diagnostics:read`, `changes:propose`, `changes:apply`, and `routes:write`. Direct route writes require an explicit `routes:write` grant and never use proposal approval.

Read projections are built from request logs, attempts, route snapshots, health counters, usage aggregates, and debug-capture metadata. Usage aggregates include total duration and observed first-response totals with their sample counts; a missing first-response sample is not reported as a zero-millisecond observation. Evidence is metadata-only: authorization material, cookies, API keys, complete headers, request/response bodies, and raw capture downloads remain outside the Agent surface. Capture results distinguish captured, not retained, and unavailable.

Proposal creation records the current snapshot revision and explicit integer target/current values without mutating routes. Administrator approval records the runtime epoch. Apply rechecks approval, epoch, snapshot revision, and expected values before committing the route patch, proposal binding, and control operation together. Administrators can revoke pending or approved proposals through `POST /api/agent-change-proposals/:proposal_id/revoke`; a proposal already bound to an operation cannot be detached. A process restart invalidates approvals tied to the old runtime epoch, and stale proposals can be revoked and recreated. A bound operation is replayed or recovered before later approval/value checks; incomplete recovery remains observable as `CONTROL_OPERATION_INCOMPLETE`.

Completed operation stage data is compacted after the retention period while the comparator tombstone remains. A compacted proposal still reads as `applied` with `can_reconcile: false`; an apply retry after the replay result expires returns `IDEMPOTENCY_RESULT_EXPIRED` rather than attempting recovery.

## Configured Model Routes

Paths below are relative to `/api/agent/v1`.

| Method | Path | Scope |
| --- | --- | --- |
| GET | `/model-routes` | `diagnostics:read` |
| POST | `/groups/{group_id}/model-routes` | `routes:write` |
| PATCH | `/groups/{group_id}/model-routes/{entry_id}` | `routes:write` |
| DELETE | `/groups/{group_id}/model-routes/{entry_id}` | `routes:write` |

The inventory is `data.groups[]`, with `group_id`, `enabled`, and `entries[]`. It includes disabled groups and entries and reports configured values, not effective schedule defaults. `priority` and `weight` are `null` when unset. Use the inventory's opaque `entry_id` for mutations; for legacy entries without a persisted entry ID, it contains the persisted test alias, which remains usable after the entry ID is backfilled.

POST requires `id` (upstream model ID) and generates entry identity on the server. POST and PATCH accept only `id`, `alias`, `reasoning_effort`, `priority`, and `weight`. PATCH leaves omitted fields untouched; `null` clears optional values, but cannot clear `id`. POST and PATCH return the resulting entry in `data`. Writes use the control configuration transaction and publish the updated runtime snapshot. They do not replace the whole model list.

Group URLs, credentials, and names are excluded from the inventory and mutation responses. The `/routes` scheduling view keeps `group_id` and emits an empty `group_name` because a configured name may be the upstream hostname. Request diagnostics retain structured error codes and attempt status, but emit an empty `error_summary` because free-form provider and transport errors can contain URLs or keys. Both fields remain present for wire compatibility.

## Core Implementation

- `internal/agent/http_routes.go` declares the versioned Agent route module and scope boundaries.
- `internal/agent/auth.go` and `internal/agent/credentials.go` authenticate Bearer credentials and resolve the caller on each request.
- `internal/agent/requests.go`, `projection.go`, `health.go`, `routes.go`, and `usage.go` expose bounded read projections.
- `internal/agent/usage.go` validates and projects duration and first-response totals without exposing unsafe integers or sample counts larger than request counts.
- `internal/control/agent_credentials.go` owns administrator credential management and idempotent creation.
- `internal/agent/model_routes.go` and `internal/control/agent_model_routes.go` own the narrow route inventory, scoped mutation handlers, and serialized single-entry updates.
- `internal/control/proposals.go` owns proposal creation, administrator approval/revocation, apply, replay, and read-only operation projections.
- Migrations `0020_agent_credentials` and `0021_agent_change_proposals` install the credential ledger and unique proposal-to-operation binding.

## Related Product Semantics And Binding Points

- Product-facing administration of Agent access is represented by the existing control-plane UI boundary; this API document does not define a new UI page.
- Code binding: `internal/agent/http_routes.go:Module`
- Code binding: `internal/control/proposals.go:CreateChangeProposal`, `ApproveChangeProposal`, `RevokeChangeProposal`, `ApplyChangeProposal`

## Operations, Calculation, And Failure Boundaries

The API uses bounded pages and validates identifiers, timestamps, models, protocols, and operations before querying. Token and estimated nano-USD values are projected as decimal strings where declared by capabilities. Read paths use query-only dependencies and do not trigger runtime recovery or configuration publication. Proposal apply is serialized by the control write lock and durable operation stages; replay uses the stored canonical result and never returns the one-time credential secret. Administrator revoke is audited and only affects proposals without a committed operation binding.
