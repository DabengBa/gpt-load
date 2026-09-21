# Agent API

The Agent surface is an independently authenticated, machine-oriented view of
the existing control-plane facts. It does not create a second request or
telemetry store.

## Authentication

Administrators create credentials through the control API:

- `POST /api/agent-credentials`
- `POST /api/agent-credentials/:credential_id/disable`

The create request requires `Idempotency-Key`. The response contains a
high-entropy Bearer secret once. Only its HMAC-SHA-256 digest is stored. The
secret is not included in logs, audit records, telemetry, errors, or replay
payloads. Disabled and expired credentials fail closed on the next request.

The only grantable Agent scopes are:

- `diagnostics:read`: redacted request, attempt, evidence metadata, route,
  health, usage, and capability reads.
- `changes:propose`: create immutable model-route weight/priority proposals.
- `changes:apply`: apply an already approved proposal with an
  `Idempotency-Key`.

Agent credentials cannot approve changes, read raw capture content, access the
administrator control routes, or access the data-plane provider endpoint.

## Read API

All responses use `schema_version: 1`. The read endpoints are:

- `GET /api/agent/v1/capabilities`
- `GET /api/agent/v1/requests`
- `GET /api/agent/v1/requests/:request_id`
- `GET /api/agent/v1/requests/:request_id/evidence`
- `GET /api/agent/v1/routes`
- `GET /api/agent/v1/health`
- `GET /api/agent/v1/usage`

Request detail exposes ordered provider attempts by an explicit logical
`attempt_ref`. Evidence links use the capture metadata's explicit logical
attempt reference when present. A capture storage sequence is never treated as
the request-log attempt sequence.

Evidence is metadata-only. Headers, authorization material, cookies, API keys,
request bodies, response bodies, and raw capture downloads are not returned.
The evidence state distinguishes captured, not retained, and unavailable.

Token counts and estimated nano-USD values use decimal-string fields where
declared by `capabilities`. A missing or incomplete fact remains represented as
unknown/incomplete rather than being converted into an empty or zero result.

All read handlers use query-only interfaces. They do not trigger provider
probes, runtime recovery, configuration publication, or lazy mutation.

## Controlled Changes

Agent proposal endpoints are:

- `POST /api/agent/v1/change-proposals` (`changes:propose`)
- `GET /api/agent/v1/change-proposals/:proposal_id` (`diagnostics:read`)
- `POST /api/agent/v1/change-proposals/:proposal_id/apply` (`changes:apply`)
- `GET /api/agent/v1/control-operations/:operation_id` (`diagnostics:read`)

Each update must contain explicit integer `weight`, `priority`,
`expected_weight`, and `expected_priority` values. Proposal creation records
the current snapshot revision and does not change routing. Only the
administrator control route can approve a pending proposal:

`POST /api/agent-change-proposals/:proposal_id/approve`

Approval records the current runtime epoch and is rejected if the proposal's
base snapshot revision is stale. Apply rechecks approval, runtime epoch,
snapshot revision, and expected route-entry values. The route update, durable
proposal-to-operation binding, and operation record commit in one transaction.
The operation then advances through `db_committed`, `snapshot_published`, and
`completed`. A proposal already bound to an operation is replayed or
recovered, even when a later request uses a different idempotency key; a key
already used for a different request is rejected.
