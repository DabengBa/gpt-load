---
description: "Runtime semantics, affinity-priority boundary, and request lifecycle of the per-dispatch-row concurrency limit."
kind: technical
topic: schedule-concurrency
relations:
  related:
    - db/features/dispatch-concurrency.md
    - tech/affinity-binding.md
code:
  paths:
    - internal/state/capacity.go
    - internal/scheduler/scheduler.go
    - internal/gateway/capacity.go
    - internal/gateway/attempts.go
    - internal/gateway/websocket_turn.go
    - internal/control/model_route_schedule.go
---
# Dispatch Row Concurrency Limit

## Owner And Scope

This document owns the implementation facts of the dispatch-row concurrency
limit: configuration plumbing, in-process capacity records, atomic admission,
candidate skipping, release, and the capacity error. The product-visible
configuration and scheduling behavior are owned by [Dispatch Row Concurrency
Limit](../db/features/dispatch-concurrency.md); affinity bindings, identity
generations, and the health bypass remain owned by
[Affinity Binding](affinity-binding.md).

## Capacity Key And Configuration

Capacity is recorded per `RouteEntryKey` (`GroupID + EntryID`).
`concurrency_limit` lives in the model entry JSON, the runtime `ModelConfig`,
and `RouteTarget`; a missing value normalizes to 0, and 0 means unlimited. The
accepted range is 0 to 2147483647. In the Dispatch Center PATCH an absent field
keeps the stored value, an explicit 0 removes the limit, and `null` is rejected.

Capacity records belong to a single service process. Snapshot publication and
capacity state updates are coordinated under the same in-process lock, and an
existing reservation is never interrupted by lowering the limit, switching to
0, or deleting the entry. Deleting an entry blocks new admission; an in-flight
record survives as a non-admittable tombstone until the last reservation
releases, so recreating the same key immediately cannot reset occupancy.
Admission validates both the snapshot the request belongs to and the entry's
existence: a request holding a stale snapshot, and a deleted or disabled entry,
return `CapacitySnapshotChanged` or `CapacityUnavailable` instead of a false
capacity-exhausted result.

## Admission Order

Candidates first pass the existing protocol, permission, model, enabled,
weight, priority, health, and affinity eligibility filters. An ordinary
candidate requests capacity before its real attempt; a capacity shortage only
skips the current `RouteEntryKey` and does not write the scheduler's
credential/upstream tried pair, consume execution retry budget, or produce a
provider attempt. When every candidate is blocked by capacity, the gateway
returns `503 schedule_capacity_exhausted`.

The scheduler's `NextCandidate()` only selects; it records no cost. The
executor calls `MarkAttempt` only after admission succeeds, and a capacity
rejection calls `SkipEntry` to exclude that entry and continue. `Next()` remains
only as a select-and-mark wrapper; the data plane uses the separated form,
because otherwise a single capacity rejection would consume that
credential/upstream pair in advance.

Atomic admission happens after the local payload, credential decryption, and
proxy preparation, which prevents over-admission between preparation and the
actual send. Only a successful admission marks a real attempt. A capacity
rejection runs no health feedback; when a real upstream failure already exists,
its evidence is preserved and the capacity reason is appended as a separate
request-level outcome.

## Affinity And Fixed Identity

A valid affinity request reuses the binding identity and eligibility checks of
[Affinity Binding](affinity-binding.md). It may claim an overflow slot only when
the target actually resolved for this request matches the binding exactly
(group, credential, identity generation, route entry, upstream model) and the
current entry is available; it still counts toward real occupancy. Exceeding
the limit never produces a health bypass or a binding takeover by itself.

A WebSocket `binding` or `requiredRef` fixed identity is a continuity
constraint, not automatically a valid affinity. A fixed identity without an
affinity exemption returns the local capacity error when its entry is full and
is not moved to another identity; only an ordinary turn without a fixed
identity may choose another candidate after a capacity block. A fixed identity
with a valid affinity may exceed the limit.

## Lifecycle

HTTP and streaming requests each hold one reservation during the actual
Forward/ForwardStream call; return, failure, cancellation, timeout, and a
pre-dispatch termination all release it idempotently. A reservation covers one
Forward call and its internal execution; retries on different candidates are
admitted separately. WebSocket admission happens inside each `executeTurn` and
covers only that active turn, so the handshake and an idle connection waiting
for the next turn hold no capacity. Release uses the record carried by the
reservation and never re-reads the entry's current configuration.

## Verification

Capacity unit tests cover atomic races, affinity overflow, exact entry skipping,
hot updates and deletion, HTTP and streaming requests, WebSocket fixed
identities, and the replay lifecycle. Control-plane tests cover configuration
round trips, invalid input, draft URL state, conflict preservation, and field
projection. The concurrency-limit UI cases run in
`web/e2e/astryx-schedule.spec.ts`; the models mock projection contract failure
recorded there earlier is fixed and its assertions match the current
projection.
