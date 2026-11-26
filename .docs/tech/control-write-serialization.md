---
description: "Recovery-barrier write invariants for the management control plane: admission ordering, replay readability, bootstrap deadlock avoidance, and credential runtime-state serialization."
kind: technical
topic: control-write-serialization
id: INV-001
status: active
paths:
  - internal/control/**
  - internal/app/**
enforcement: fresh-audit
code:
  paths:
    - internal/control/service.go
    - internal/control/operation_recovery.go
    - internal/control/bootstrap.go
    - internal/control/credential_mutations.go
---
# Control-Write Serialization Invariants

## Invariants

- WHEN committed operation recovery still has pending work, the system SHALL NOT admit any new committed side effect until the recovery barrier releases; every admission edge MUST take the barrier.
- WHEN a completed operation is replayed while recovery is pending, the system SHALL keep the replay readable; the barrier MUST only gate new-operation admission, not replay reads.
- WHEN the bootstrap sequence writes initial state, it MUST NOT wait on the recovery barrier it would deadlock behind; bootstrap writes are idempotent repairs that the drain coordinates.
- Completed-operation retention cleanup MUST NOT be blocked by the recovery barrier; cleanup only touches rows whose upstream side effect completed and whose result is past the retention cutoff.
- Credential runtime state changes (health, cooldown, blacklist, restore) MUST be serialized through `MutationCoordinator`/`doCredentialMutations`, not `writeMu` alone, because data-plane failure recording does not take `writeMu`.
- The recovery drain itself implements the barrier and MUST NOT route through it.

## Boundary

These invariants are cross-boundary ordering guarantees between the control
plane, operation recovery, bootstrap, and credential state. Per-lock-site call
taxonomies are excluded; code and tests are the source of truth.
