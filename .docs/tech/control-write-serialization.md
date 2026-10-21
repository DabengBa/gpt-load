---
description: "The writeMu lock-site taxonomy for the management control plane: which writes take the operation-recovery barrier, which are exempt, and the rationale that keeps each exemption honest."
kind: technical
topic: control-write-serialization
code:
  paths:
    - internal/control/service.go
    - internal/control/operation_recovery.go
    - internal/control/operation_error.go
    - internal/control/bootstrap.go
    - internal/control/credential_mutations.go
    - internal/control/idempotency_operation.go
    - internal/control/proposals.go
    - internal/control/model_price.go
    - internal/control/group_models.go
---

# Control-Plane Write Serialization

## Scope

- **Owner:** `internal/control` management-plane mutations under
  `Service.writeMu`.
- **Authoritative for:** which write paths enforce the durable
  operation-recovery barrier, which are exempt, and why each exemption is
  structurally safe.
- **Excludes:** data-plane request-log writes (request-log worker owns those),
  and read-side `writeMu.RLock` sites, which take the read lock only to
  snapshot persisted + runtime state consistently and never mutate.

## The contract

Every `writeMu.Lock()` site belongs to exactly one class:

1. **Ordinary management mutation** — takes the lock, calls
   `enforceOperationRecoveryBarrierLocked`, mutates inside
   `withControlTransaction`, publishes runtime, and on post-commit publish
   failure runs inline `recoverCommittedRuntime`. Sites: `writeGroupConfig`,
   `writeConfig`, `writeCredentialConfig` (all in `service.go`),
   `writePriceConfig` (`model_price.go`), and `GetGroupModels`'s lazy
   backfill (`group_models.go`) which routes through `writeGroupConfig`.
2. **Idempotent durable operation** — `ControlOperation` stage machine
   (`idempotency_operation.go`, proposal Apply in `proposals.go`). Replay
   reads precede the barrier; the barrier gates only *admission* of new
   operations, since replay must stay readable while recovery is pending.
3. **Startup bootstrap** — `EnsureInitialState` (`bootstrap.go`) runs before
   `DrainCommittedOperations` during `app.Start`; sitting behind the barrier
   it drains would deadlock. Its writes are idempotent repairs the drain
   reconciles.
4. **Cleanup of an existing operation** — `CompactCompletedOperations`
   (`operation_recovery.go`). It removes replay payloads whose upstream side
   effect already happened and whose result completed before the retention
   cutoff; blocking it would only postpone reclaiming space. It touches only
   `completed_at_ms IS NOT NULL` rows, which pending recovery never reads.
5. **Durable metadata / runtime-state mutation without committed config** —
   proposal Create/Approve/Revoke (`proposals.go`) write only proposal rows;
   Apply revalidates base revision, runtime epoch and expected values under
   the barrier, so a proposal built on an unrecovered snapshot self-corrects.
   `RestoreGroupCredential` (`credential_mutations.go`) repairs live
   cooldown/blacklist state and commits nothing; its registry/stats mutation
   runs inside `doCredentialMutations` so data-plane failure recording cannot
   interleave.
6. **Recovery mechanism itself** — `DrainCommittedOperations` implements the
   barrier; it must never run it.

## Invariants this preserves

- A new committed side effect never lands while unfinished operation
  recovery is pending (barrier at every admission edge).
- Retention cleanup of already-completed operations is never blocked by the
  barrier (class 4 exempts `CompactCompletedOperations`, which reads only
  `completed_at_ms IS NOT NULL` rows).
- Replay of finished operations stays readable during pending recovery
  (replay checks precede the barrier in class 2).
- Credential runtime-state mutations (health, cooldown, blacklist, restore)
  serialize through `MutationCoordinator`/`doCredentialMutations`, not just
  `writeMu`, because data-plane failure recording does not take `writeMu`.
