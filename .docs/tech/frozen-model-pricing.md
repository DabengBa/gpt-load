---
description: "Frozen model pricing, v7 receipt arithmetic, and incompatible deployment prerequisites after removal of configurable price multipliers."
kind: technical
topic: frozen-model-pricing
code:
  paths:
    - internal/pricing/receipt.go
    - internal/pricing/quote.go
    - internal/storage/migration.go
---
# Frozen Model Pricing

## Calculation Boundary

Group and access-key configurable price multipliers are removed from configuration,
APIs, runtime pricing, and UI controls. There is no default-one compatibility path.
New requests use model reference prices frozen at request time, with the existing
pricing mode and matched tier. These amounts are estimates, not provider bills.

Component arithmetic is unchanged: apply the component price and coefficient,
round each priced component to the nearest 0.000000001 USD using the existing
rules, and sum the rounded components. Cache coefficients such as 8/5 remain;
they are not configurable group or access-key multipliers. Unpriced portions are
excluded, and partial usage or incomplete pricing remains explicitly identified.
HTTP, streaming, and WebSocket accounting use the same calculated amount for
request logs, usage statistics, and cost quotas.

## Receipt Boundary

New pricing receipts use schema v7 only, with frozen component pricing and one
`total_nano_usd`. There is no base-versus-adjusted total, `price_multipliers`, or
`base_total_nano_usd`. Older schemas, including v5/v6, are not interpreted by the
application. Historical receipts are not repriced or converted automatically.

## Deployment Prerequisites

This is an incompatible deployment boundary, not an in-place upgrade procedure.
An existing database must not be assumed usable with this revision. Before any
deployment, the operator must separately approve and complete all three items:

1. **Schema and migration ledger:** independently prepare and verify a schema
   matching the current registry and an active applied ledger that is its exact prefix.
   Explicitly retired IDs in `removedMigrationIDs`, including
   `0009_price_multipliers`, remain in the ledger but are excluded from the active
   chain. Unknown IDs and missing active migrations are rejected before mutation.
   Remaining migration IDs keep their identity; the gap from
   0008 to 0010 does not authorize rewriting historical ledger entries. External
   archival, retention scope, and schema preparation require a separate decision.
2. **Old receipts:** decide how incompatible historical receipts are archived,
   retained, and accessed outside the v7-only application before cutover. The
   application supplies no old-receipt reader, migration, or automatic cleanup.
3. **Idempotency history:** decide the handling of old create/update replay
   records and response payloads containing retired fields. Verify the approved
   replay policy before enabling writes; old payloads must not be assumed valid
   under the new contract.

These prerequisites are not authorization to delete data. This document provides
no cleanup commands, old-receipt migration, or runtime compatibility layer.
Development changes do not modify a live database, commit, push, or deploy.
Existing cumulative cost amounts and configured limits are preserved by default;
there is no default limit reset, historical counter reset, or historical
repricing. Any separately proposed destructive preparation requires explicit
approval, archival and recovery decisions, and validation outside this task.
