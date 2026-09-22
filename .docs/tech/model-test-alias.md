---
description: "The persistence, validation, route-index, and control-plane contract for stable six-character aliases that target one group/model route entry."
kind: technical
topic: model-routing
relations:
  related:
    - db/features/model-test-alias.md
code:
  paths:
    - internal/state/external_model.go
    - internal/state/loader/loader.go
    - internal/state/snapshot.go
    - internal/control/group_write.go
    - internal/control/group_models.go
    - internal/control/group_create.go
    - internal/control/group_copy.go
    - internal/control/group_idempotency.go
    - internal/control/group_options.go
    - internal/gateway/models.go
    - web/src/frontends/classic/app/resources/groups.ts
    - web/src/frontends/classic/features/groups/models/GroupModelsTab.vue
    - web/src/frontends/classic/features/models/model-draft.ts
---
# Model Test Alias Routing

## Scope

- **Owner:** persisted group model entries, test-alias allocation and
  validation, compiled route indexes, group model control resources, and the
  classic model editor projection.
- **Authoritative for:** alias namespace rules, lifecycle behavior, single-entry
  route resolution, and the separation between client and upstream model
  attribution.
- **Excludes:** ordinary client-alias routing semantics, provider pricing
  definitions, credential authorization, and request-log retention.

## Responsibility

The model-routing boundary gives each persisted group/model entry a stable,
server-owned short name. It carries that name from persisted group JSON into
the runtime snapshot, indexes it as an exact route key, and exposes it to
operators without allowing a model-list save request to edit the allocation.

## Architecture And Constraints

`test_alias` is stored inside each group's model JSON; it does not require a
separate database table or migration. A test alias is exactly six lowercase
letters or digits. The global namespace rejects duplicate test aliases and
rejects collisions with every upstream model ID and ordinary external model
name, including names from disabled groups.

At startup, `BackfillTestAliases` reads groups in ascending ID order, validates
existing route entries, assigns missing aliases, and persists the changed JSON
in one transaction. Existing aliases are never regenerated. The same allocator
is used by group creation, model updates, idempotent creation, and group copy;
copying clears source aliases before allocation so the copy cannot target the
source entry by accident. Idempotency input excludes the server-owned alias,
while a replay returns the original durable operation result.

## Core Implementation

- `internal/state/external_model.go` defines the six-character contract,
  cryptographically random allocation, global collision validation, and the
  shared ordinary external-name helper.
- `internal/state/loader/loader.go:BackfillTestAliases` performs startup
  backfill while preserving unknown model JSON fields and aborting the
  transaction on invalid or conflicting persisted values.
- `internal/state/snapshot.go:appendExecutionTargets` adds the ordinary
  external name and the test alias as separate keys that point to the same
  `RouteTarget`. The target retains the owning group, upstream model, route
  entry identity, weight, priority, and resolved provider target.
- `internal/control/group_models.go` preserves an existing alias when a model
  entry is retained, returns it from the read resource, and allocates aliases
  for new entries. Group creation, copying, and idempotent writes use the same
  allocation boundary; group options include test aliases as visible model
  choices.
- `internal/gateway/models.go` includes test-alias keys in model discovery and
  applies the caller's protocol, model, and group filters to the indexed target
  just like other model names.
- `web/src/frontends/classic/app/resources/groups.ts` requires the read value
  to match the six-character format. `model-draft.ts` and
  `GroupModelsTab.vue` carry it as read-only display state while excluding it
  from the normalized write payload.
- Gateway request attribution keeps the requested name as the client model and
  the selected route's model as the upstream model; durable request-log and
  usage tests verify that the short alias does not replace the upstream pricing
  identity.

## Related Product Semantics And Binding Points

- Product behavior is owned by [Provider-Specific Model Test Aliases](../db/features/model-test-alias.md).
- Code binding: `internal/state/external_model.go:ValidateTestAliases`,
  `GenerateTestAlias`
- Code binding: `internal/state/loader/loader.go:BackfillTestAliases`
- Code binding: `internal/state/snapshot.go:appendExecutionTargets`
- Code binding: `internal/control/group_models.go:UpdateGroupModels`
- Code binding: `internal/gateway/models.go:collectVisibleModelIDs`

## Operations, Calculation, And Failure Boundaries

- **Allocation:** collect all upstream IDs, ordinary external names, and
  existing test aliases, then generate a six-character value from the
  lowercase alphanumeric alphabet until an unused value is found. The random
  allocator stops after its bounded collision-retry budget and fails the
  enclosing write instead of silently reusing a name.
- **Persistence:** startup backfill and control writes persist the generated
  value before publishing a snapshot. A failed transaction or failed snapshot
  compilation does not publish a partially allocated route.
- **Validation:** malformed, duplicate, or standard-name-conflicting persisted
  aliases prevent startup/config compilation. The classic client also rejects
  malformed read responses rather than displaying an untrusted route name.
- **Lifecycle:** retaining the same model entry preserves its value; deleting
  or copying an entry allocates independently. The alias is not a client-set
  secret and does not bypass access-key, protocol, or group filters.
