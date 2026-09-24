---
description: "Persist route-entry reasoning overrides and apply the selected effort to provider requests."
kind: technical
topic: model-routing
relations:
  related:
    - db/features/dispatch-reasoning-policy.md
    - tech/model-test-alias.md
code:
  paths:
    - internal/control/model_route_schedule.go
    - internal/reasoning
    - internal/state/runtime_settings.go
    - internal/state/snapshot.go
    - internal/state/loader/loader.go
    - internal/gateway/handler.go
    - internal/dialect/request_reasoning.go
    - web/src/frontends/classic/features/monitor/SchedulePanelDetail.vue
---
# Dispatch Reasoning Policy

## Responsibility

The schedule control resource owns route-entry reasoning choices. Runtime
preparation resolves the selected route's effective policy and writes the
appropriate protocol field before dispatch. Product semantics are defined by
[Dispatch Center Reasoning Policy](../db/features/dispatch-reasoning-policy.md).

## Architecture And Constraints

Reasoning overrides persist as `reasoning_effort` in the group's model JSON.
Entry identity is `group_id + entry_id`, not a model-name lookup. The state
loader carries item overrides into the same runtime snapshot used for routing.

The schedule PATCH accepts entry `updates` with `group_id`, `entry_id`, and
`reasoning_effort`. An absent field is unchanged; `null` clears the item
override; a supported string sets it. `snapshot_revision` is required. Revision
conflicts return 409, and all affected groups are written in one transaction
before the new snapshot is published. Group-level reasoning defaults and their
PATCH fields are not part of the contract.

Validation evaluates each changed item against its own provider/model
capability. A failure aborts the whole transaction. The detail resource
projects reasoning configuration and capability on each candidate entry.

The former `reasoning_effort_overrides` map and group-level
`reasoning_effort_default` setting are not part of runtime resolution. There is
no automatic data conversion; existing stored group defaults must be removed
and desired values configured as item overrides.

## Core Implementation

`internal/reasoning/config.go:ResolveEffort` resolves entry, client, then
provider default. Entry values are centrally owned and validated; client values
retain their original semantics when no item override exists. An empty resolved
value leaves the provider default in control.

`internal/reasoning/capabilities.go` projects Bifrost's model capability records
and provider capability rules into known/supported state, levels, and a reason.
`ValidateEffort` requires the requested label exactly; `none` additionally
requires the ability to disable reasoning. Unsupported item values fail before
provider dispatch rather than being silently downgraded.

Gateway request preparation first applies general parameter overrides, then
applies a configured item reasoning override. Consequently an explicit item
override wins when both mechanisms write the reasoning field. Without an item
override, reasoning resolution does not rewrite the body; independent
parameter-override rules still operate normally. Capability validation uses
the actual upstream model of the selected route.

For supported generation operations, the dialect layer writes the client
protocol field before any provider conversion:

| Protocol | Effort field |
| --- | --- |
| OpenAI Chat | `reasoning_effort` |
| OpenAI Responses | `reasoning.effort` |
| Anthropic | `output_config.effort` |
| Gemini | `generationConfig.thinkingConfig.thinkingLevel` |

Native and converted, streaming and non-streaming requests use this preparation
path. Provider adapters serialize the resulting policy for their upstream
protocol. `internal/execution/bifrost/reasoning_policy_wire_test.go` verifies
observed request bodies across these paths; a successful control PATCH alone
is not evidence of the final provider field.

## Failure Boundaries And Verification

- An unsupported item override cannot be saved. Clearing an override restores
  client/provider behavior and does not modify other entries.
- A concurrent writer invalidates the supplied revision. Failed validation or
a revision conflict must not publish a partially updated policy.
- Client-owned effort is not subjected to central-policy normalization merely
  because the request passes through Gateway.
- `internal/control/model_route_schedule_test.go` covers per-entry capability
  validation, clearing, and atomicity.
- `internal/control/reasoning_policy_persisted_wire_integration_test.go`
  covers persistence through the runtime snapshot to the observed provider
  request. Dialect, reasoning, and Gateway tests cover field selection and
  precedence.
- `web/e2e/schedule-editing.spec.ts` verifies item capability feedback,
  saving/clearing, and narrow-screen interaction with mocked APIs. These tests
  do not establish acceptance by a live provider.
