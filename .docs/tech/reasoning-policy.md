---
description: "Persist group and route-entry reasoning policies, validate whole-group capabilities atomically, and apply the selected effort to provider requests."
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

The schedule control resource owns group defaults and route-entry reasoning
choices. Runtime preparation resolves the selected route's effective policy
and writes the appropriate protocol field before dispatch. Product semantics
are defined by [Dispatch Center Reasoning Policy](../db/features/dispatch-reasoning-policy.md).

## Architecture And Constraints

Group defaults persist as `overrides.reasoning_effort_default`; entry overrides
persist as `reasoning_effort` in the group's model JSON. Entry identity is
`group_id + entry_id`, not a model-name lookup. The state loader carries both
values into the same runtime snapshot used for routing.

The schedule PATCH accepts `group_updates` with `group_id` and
`reasoning_effort_default`, and entry `updates` with `group_id`, `entry_id`, and
`reasoning_effort`. An absent field is unchanged; `null` clears a value; a
supported string sets it. `snapshot_revision` is required. Revision conflicts
return 409, and all affected groups are written in one transaction before the
new snapshot is published.

Validation evaluates the final merged group settings and model entries,
including changes in the same PATCH. Every model in each affected group is
checked against its effective entry-or-group policy, including entries outside
the currently selected external model. A failure aborts the whole transaction.
The detail resource includes full-group `reasoning_entries` so the UI can
preview the same cross-model constraint.

The former `reasoning_effort_overrides` map is removed from parsing, runtime
resolution, and the Advanced Settings editor. Old persisted values have no
runtime effect; there is no migration or compatibility reader.

## Core Implementation

`internal/reasoning/config.go:ResolveEffort` resolves entry, group, client, then
provider default. Entry and group values are centrally owned and validated;
client values retain their original semantics when no central value exists.
An empty resolved value leaves the provider default in control.

`internal/reasoning/capabilities.go` projects Bifrost's model capability records
and provider capability rules into known/supported state, levels, and a reason.
`ValidateEffort` requires the requested label exactly; `none` additionally
requires the ability to disable reasoning. Unsupported central values fail
before provider dispatch rather than being silently downgraded.

Gateway request preparation first applies general parameter overrides, then
applies an effective central reasoning policy. Consequently an explicit central
policy wins when both mechanisms write the reasoning field. Without a central
policy, reasoning resolution does not rewrite the body; independent parameter
overrides still operate normally. Capability validation uses the actual
upstream model of the selected route.

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

- An unsupported group default cannot be saved unless effective entry overrides
  make every model in the group valid. Deleting an exception is validated by
  the same rule as creating one.
- A concurrent writer invalidates the supplied revision. Failed validation or
  a revision conflict must not publish a partially updated policy.
- Client-owned effort is not subjected to central-policy normalization merely
  because the request passes through Gateway.
- `internal/control/model_route_schedule_test.go` covers whole-group
  validation, hidden models, same-batch exceptions, clearing, and atomicity.
- `internal/control/reasoning_policy_persisted_wire_integration_test.go`
  covers persistence through the runtime snapshot to the observed provider
  request. Dialect, reasoning, and Gateway tests cover field selection and
  precedence.
- `web/e2e/schedule-editing.spec.ts` verifies capability feedback, hidden-model
  effects, saving/clearing, and narrow-screen interaction with mocked APIs.
  These tests do not establish acceptance by a live provider.
