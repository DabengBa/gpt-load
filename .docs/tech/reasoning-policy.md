---
description: "Persist group and route-entry reasoning policies, write the selected effort to provider requests, and leave final acceptance to the target Provider."
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
string from the unified effort set sets it. `snapshot_revision` is required.
Revision conflicts return 409, and all affected groups are written in one
transaction before the new snapshot is published.

The detail resource includes full-group `reasoning_entries` so the UI can show
the effective policy preview for entries outside the currently selected
external model. The resource projection strictly accepts the reasoning
configured/effective/source contract and the unified effort set; it does not
read model capability fields.

The center validates effort labels and tri-state update semantics locally. It
does not infer, substitute, or map one effort to another based on a model
catalogue. Unknown and unlisted models remain editable and saveable under the
center contract. The target Provider is the final authority for whether a
specific effort is accepted.

The former `reasoning_effort_overrides` map is removed from parsing, runtime
resolution, and the Advanced Settings editor. Old persisted values have no
runtime effect; there is no migration or compatibility reader.

## Core Implementation

`internal/reasoning/config.go:ResolveEffort` resolves entry, group, client, then
provider default. Entry and group values are centrally owned and validated
against the unified effort set; client values retain their original semantics
when no central value exists. An empty resolved value leaves the Provider
default in control.

Gateway request preparation first applies general parameter overrides, then
applies an effective central reasoning policy. Consequently an explicit central
policy wins when both mechanisms write the reasoning field. Without a central
policy, reasoning resolution does not rewrite the body; independent parameter
overrides still operate normally. The gateway sends the selected effort to the
Provider path without a pre-dispatch model capability gate.

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

- A malformed or unknown center effort is rejected by the center projection or
  PATCH validation. The UI only submits values from the unified effort set.
- A concurrent writer invalidates the supplied revision. Failed validation or
  a revision conflict must not publish a partially updated policy.
- Client-owned effort is not subjected to central-policy normalization merely
  because the request passes through Gateway.
- `internal/control/model_route_schedule_test.go` covers whole-group policy
  persistence, hidden models, same-batch exceptions, clearing, and atomicity.
- `internal/control/reasoning_policy_persisted_wire_integration_test.go`
  covers persistence through the runtime snapshot to the observed provider
  request. Dialect, reasoning, and Gateway tests cover field selection and
  precedence.
- `web/scripts/verify-group-settings-effort.mjs` verifies strict projection,
  the unified effort set, and reasoning values without capability fields.
- `web/e2e/schedule-editing.spec.ts` verifies all effort values are editable
  for unknown models, full-group previews, saving/clearing, and narrow-screen
  interaction with mocked APIs. These tests do not establish acceptance by a
  live Provider.

Provider rejection remains a Provider response boundary. GPT-Load does not
rewrite a rejected effort into another effort.
