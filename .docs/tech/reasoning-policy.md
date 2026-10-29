---
description: "Persist route-entry reasoning overrides and apply the selected effort to provider requests."
kind: technical
topic: model-routing
relations:
  related:
    - db/features/dispatch-reasoning-policy.md
    - tech/model-test-alias.md
    - tech/reasoning-replay.md
code:
  paths:
    - internal/control/model_route_schedule.go
    - internal/reasoning
    - internal/state/runtime_settings.go
    - internal/state/snapshot.go
    - internal/state/loader/loader.go
    - internal/gateway/handler.go
    - internal/dialect/request_reasoning.go
    - web/src/frontends/astryx/features/monitor/SchedulePanelDetail.tsx
---
# Dispatch Reasoning Policy

## Responsibility

The schedule control resource owns route-entry reasoning choices. Runtime
preparation resolves the selected route's effective policy and writes the
appropriate protocol field before dispatch. Product semantics are defined by
[Dispatch Center Reasoning Policy](../db/features/dispatch-reasoning-policy.md).
Conversation-history conversion is owned by
[Reasoning History Replay](reasoning-replay.md).

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

Validation checks each changed item against the central effort vocabulary;
the first invalid entry aborts the whole transaction. The detail resource
projects the configured value, the effective value, and its source for each
candidate entry. It does not project per-model capability or supported levels.

The former `reasoning_effort_overrides` map and group-level
`reasoning_effort_default` setting are not part of runtime resolution. There is
no automatic data conversion; existing stored group defaults must be removed
and desired values configured as item overrides.

## Core Implementation

`internal/reasoning/config.go:ResolveEffort` resolves entry, client, then
provider default. Entry values are centrally owned and validated against the
effort vocabulary (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`);
client values retain their original semantics when no item override exists and
are never normalized. An empty resolved value leaves the provider default in
control.

There is no capability registry. Which levels a saved entry actually supports
is a Provider fact, so an entry value that the vocabulary accepts is persisted
and sent; a provider that rejects it fails that attempt and the ordinary retry
and health policy applies. The editor therefore offers the same vocabulary for
every entry instead of a per-model allowlist.

Gateway request preparation first applies general parameter overrides, then
applies a configured item reasoning override. Consequently an explicit item
override wins when both mechanisms write the reasoning field. Without an item
override, reasoning resolution does not rewrite the body; independent
parameter-override rules still operate normally.

`internal/gateway/attempts.go` writes the field only for generation operations
whose client protocol defines one (`SupportsReasoningEffortOverride`), and only
when the winning source is the entry override — a client-owned effort is left
untouched. The dialect layer sets the client-protocol field with `sjson`
before any provider conversion, creating the parent objects if needed:

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

The frontend stores reasoning edits in the same typed `ScheduleDrafts` map as
priority, weight, and concurrency, serialized through `schedule_draft` in the
current URL. Missing `reasoning_effort` means unchanged; explicit `null` means
clear the override. Refresh and enable/disable mutations retain the local
draft while adopting the latest observed snapshot revision. Enable/disable
and draft saves for the same model context are serialized in the editor. A
queued write advances only after its predecessor publishes the next revision
and a detail refresh observes that exact revision; an intervening change,
failed refresh, or context change stops subsequent queued writes. A successful
PATCH remains committed even if its subsequent refresh fails. Revision
conflicts trigger a refresh and preserve unsaved drafts. The navigation
blocker permits same-model URL updates and protects model changes, Back, and
page exits with save/discard/stay choices. Saving before navigation does not
replace the outgoing draft URL: only a successful PATCH releases the original
blocked navigation. The separate price editor retains its own guard.

The scheduling UI shows six compact columns from 768px upward. A 180px model
index is visible at 1440px and above; smaller screens use a searchable model
popover. Below 768px candidates become compact records with 44px controls.
Price editing, concurrency, and breaker details are disclosed in each row's
More popover; recovery remains directly available for unhealthy candidates.

- An item override outside the supported vocabulary cannot be saved. Clearing
  an override restores client/provider behavior and does not modify other
  entries.
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
- `web/e2e/astryx-schedule.spec.ts` verifies item capability feedback,
  saving/clearing, and narrow-screen interaction with mocked APIs. These tests
  do not establish acceptance by a live provider.
