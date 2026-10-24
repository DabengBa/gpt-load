---
id: feature.dispatch-reasoning-policy
type: feature
name: Dispatch Center reasoning policy
route: /schedule
related: [feature.model-test-alias]
---
# Dispatch Center Reasoning Policy

## ID Explanation

`feature.dispatch-reasoning-policy` owns administrator control of reasoning
intensity for individual model routes in Dispatch Center. It covers the level
choices shown in the editor, saving, clearing, and the resulting effective
value. Provider transport fields and persistence belong to the technical
reasoning-policy owner.

## Purpose

An administrator can set or clear reasoning intensity for one candidate route
at a time, and can see the value that will actually take effect before the next
request runs.

## User-Visible Contract

- Administrators enter Dispatch Center at `/schedule`, select an external
  model, and edit its route details. Reasoning is configurable only as an
  entry override; there is no group-level default.
- The effective order is entry override, client request, then provider
  default. Clearing an entry override lets that request use the client or
  provider value. An unset value is distinct from an explicit `none` setting.
- An entry override belongs to the actual saved route, so routes sharing an
  upstream model can have different choices. Requests using a
  [model test alias](feature.model-test-alias) select that same owning entry.
- Details show the configured value, effective level, and policy source for
  each candidate route. Every candidate offers the same level list; which
  levels a provider actually honors is a provider fact, reported by a failed
  request rather than pre-validated per model.
- Before saving, the editor validates each changed item's requested level
  against the supported level list. A validation failure saves none of the
  changes. A concurrent configuration change requires refreshing the current
  configuration before resubmission; failed saves retain the draft.
- Reasoning, priority, weight, and concurrency edits form one draft. Reloading
  or refreshing the current model, or immediately enabling or disabling a
  candidate, does not discard that draft. Switching models, going back, or
  leaving Dispatch Center offers save and continue, discard and continue, or
  stay. A failed save keeps the current model and draft visible.
- Existing group-level defaults are no longer used or automatically
  transferred. Administrators must configure desired values as item overrides.

## Acceptance Workflows

### Save and clear an item override

- **Role and purpose:** An administrator wants `high` reasoning for one
  candidate route.
- **Entry and action:** Open that candidate's route details in Dispatch Center,
  select `high`, and save. Then clear the item override and save again.
- **Expected result:** The selected item's configured value and effective
  source are shown after saving. Clearing it restores the client/provider
  default path without changing other candidates.
- **Maintenance and failure signal:** An invalid level value is rejected and
  cannot be saved. No partial change should persist. A level the provider
  refuses is visible as that route's failed request, not as a save error.

## Boundaries

This feature controls categorical reasoning intensity per candidate. It does
not change model aliases, route weights, priorities, credentials, provider
pricing, or provider-side capability reporting. An item override is never
silently replaced by another level. Without an item override, client reasoning
semantics remain unchanged; independent parameter-override rules remain a
separate configuration mechanism.
