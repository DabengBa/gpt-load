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
intensity for individual model routes in Dispatch Center. It covers capability
feedback, saving, and clearing an item override. Provider transport fields and
persistence belong to the technical reasoning-policy owner.

## Purpose

An administrator can set or clear reasoning intensity for one candidate route
at a time, while seeing whether that model supports the choice before saving.

## User-Visible Contract

- Administrators enter Dispatch Center at `/schedule`, select an external
  model, and edit its route details. Reasoning is configurable only as an
  entry override; there is no group-level default.
- The effective order is entry override, client request, then provider
  default. Clearing an entry override lets that request use the client or
  provider value. An unset value is distinct from an explicit `none` setting,
  which is available only when the model supports disabling reasoning.
- An entry override belongs to the actual saved route, so routes sharing an
  upstream model can have different choices. Requests using a
  [model test alias](feature.model-test-alias) select that same owning entry.
- Details show capability status and reason, supported levels, effective level,
  and policy source for each candidate route.
- Before saving, the editor validates each changed item's requested level
  against that item's model capability. A validation failure saves none of the
  changes. A concurrent configuration change requires refreshing the current
  configuration before resubmission; failed saves retain the draft.
- Existing group-level defaults are no longer used or automatically
  transferred. Administrators must configure desired values as item overrides.

## Acceptance Workflows

### Save and clear an item override

- **Role and purpose:** An administrator wants `high` reasoning for one
  supported candidate route.
- **Entry and action:** Open that candidate's route details in Dispatch Center,
  select `high`, and save. Then clear the item override and save again.
- **Expected result:** The selected item's configured value and effective
  source are shown after saving. Clearing it restores the client/provider
  default path without changing other candidates.
- **Maintenance and failure signal:** An unsupported level is not offered as a
  supported choice and cannot be saved. No partial change should persist.

## Boundaries

This feature controls categorical reasoning intensity per candidate. It does
not change model aliases, route weights, priorities, credentials, or provider
pricing. An item override is never silently substituted when the chosen
provider/model does not support it. Without an item override, client reasoning
semantics remain unchanged; independent parameter-override rules remain a
separate configuration mechanism.
