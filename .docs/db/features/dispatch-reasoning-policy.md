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
intensity for a provider group and its individual model routes in Dispatch
Center. It covers inheritance, saving, and clearing a policy. Provider
transport fields and persistence belong to the technical reasoning-policy owner.

## Purpose

An administrator can set a common reasoning intensity for a provider group and
make exceptions for individual model routes, while seeing the configured value,
effective value, and policy source before saving.

## User-Visible Contract

- Administrators enter Dispatch Center at `/schedule`, select an external
  model, and edit its route details. Group defaults and entry overrides are
  maintained here; Provider Advanced Settings no longer offers the old
  per-model reasoning override editor.
- The effective order is entry override, group default, client request, then
  provider default. Clearing an entry override restores inheritance; clearing
  the group default lets entries without overrides follow the client or
  provider. An inherited value is distinct from an explicit `none` setting.
- An entry override belongs to the actual saved route, so routes sharing an
  upstream model can have different choices. Requests using a
  [model test alias](feature.model-test-alias) select that same owning entry.
- The editor offers the same unified effort values for every route. Unknown or
  unlisted models do not disable the controls, invalidate a draft, or block a
  save. The selected value is sent to the target Provider, which is responsible
  for accepting or rejecting it.
- Details show the configured value, effective value, and policy source. A
  group default applies across the group, including models outside the
  currently selected external-model detail; the draft preview shows those
  effective values.
- A save applies the group and entry changes together. A validation failure
  saves none of them. A concurrent configuration change requires refreshing the
  current configuration before resubmission; failed saves retain the draft.
- Existing values from the removed per-model Advanced Settings configuration
  do not remain active and are not automatically transferred. Administrators
  must configure the intended defaults and exceptions in Dispatch Center.

## Acceptance Workflows

### Save a group default with a model exception

- **Role and purpose:** An administrator wants `low` reasoning for a group
  containing models with different or unknown Provider capabilities.
- **Entry and action:** Open the group's route details in Dispatch Center,
  select any unified effort value as the group default, and optionally give a
  route an explicit override before saving.
- **Expected result:** All unified effort values remain selectable, including
  for unknown or unlisted models. Saving succeeds when the center contract is
  valid; the Provider receives responsibility for final acceptance. Reopening
  shows the group default and saved exception with their effective sources.
- **Maintenance and failure signal:** Clearing an exception restores the group
  or client/provider inheritance and remains editable. No partial change should
  persist when the revision is stale or the center value is not in the unified
  effort set.

## Boundaries

This feature controls categorical reasoning intensity. It does not change
model aliases, route weights, priorities, credentials, or provider pricing.
GPT-Load writes the selected central policy value; it does not silently map one
effort to another. Provider-specific capability and acceptance decisions remain
outside this UI contract. Without a central policy, this feature leaves client
reasoning semantics unchanged; independent parameter-override rules remain a
separate configuration mechanism.

The product semantics in this document are intentionally independent of any
particular Provider capability catalogue. See the technical owner for protocol
writing and verification facts.

## Traceability

Implementation and validation facts are owned by
[Dispatch Reasoning Policy technical notes](../../tech/reasoning-policy.md).
