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
Center. It covers inheritance, capability feedback, saving, and clearing a
policy. Provider transport fields and persistence belong to the technical
reasoning-policy owner.

## Purpose

An administrator can set a common reasoning intensity for a provider group and
make exceptions for individual model routes, while seeing which models support
the resulting choice before saving.

## User-Visible Contract

- Administrators enter Dispatch Center at `/schedule`, select an external
  model, and edit its route details. Group defaults and entry overrides are
  maintained here; Provider Advanced Settings no longer offers the old
  per-model reasoning override editor.
- The effective order is entry override, group default, client request, then
  provider default. Clearing an entry override restores inheritance; clearing
  the group default lets entries without overrides follow the client or
  provider. An inherited value is distinct from an explicit `none` setting,
  which is available only when the model supports disabling reasoning.
- An entry override belongs to the actual saved route, so routes sharing an
  upstream model can have different choices. Requests using a
  [model test alias](feature.model-test-alias) select that same owning entry.
- Details show capability status and reason, supported levels, effective level,
  and policy source. A group default applies across the group, including models
  outside the currently selected external-model detail.
- Before saving, the editor evaluates the group default together with all entry
  overrides, shows affected models, and disables saving an unsupported
  combination. Clearing an exception can therefore be rejected if it would
  expose that model to an unsupported group default.
- A save applies the group and entry changes together. A validation failure
  saves none of them. A concurrent configuration change requires refreshing
  the current configuration before resubmission; failed saves retain the draft.
- Existing values from the removed per-model Advanced Settings configuration
  do not remain active and are not automatically transferred. Administrators
  must configure the intended defaults and exceptions in Dispatch Center.

## Acceptance Workflows

### Save a group default with a model exception

- **Role and purpose:** An administrator wants `low` reasoning for a group
  containing `gemini-3.7-flash` and `gemini-3.1-flash-lite-image`.
- **Entry and action:** Open the group's route details in Dispatch Center and
  select `low` as the group default. Inspect affected-model feedback, then give
  the image model an explicit `high` override before saving.
- **Expected result:** The unsupported default alone cannot be saved. With the
  supported image-model exception, saving succeeds; reopening shows the group
  default and the saved exception with their effective sources.
- **Maintenance and failure signal:** Clearing the image-model exception must
  again prevent saving while it would inherit `low`, even when that model is
  outside the current external-model detail. No partial change should persist.

## Boundaries

This feature controls categorical reasoning intensity. It does not change
model aliases, route weights, priorities, credentials, or provider pricing.
A central policy never silently substitutes a different intensity when the
chosen provider/model does not support the requested value. Without a central
policy, this feature leaves client reasoning semantics unchanged; independent
parameter-override rules remain a separate configuration mechanism.
