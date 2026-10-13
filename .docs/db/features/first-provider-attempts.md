---
id: feature.first-provider-attempts
type: feature
name: Same-provider maximum attempts
---
# Same-Provider Maximum Attempts

## ID Explanation

`feature.first-provider-attempts` owns the administrator-visible setting that
controls how many times one request may try its first provider before the
gateway moves to another candidate. It owns the settings entry, value rules,
save feedback, and the user-visible request boundary; retry orchestration,
credential refresh, and transport details belong to the technical owner.

## Purpose

An administrator can allow a request to make more than one attempt with the
provider selected first, while keeping a single limit over all upstream
attempts for that request.

## User-Visible Contract

- The setting appears in the administrator runtime settings under retry and
  credential health as **Same-provider maximum attempts**.
- The first execution counts as attempt one. The default is 1, which means the
  first provider is not repeated before another candidate may be selected.
- The value must be a whole number of at least 1. Values below 1 are rejected
  and the invalid setting is not saved.
- Same-provider attempts share the request's global maximum upstream attempt
  budget. Increasing this setting cannot make a request exceed that total;
  switching providers and repeating the first provider draw from the same
  budget.
- After an administrator saves a valid value, the settings view reports that
  the settings were saved and shows the effective value. When saving fails, the
  entered value remains available for correction and the view reports that the
  runtime settings could not be updated.
- The setting is a global runtime policy. It does not select a provider,
  change provider credentials, or bypass the existing health and availability
  rules.

## Acceptance Workflows

### Set the same-provider maximum

- **Role and purpose:** An administrator wants the first provider to receive
  another try before the request moves to another provider.
- **Entry and action:** Open runtime settings, find Retry and credential health,
  enable the override for Same-provider maximum attempts, enter a whole number
  of at least 1, and save the settings.
- **Expected result:** The valid value is saved, the effective value is shown,
  and a request can use up to that many attempts with its first provider while
  remaining within the request's global attempt budget.
- **Failure signal:** Entering 0, a negative number, or a non-integer shows the
  lower-bound validation error and does not publish a new setting.

## Boundaries

- This feature owns the administrator entry and the meaning of the count.
- The count includes the first attempt and shares the request-wide total
  attempt budget; it does not promise that every permitted attempt will run if
  health or availability rules stop the provider earlier.
- HTTP failures, credential refresh, buffered streaming, and provider rotation
  remain implementation behavior documented by
  [Provider feedback](../../tech/provider-feedback.md).
- The feature does not define provider APIs, credentials, model selection,
  request-log fields, or transport retry mechanics.
