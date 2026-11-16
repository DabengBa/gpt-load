---
id: feature.request-log-filtering
type: feature
name: Request log filtering
related: [feature.monitor-navigation-shortcuts]
---
# Request Log Filtering

## ID Explanation

`feature.request-log-filtering` owns the user-visible filtering of request
logs, including historical group and client-model choices. Cross-page links
are owned by [Monitor Cross-Page Navigation Shortcuts](feature.monitor-navigation-shortcuts);
the candidate query contract belongs to [Request Log Facets](../../tech/request-log-facets.md).

## Purpose

Operators need to narrow historical requests without losing groups that have
since been deleted or models that are no longer configured.

## User-Visible Contract

- Administrators can search and select an any-attempt group or client model
  from values observed in request history. Choices reflect the applied time
  range and other applied filters, not unapplied form edits or just the current
  page. Each choice list ignores its own selected value so that alternatives
  remain available.
- A historical group can remain available after deletion. If its name is
  unavailable, its ID is shown; an already selected group or model remains
  visible even when absent from the latest choices.
- Selecting or clearing either choice immediately updates the log results and
  starts pagination over, without applying or discarding other draft edits.
  Other form fields continue to use Apply. Refreshing the list also refreshes
  the historical choices.
- If choices cannot be loaded, the page reports that some filter options are
  unavailable and disables these selectors rather than substituting current
  configuration for history. An invalid applied affinity key must be corrected
  before these choices can be applied.
- Access-key users do not see the group selector; their client-model filter
  remains a text field applied with the form, within their own log scope.

## Acceptance Workflows

### Narrow historical requests without losing a draft

- **Role and purpose:** An administrator investigates requests for a historical
  group and model while preparing another filter.
- **Entry and action:** Open request logs, apply a time range, edit another
  field without applying it, then search for and select a group or model.
- **Expected result:** The list immediately reflects that choice from the
  beginning of pagination, while the other edit remains in the form but is
  not applied. Clearing the choice also takes effect immediately. A selected
  historical value remains visible if it is absent from refreshed choices.
- **Failure signal:** A draft is silently applied or lost, an option is
  inferred from current configuration instead of history, or a choice-load
  failure is presented as a successful empty list.

## Boundaries

Filtering does not change log retention, routing, or group configuration.
The group filter concerns any matching attempt, not only the final route.
