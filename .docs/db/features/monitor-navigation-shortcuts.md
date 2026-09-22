---
id: feature.monitor-navigation-shortcuts
type: feature
name: Monitor cross-page navigation shortcuts
---
# Monitor Cross-Page Navigation Shortcuts

## ID Explanation

`feature.monitor-navigation-shortcuts` owns the user-visible links and
context-preserving navigation that connect group model maintenance, the
schedule center, and request logs. It stops at navigation, row targeting, and
the decision to show or hide a maintenance link; it does not own scheduling
rules, group changes, request-log filtering, or log retention.

## Purpose

Operators often reach a related maintenance or scheduling task while
inspecting a model or a request. This feature keeps the model, source group,
and route identity that the operator was already viewing, so the next page
opens at the relevant place instead of requiring a second search.

## User-Visible Contract

### From group model maintenance to the schedule center

- Each model row can open the schedule center with its external model context
  and originating group preserved.
- When the row has a known route entry, the schedule center highlights that
  exact entry and brings it into view.
- When only the model identity is available, the schedule center selects the
  first matching entry in the originating group after the schedule details
  load.
- Refreshing the schedule page and using browser back or forward preserves the
  navigation context. If the requested row no longer exists, the selection is
  cleared quietly while the model context remains available.

### From request logs to group maintenance

- In a request-log list, the group name remains the control for narrowing the
  log list to that group. A separate maintenance link opens the group's
  maintenance page.
- The request-log detail view exposes the same maintenance link for a group
  only after the group identity has been confirmed to exist.
- Deleted groups remain readable as historical log references, but do not
  expose a maintenance link. A missing or unresolved group likewise has no
  maintenance link.

### Narrow screens and keyboard use

- The maintenance action remains a touch-sized control on narrow screens.
- Keyboard users can see focus on the navigation control while moving between
  the log, group, and schedule views.

## Acceptance Workflows

### Target a schedule row from model maintenance

- **Role and purpose:** An administrator wants to inspect the route entry for a
  model just edited in a group.
- **Entry and action:** Open the group's model maintenance view and activate
  the model row's schedule action.
- **Expected result:** The schedule center opens with the same model and group
  context, highlights the matching entry, and scrolls it into view. Refresh and
  browser history restore the same target.
- **Failure signal:** If the target entry is gone, no stale row is selected;
  the model context remains and the page does not show an error for the stale
  navigation request.

### Open group maintenance from a request log

- **Role and purpose:** An administrator wants to change the group associated
  with a visible request or inspect its configuration.
- **Entry and action:** Open request logs, then use the separate maintenance
  action beside a resolved group in the list or detail view.
- **Expected result:** The group maintenance page opens for that group, while
  the existing group-name control continues to filter logs when the operator
  chooses the filter action instead.
- **Failure signal:** Historical, deleted, or unresolved groups remain visible
  as log data but do not offer a link to a page that cannot be opened.

## Boundaries

- This feature owns cross-page context, target-row highlighting, and
  maintenance-link availability.
- The schedule center owns route selection, weight and priority editing,
  probing, and recovery actions after navigation completes.
- Group maintenance owns group and model changes; request logs own filtering,
  detail inspection, and historical display.
- The feature does not change routing eligibility, log data, group lifecycle,
  permissions, or persistence contracts.
