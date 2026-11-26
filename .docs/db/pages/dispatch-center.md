---
id: page.dispatch-center
type: page
name: Dispatch Center
route: /schedule
related: [feature.dispatch-concurrency, feature.dispatch-reasoning-policy, feature.monitor-navigation-shortcuts, feature.realistic-probe]
---
# Dispatch Center

## ID Explanation

`page.dispatch-center` describes the Dispatch Center page and how administrators
interpret its candidate rows. Individual concurrency and reasoning controls are
owned by their respective feature documents.

## Page Purpose

Administrators select an external model to inspect and maintain its dispatch
candidates. Access-key users see a read-only view instead of the administrator
editor.

## Page Structure

The page has a model selector and a details area. For the selected model, the
administrator's candidate rows show group and upstream model, saved scheduling
eligibility, entry health, group statistics, configuration controls, same-tier
configured share, and actions. Advanced row controls include
[concurrency](feature.dispatch-concurrency) and
[reasoning policy](feature.dispatch-reasoning-policy). The page can also be
opened from [model maintenance](feature.monitor-navigation-shortcuts).
The selected generation model can open
[Actual Request Testing](feature.realistic-probe) with its protocol and external
model preserved.

## Shared Upstream Pricing

The candidate price entry opens upstream pricing. Administrators can edit or
reset base prices, tiered prices and existing Fast-mode prices. These prices
are shared by channel and upstream model; before saving, the page shows the
related upstreams and affected groups. Directory synchronization affects the
global model catalog and automatic pricing, and automatic pricing remains
available.

Access-key users see only the models, protocols and reference prices within
their authorization. Reference prices use USD per million tokens for cost
estimation and are not a supplier invoice.

## Ownership Boundaries

The standalone `/models` page is removed. Group details own each group's model
rows and aliases; the Dispatch Center owns cross-group scheduling and shared
pricing.

## Reading Candidate Rows

- Saved scheduling eligibility is separate from entry health: a healthy entry
  can still be ineligible because of another scheduling condition. A schedulable
  entry can have lower priority than another schedulable entry; this does not
  mean it was selected for a request. The displayed reason explains the saved
  snapshot; an unknown reason is shown as such.
- Smaller priority numbers are considered first. Configured share is normalized
  by weight within each priority tier. Disabled groups and entries are excluded;
  entries in cooldown or blacklisted still count. This share is not the
  probability of selection for the next request.
- Rows are grouped by saved priority and ordered within each group by saved,
  unrounded configured share, highest first. Equal shares retain the existing
  group and entry key order. Editing a draft does not move rows; a successful
  save reorders them according to the refreshed saved configuration.
- Text-and-dot soft badges distinguish saved eligibility, disabled groups and
  entries, and other reasons without using color to indicate priority. Disabled
  rows mute non-interactive model and share text, not the controls. A muted
  leading marker also calls out rows disabled by either the group or the entry.
  The entry switch reflects only its own setting and can remain on while its
  group is disabled.
- Unsaved priority and weight edits show an unsaved share preview; they do not
  change actual scheduling or the saved eligibility snapshot. Enabling or
  disabling an entry is submitted immediately without saving the other edits;
  rapid switch changes retain the latest intention. Recovery also takes effect
  immediately, while other edits are saved together.
- If another configuration change conflicts with a switch, the switch can be
  retried automatically only when its target is still unchanged. Otherwise a
  conflict is shown instead of overwriting the changed target. An existing
  draft is retained but is not automatically saved over external changes.
- If a write's result is unknown, the page warns the administrator to verify
  the configuration and blocks further switches and draft saves for that
  model. A refresh alone does not prove an unknown write did not commit later.
  If a write succeeded but its subsequent refresh failed, further switches and
  draft saves are blocked until the saved configuration is refreshed.
- Request counts and success rates describe the group over the past 24 hours,
  not the individual upstream model. With no group requests, success rate is
  shown as — rather than 0%.
