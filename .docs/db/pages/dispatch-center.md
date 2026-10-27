---
id: page.dispatch-center
type: page
name: Dispatch Center
route: /schedule
related: [feature.dispatch-concurrency, feature.dispatch-reasoning-policy, feature.monitor-navigation-shortcuts]
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

## Reading Candidate Rows

- Saved scheduling eligibility is separate from entry health: a healthy entry
  can still be ineligible or in a standby priority tier because of another
  scheduling condition. The displayed reason explains the saved snapshot; an
  unknown reason is shown as such.
- Smaller priority numbers are considered first. Configured share is normalized
  by weight within each priority tier. Disabled groups and entries are excluded;
  entries in cooldown or blacklisted still count. This share is not the
  probability of selection for the next request.
- A muted leading marker calls out rows disabled by either the group or the
  entry. The entry switch reflects only its own setting and can remain on while
  its group is disabled.
- Unsaved priority and weight edits show an unsaved share preview; they do not
  change actual scheduling or the saved eligibility snapshot. Enabling or
  disabling an entry takes effect immediately; if the action fails, its prior
  state is restored. Recovery also takes effect immediately, while other edits
  are saved together.
- Request counts and success rates describe the group over the past 24 hours,
  not the individual upstream model. With no group requests, success rate is
  shown as — rather than 0%.
