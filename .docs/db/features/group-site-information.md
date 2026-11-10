---
id: feature.group-site-information
type: feature
name: Group site information
---
# Group Site Information

## ID Explanation

`feature.group-site-information` owns the administrator experience for
connecting a group to its provider's New API site and inspecting that site's
account information. Request protocols, persistence, refresh scheduling and
security boundaries belong to [Group Site Information](../../tech/group-site-information.md).

## Purpose

Administrators can see the provider account balance and the effective API Key
group ratio alongside the GPT-Load group, and can review announcements from
configured sites without leaving the management interface.

## User-Visible Contract

- Group settings and the new-group import flow accept a site user ID and access
  token. The site root is derived from the provider website; changing the root
  for an already configured site requires confirmation before saving.
- The access token is stored encrypted and is never returned for display. A
  configured token is represented by its saved state; clearing it is an
  explicit action. Copying a group does not copy site access credentials.
- The group collection shows the latest known account balance and API Key
  group ratio. Missing or unavailable values are shown as unavailable rather
  than inferred from a different account or group.
- Site information refreshes automatically on a 60-minute cycle. A failed
  refresh retains the last successful information where available and exposes
  a failure reason and a retry action.
- The [Announcements page](../pages/announcements.md) aggregates site
  announcements, distinguishes source status, orders entries by event time,
  and supports filtering and search. A filter with no matches has a clear
  action to reset it.

## Acceptance Workflows

### Connect a provider site

- **Role and purpose:** An administrator wants to see information for the
  provider account used by a group.
- **Entry and action:** Edit a group or create one, enter its provider website,
  site user ID and access token, then save.
- **Expected result:** The site root is derived from the website, a changed
  configured root is confirmed, and fetched account information appears in the
  group collection and settings.
- **Failure signal:** The token is echoed, copied to a cloned group, or sent to
  a different root without confirmation.

### Recover a failed site refresh

- **Role and purpose:** An administrator needs to distinguish stale information
  from a current successful refresh.
- **Entry and action:** Open group settings or the Announcements page and use
  the retry action for a failed source.
- **Expected result:** The source is queued for synchronization; its status and
  reason update while prior announcements and last successful values remain
  available when possible.
- **Failure signal:** A failure silently removes prior announcements or a retry
  is reported as successful before synchronization completes.

## Boundaries

- The displayed balance is the provider account balance, not the remaining
  quota of one GPT-Load credential; the ratio describes the effective provider
  group for the configured account.
- These values are informational and do not change GPT-Load billing, routing,
  quota or health decisions.
- Provider-specific protocols and error redaction are implementation details
  owned by the technical document.
