---
description: New API site access, refresh, announcement aggregation, security and compatibility contracts for group management
kind: technical
topic: group-site-information
relations:
  related:
    - tech/navigation-performance.md
    - tech/legacy-db-conversion.md
code:
  paths:
    - internal/control/site_info.go
    - internal/control/site_info_http.go
    - internal/control/site_info_sync.go
    - internal/storage/models/site_info.go
    - internal/storage/migrations/0029_group_site_info.go
    - internal/storage/migrations/0030_announcement_source_status.go
    - web/src/frontends/astryx/features/groups/settings/GroupSettingsSiteAccessSection.tsx
    - web/src/frontends/astryx/features/announcements/AnnouncementsView.tsx
    - web/src/shared/control/resources/announcements.ts
---

# Group Site Information

Product behavior is owned by [Group Site Information](../db/features/group-site-information.md),
[Groups](../db/pages/groups.md), and
[Announcements](../db/pages/announcements.md).

## Integration Contract

The administrator configures a provider website, New API user ID and access
token per GPT-Load group. The provider website's scheme and host define the
site root. Changing a configured root requires explicit confirmation before
the browser submits the access token to that root. URL validation rejects
unsupported schemes, credentials in the URL and invalid new ports while
preserving safe reads and unrelated edits for legacy values.

Requests authenticate with `Authorization` and `New-Api-User`. Credentials are
encrypted at rest and never projected back to the client. Cross-origin
redirects are not followed; errors persisted for display are sanitized and do
not include request credentials or response bodies.

The synchronizer fetches site status, announcement and notice data, the
authenticated user/group information, and the authenticated token list needed
to match the configured API Key without placing a full API key in a URL. The
effective ratio is scoped to the authenticated user and the matched token's
effective group; failed authenticated group lookup does not fall back to an
anonymous pricing view. The provider account balance is distinct from the
remaining quota of an individual token.

## Refresh and Retry

Configured sites are refreshed on a 60-minute schedule. The coordinator bounds
work to eight concurrent sites and a 45-second per-group deadline. A retry is
available through `POST /api/groups/:group_id/site-info/retry`; it marks the
configured source pending and wakes the synchronizer. The route is an
administrator mutation and is audited. It does not perform a synchronous
provider request or promise that the refresh has completed.

Failures retain the last successful balance, ratio and announcements where
possible, while source state records status, fetch time and a sanitized error.
Announcement entries and source states are read from the same database
snapshot so the page cannot combine different refresh generations.

## Persistence and Compatibility

Site credentials and snapshots live in `group_site_configs`; aggregated
announcements live in `site_announcements`. Migrations 0029 and 0030 create
these tables without adding columns to the legacy `groups` table. The
independent-table decision preserves the offline legacy converter's exact
schema contract; see [ADR-0004](../adr/0004-group-site-info-independent-tables.md)
and [Legacy Database Conversion](legacy-db-conversion.md).

New collection and settings fields are opt-in for the updated frontend so
older clients retain their existing response shape. Announcement source
`last_error` is optional at the projection boundary; clients treat its absence
as an empty diagnostic. Access tokens are not included in copy-group payloads
or import recovery storage.

## Verification

Control behavior is covered by `internal/control/site_info*_test.go`, migration
contracts by `internal/storage/migrations/0029_group_site_info_test.go` and
`0030_announcement_source_status_test.go`, and frontend interaction by
`web/e2e/astryx-announcements-*.spec.ts` and
`web/e2e/astryx-site-info-visual.spec.ts`. A local mocked provider verifies the
browser workflow but does not establish compatibility with every live New API
deployment.
