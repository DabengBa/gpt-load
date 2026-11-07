# Group Site Information

- **Change:** Group maintenance can connect to a provider's New API site and
  show account balance and effective API Key group ratio. Administrators can
  review aggregated site announcements and retry failed source refreshes.
- **Why:** Operators previously had to visit each provider site separately to
  check account status and announcements, and could not see the effective site
  group ratio beside its GPT-Load group.
- **Scope:** Group create/edit and collection, plus the `/announcements` page.
  Site information is informational and does not change local billing or
  routing. Access tokens are encrypted, never echoed, and not copied with a
  group.
- **Current owners:** [product behavior](../db/features/group-site-information.md),
  [Groups](../db/pages/groups.md), [Announcements](../db/pages/announcements.md),
  and [technical contract](../tech/group-site-information.md).
- **Evidence and delivery:** `internal/control/site_info*_test.go`,
  `web/e2e/astryx-announcements-*.spec.ts`,
  `web/e2e/astryx-site-info-visual.spec.ts`; delivered from branch
  `feat/site-info` (PR pending).
