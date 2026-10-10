---
description: On-demand management pages, intent preloading, and group query freshness without replacing editor drafts.
kind: technical
topic: navigation-performance
code:
  paths:
    - web/src/frontends/astryx/app/router.tsx
    - web/src/frontends/astryx/features/groups/GroupsView.tsx
    - web/src/shared/control/invalidation.ts
    - web/src/shared/control/resources/groups.ts
    - web/src/shared/controllers/auth-session.ts
---

# Navigation loading and cache policy

## Page loading and navigation intent

The router loads the eight business views (home, group collection, group detail, import, logs, settings, monitor, and schedule) through dynamic imports. Login remains an eager view. The Go server continues to serve the embedded frontend; direct page URLs use the same router and authentication checks.

The router's intent preload policy warms target page code when a navigation link is hovered or focused, without changing the current location. Authentication and administrator route guards still apply. A route loading failure displays the localized failure message and retry button; retry resets the error boundary and reloads the browser to request the page again.

In the group collection, pointer entry or keyboard focus on a group detail link additionally prefetches that group's summary for a validated administrator session. This data prefetch excludes settings, models, and credentials. Repeated intent and the eventual click share fresh cached data or the in-flight summary request. Anonymous and access-key sessions cannot trigger this summary prefetch.

## Query freshness and editor snapshots

Group collection, options, and summaries use a 30-second stale window and may refresh on focus or reconnect. Options always refetch on mount; collection filter changes retain previous data. Group settings and models retain the manual snapshot policy (infinite stale time, no focus or reconnect refresh). Successful saves write their own exact snapshots and invalidate dependent reads. Model saves stale pricing/catalog without immediate refetch; settings saves refresh those active consumers. Group create and delete additionally invalidate health and the broad home/model reads.

Mutation invalidation uses exact keys for singleton reads and prefixes for filtered collections or per-group families. Settings and model mutation helpers cover home base because home inventory derives from group state. Logout cancels protected queries, removes protected query data, removes the auth session query, and clears mutation cache entries. The API client's default unauthorized handler routes a 401 through session clear; callers may explicitly disable that handler.

## Audit

| Mutation              | Affected reads                                                                                                    | Policy                                                                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Create/delete group   | options, collection, health, home, models, prices, route schedule                                                 | exact options/health plus collection and dependent prefixes                                                                                                     |
| Replace group models  | collection pages, exact group summary/options/home base, catalog/prices, route schedule                           | `cacheGroupModels` writes the exact saved models and patches summary/options; `invalidateGroupModelDependents` stales reads, leaving other group editors intact |
| Update group settings | collection pages, exact group summary/models/options/home base, group credentials, catalog/prices, route schedule | `cacheGroupSettings` writes the saved settings; `invalidateGroupSettingsDependents` refreshes active dependents, leaving other group editors intact             |
| Import credentials    | group summary/credentials, collection, home, route schedule                                                       | exact summary/health plus dependent prefixes                                                                                                                    |

## Regression coverage

`web/e2e/astryx-navigation-performance.spec.ts` covers cold login, navigation intent, direct page loading, and retry after a failed page import. `web/e2e/astryx-group-prefetch.spec.ts` covers administrator hover/focus, summary request reuse, and the anonymous/access-key boundaries. These browser tests use Vite and mocked API responses; they do not measure production latency or verify the Go embedded frontend's CSP behavior.

`web/scripts/group-query-freshness.test.ts` and `web/scripts/mutation-cache.test.ts` use real `QueryClient` and `QueryObserver` instances with mocked fetch to verify read freshness, editor snapshots, mutation invalidation, and logout/401 cancellation and cache cleanup. `web/e2e/astryx-group-freshness.spec.ts` checks focus refresh and preservation of unsaved editor changes.
