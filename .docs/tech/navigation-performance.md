---
description: Current navigation and group query freshness and mutation cache policy.
kind: technical
topic: navigation-performance
code:
  paths:
    - web/src/shared/control/invalidation.ts
    - web/src/shared/control/resources/groups.ts
    - web/src/shared/controllers/auth-session.ts
---

# Navigation performance cache policy

Group collection, options, and summaries use a 30-second stale window and may refresh on focus or reconnect. Options always refetch on mount; collection filter changes retain previous data. Group settings and models retain the manual snapshot policy (infinite stale time, no focus or reconnect refresh). Successful saves write their own exact snapshots and invalidate dependent reads. Model saves stale pricing/catalog without immediate refetch; settings saves refresh those active consumers. Group create and delete additionally invalidate health and the broad home/model reads.

Mutation invalidation uses exact keys for singleton reads and prefixes for filtered collections or per-group families. Settings and model mutation helpers cover home base because home inventory derives from group state. Logout cancels protected queries, removes protected query data, removes the auth session query, and clears mutation cache entries. The API client's default unauthorized handler routes a 401 through session clear; callers may explicitly disable that handler.

## Audit

| Mutation              | Affected reads                                                                                                    | Policy                                                                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Create/delete group   | options, collection, health, home, models, prices, route schedule                                                 | exact options/health plus collection and dependent prefixes                                                                                                     |
| Replace group models  | collection pages, exact group summary/options/home base, catalog/prices, route schedule                           | `cacheGroupModels` writes the exact saved models and patches summary/options; `invalidateGroupModelDependents` stales reads, leaving other group editors intact |
| Update group settings | collection pages, exact group summary/models/options/home base, group credentials, catalog/prices, route schedule | `cacheGroupSettings` writes the saved settings; `invalidateGroupSettingsDependents` refreshes active dependents, leaving other group editors intact             |
| Import credentials    | group summary/credentials, collection, home, route schedule                                                       | exact summary/health plus dependent prefixes                                                                                                                    |

The focused regression lane uses real `QueryClient` instances, real `QueryObserver` consumers, mocked fetch, and cleans each client in `finally`. Logout and 401 cleanup assertions are verification of an already-correct policy (real in-flight abort-aware request is cancelled and mutations are cleared); the group settings helper home coverage supplied the red proof.
