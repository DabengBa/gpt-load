---
description: "Current Astryx frontend layout: app shell, feature modules, shared layers, and the intentional cross-feature consumption rules."
kind: technical
topic: frontend-structure
code:
  paths:
    - web/src/frontends/astryx/app/router.tsx
    - web/src/frontends/astryx/features
    - web/src/shared/domain
    - web/src/shared/controllers
    - web/src/shared/control
---

# Astryx Frontend Structure

`web/src/frontends/astryx` is the only management UI. Its modules group
application infrastructure, business features, and reused components:

- `app/` owns route registration, services, authentication and shell,
  i18n wiring, and application-wide hooks. `router.tsx` lazily loads route
  views; business-specific hooks live beside their features.
- `features/` owns business views, components, and hooks in `access-keys`,
  `groups`, `home`, `import`, `logs`, `models`, `monitor`, and `settings`.
  The monitor feature also contains the schedule view. A feature is not
  necessarily a route: model editing and access-key management are composed
  into other features. There is no cross-feature component barrel.
- `components/` contains reused application components, including copy
  controls, channel selection, detail panels, header-rule editing, and
  section chrome. They can depend on shared business types and app services,
  but should not reach into a feature's private implementation.

The shared layers are framework-free by contract:

- `shared/domain/` holds pure models, codecs, and price/form logic with no
  React imports. Consumers import specific modules; there is no barrel.
- `shared/controllers/` holds framework-free controllers driving editor drafts
  and collection behavior.
- `shared/control/` owns API resource types, endpoint functions, query
  options, and cache invalidation. `shared/http/` owns the HTTP client and
  error types. Shared routing codecs, catalogs, assets, and utilities remain
  in their respective directories.

Intentional cross-feature consumption is allowed where a domain genuinely
reuses another domain's widget: `home` imports `CostLimitWindowTime` from
`features/access-keys`, settings composes access-key management, and groups,
import, and monitor consume model capabilities from `features/models`.
`ModelAliasEditor` belongs to models; `CopyChip` and `ChannelPresetPicker`
belong to components. Log details, filters, and route presentation belong to
logs. Keep these dependencies explicit through specific module imports;
do not add broad barrels or move business widgets into generic UI solely to
eliminate cross-feature imports. `shared/` may never import from the frontend.

Historical migration inventories live in `.docs/tech/astryx-migration-plan.md`
and are records of past phases, not a map of current paths. For current
behavior details of a domain, read the feature directory; for routing and
preloading, read `app/router.tsx`.
