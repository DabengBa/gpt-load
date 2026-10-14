---
description: "Current Astryx frontend layout: app shell, feature modules, shared layers, and the intentional cross-feature consumption rules."
kind: technical
topic: frontend-structure
relations:
  related:
    - tech/navigation-performance.md
code:
  paths:
    - web/src/frontends/astryx/app/router.tsx
    - web/src/frontends/astryx/components/DetailPanel.tsx
    - web/src/frontends/astryx/entry.css
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

Historical migration inventories live in
[the migration plan](astryx-migration-plan.md) and are records of past phases,
not a map of current paths. For current behavior details of a domain, read the
feature directory. [Navigation loading and cache policy](navigation-performance.md)
owns route loading, intent preloading, and query freshness; its implementation
starts in `app/router.tsx`.

## Detail panels

`components/DetailPanel.tsx` owns the shared right-side modal shell. It uses
the Astryx `Dialog` primitive selected in
[ADR-0002](../adr/0002-astryx-detail-layer-primitive.md), with a full-height
layout, a scrollable content region, and an optional footer. Keep the panel
mounted and drive its `isOpen` prop so the dialog can restore invoker focus.
`dismissible={false}` blocks Escape, backdrop dismissal, and the close button.

The default `compact` width is `min(92vw, 520px)`. Log details, access-key
editing, and upstream-price editing select `width="wide"`, which uses
`min(92vw, 640px)`. At viewport widths of 520px or less, both sizes become
full-width. Header, content, and footer share a 24px desktop horizontal gutter
and an 18px narrow-screen gutter. The overrides in `entry.css` are scoped to
`[data-detail-panel]`; standard dialogs retain Astryx's own layout rules.
The slide-in animation is disabled when reduced motion is requested.

The layout must explicitly establish its own vertical padding because the
zero-padding dialog surface otherwise passes zero outer padding to its slots.
`DialogHeader` uses negative title and close-button margins for visual
alignment; the header must leave enough space above and below these elements
to keep the title, subtitle, and close button fully inside the panel. Content
scrolling must not move or obscure the header or footer. The programmatic
content focus anchor does not draw an outline around the whole scrollport;
interactive controls retain their keyboard focus indicators.

`contextNotice` renders a sticky, opaque context band at the top of the
scrollable content, directly below the header. Upstream-price editing uses
this slot for affected client-model/group counts and the save scope, so that
context stays visible while the form scrolls. Access-key editing keeps its
impact summary in the form and its save-blocker feedback in the footer.

The panel's geometry and standard-dialog style isolation are covered by
`web/e2e/astryx-detail-panel-geometry.spec.ts` and
`web/e2e/astryx-standard-dialog-scope.spec.ts`. Feature interaction coverage
lives in the access-key, import, group-model-discovery, request-log, and
unified-schedule-price E2E suites.
