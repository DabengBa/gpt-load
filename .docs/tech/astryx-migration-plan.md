---
description: "Plan and risk assessment for migrating the management UI from Vue 3 to a React 19 + StyleX + Astryx stack."
kind: technical
topic: frontend-migration
code:
  paths:
    - web
    - internal/webui
---
# Astryx + StyleX Migration Plan

## Responsibility

This document owns the evaluation and delivery plan for replacing the
`web/src/frontends/classic` Vue 3 frontend with a React 19 frontend built on
Meta's Astryx design system (`@astryxdesign/core`, currently `0.6.3`, MIT) and
StyleX (`@stylexjs/stylex`, `^0.19.1`). It records corrected facts, the full
stack mapping, component coverage, the migration strategy, and the verification
plan. It is a plan, not an approval: execution starts only after the decision
below is accepted.

## Decision Frame

Astryx is React-only: `@astryxdesign/core@0.6.3` declares peer dependencies on
`react >= 19`, `react-dom >= 19`, and `@stylexjs/stylex ^0.19.0`. Adopting it is
therefore a full framework rewrite, not a styling-library swap. The scope under
migration is roughly 87k lines across 345 files in `web/src`, including 172 Vue
SFCs and 13 feature domains.

StyleX without Astryx is a separate, much smaller option: `stylex.attrs` is an
official first-class API for `class`-based frameworks, and `stylex.create` kept
in `.ts` modules compiles through the official `@stylexjs/unplugin` Vite
integration. Community plugins (`vue-macros` `defineStyleX`, `@stylex-extend`)
are only needed to author styles inside SFCs. That path delivers atomic CSS but
none of the design system, and is out of scope here.

## Corrections To Prior Assessment

- Version pinned to `0.6.3` (npm latest), not `0.6.0` shown on the marketing
  page. License verified as MIT.
- Dual-frontend coexistence is not free. `internal/webui/server.go` caches a
  single `dist/index.html` at startup and serves it for every page route, with
  dedicated handlers only for `theme-bootstrap.js` and `favicon.svg`. Route-level
  grey release between `classic` and the new frontend requires a small Go
  change (per-prefix index selection), plus equivalent bootstrap/favicon
  handling. CSP already allows `style-src-attr 'unsafe-inline'`, so React inline
  styles and external StyleX CSS are compatible without CSP changes.
- Component coverage is better than first estimated (see below), and Astryx
  utilities replace several hand-rolled services, not just components.
- Playwright specs are only partially portable: flows transfer, but selectors
  bound to `App*` DOM structure must be rewritten. `web/scripts/verify-*` need a
  per-script DOM/artifact coupling audit.
- The `.docs/db` semantic database binds Doc IDs to code paths; the rewrite will
  drift them and must be reconciled via `pnpm --dir web run docs:check` /
  `docs:build`.
- `web/package.json` security `overrides` (brace-expansion, minimatch, nanoid,
  postcss) must be re-evaluated against the React dependency tree.
- `astryx init` writes AGENTS.md/CLAUDE.md cheat sheets; these must be merged
  with the repository's existing agent rules, not overwrite them.

## Target Stack Mapping

| Current | Target | Migration nature |
| --- | --- | --- |
| Vue 3.5 SFC + Composition API | React 19 + JSX/TSX | Full rewrite |
| vue-router 5 | TanStack Router (preferred: type-safe, same family as query) or React Router 7 | Rewrite; keep `internal/webui/page_routes.json` as the single route manifest |
| @tanstack/vue-query | @tanstack/react-query | Smooth; query keys and fetchers port |
| vue-i18n + TS locale modules | Astryx `InternationalizationProvider`/`useTranslator` (preferred, fewer deps) or react-i18next | Catalog content ports mechanically; pluralization syntax needs audit |
| reka-ui + ~50 `App*` components | `@astryxdesign/core` 150+ components, `xstyle` for overrides, `swizzle` for domain forks | Component layer replaced wholesale |
| Tailwind CSS 4 + custom tokens.css | StyleX `stylex.create` via `@stylexjs/unplugin`; optional Tailwind v4 bridge during transition | Styling paradigm switch; declare cascade layer order explicitly while both systems coexist |
| @lucide/vue | lucide-react or Astryx `Icon` | Mechanical |
| provide/inject DI | React Context | Concept ports, code rewrites |
| composables | hooks | ref/reactive model becomes useState/useMemo; largest per-file cost |
| Hand-rolled services | Astryx utilities | Net deletions: `useAnnounce` replaces `RouteAnnouncer`, `useClipboard` replaces `use-clipboard-copy`, `Theme`/`useTheme`/`Media Theme` replace the theme controller, `useImperativeDialog`/`AlertDialog` replace confirm plumbing |
| Vite 8 + @vitejs/plugin-vue | Vite 8 + @vitejs/plugin-react + `@stylexjs/unplugin` (+ `@astryxdesign/build` if source builds needed) | Retained; multi-entry config |
| vue-tsc / eslint-plugin-vue | tsc + eslint-plugin-react-hooks + `@stylexjs/eslint-plugin` | Toolchain port |
| node:test contracts, Playwright | Unchanged harnesses; React component tests optional (Vitest) | Mostly retained |
| Go embed serving | Unchanged except dual-index support during coexistence | Small additive change |

Survives unchanged: Go backend and embed pipeline (`Makefile` `_web-build` seam,
dev proxy targets), `@shared/http` client and `@shared/preferences` (plain TS,
reused as-is), locale message content, contract tests, pnpm/Node toolchain.

## Component Coverage

Verified against the published component index. Strong coverage: `Table` plus 15
table hooks (sortable, filtering, pagination, sticky columns, tree/grouped rows,
column resize/settings, selection state), `Date Range Input`, `Typeahead`,
`Toast`/`useToast`, `Command Palette`, `Side Nav`/`App Shell`, `Timestamp`,
`Markdown`, `Code Block`, `Skeleton`, `Empty State`, `Pagination`.

Confirmed gaps:

- No charting components. `components/charts/TrendChart.vue` must be hand-ported
  (it is bespoke SVG) or replaced by a React chart library; decide per usage.
- No standalone drawer; use `Dialog` or `Bottom Sheet` for `AppDrawer` cases.
- Domain components (`CredentialHealthBar`, `CopyChip`, `MobileRecordCard`,
  typed-confirmation, async-surface states) have no direct equivalents and are
  re-implemented on Astryx primitives, or `swizzle`d where a near-match exists.
- `@internationalized/date` is already a dependency and is the same foundation
  Astryx date inputs use; keep it.

## Migration Strategy

Recommended: parallel-frontend track, reusing the existing `frontends/` layout.

1. Scaffold `web/src/frontends/astryx` as a second Vite entry (React 19,
   `@astryxdesign/core` + a chosen theme, `@stylexjs/unplugin`). Run
   `astryx init` and merge its agent docs with existing rules. Register the
   Astryx MCP server for agent workflows ("full配套" scope).
2. Extend `internal/webui` to serve a second index for an `/v2` route prefix (or
   equivalent prefix split), keeping classic as default. Reuse
   `page_routes.json` as the manifest for both routers.
3. Port the shared layer first: `@shared/http`, auth session, locale catalogs,
   query client config. Then migrate feature domains in order of rising
   coupling: settings/preferences → home → models/model-prices →
   access-keys → monitor → logs → groups → import.
4. Keep classic fully functional until the last domain ships; cut over by
   flipping the default index, then delete `frontends/classic` and the
   coexistence Go code in a follow-up.
5. During coexistence, declare cascade layer order explicitly; unlayered styles
   override Astryx's `@layer astryx-base` regardless of specificity.

Fallback: big-bang rewrite is simpler operationally but forfeits the ability to
diff behavior per route and is only acceptable if the parallel track is blocked.

## Risks

- Astryx is Beta (`0.6.x`) with high upstream velocity; budget for codemod-
  assisted upgrades (`astryx upgrade`) and pin exact versions.
- Young Meta OSS project; roadmap and maintenance commitment are unproven.
- React 19 floor locks the frontend to the React ecosystem.
- Chart gap and domain-component rebuilds are the main functional unknowns;
  spike them early (one monitor chart, one groups table).
- Doc ID drift in `.docs/db` must be reconciled at each phase boundary.

## Verification

Per phase and at cutover: `pnpm --dir web type-check` (tsc), `lint`,
`format`, Playwright e2e (rewritten selectors), `node:test` contract suites,
`verify:*` scripts after DOM-coupling audit, `pnpm --dir web run docs:check` +
`docs:build`, and `make build` for the embedded Go artifact.

## Open Questions

- Theme selection (`theme-neutral` default vs custom theme via `astryx theme`).
- Router choice (TanStack Router vs React Router 7) and i18n library choice
  (Astryx translator vs react-i18next) — pick during scaffolding spike.
- Whether to keep the Tailwind bridge long-term or converge on pure StyleX.
