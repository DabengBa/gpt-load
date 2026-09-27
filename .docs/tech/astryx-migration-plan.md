---
description: "Plan, baseline inventory, coexistence architecture, and risk assessment for migrating the management UI from Vue 3 to React 19 + Astryx + StyleX."
kind: technical
topic: frontend-migration
relations:
  related:
    - adr/0001-collection-read-model-scale.md
    - db/features/monitor-navigation-shortcuts.md
    - db/features/model-test-alias.md
    - db/features/dispatch-reasoning-policy.md
    - db/features/usage-timing-metrics.md
code:
  paths:
    - web/src
    - web/vite.config.ts
    - web/package.json
    - web/e2e
    - web/scripts
    - internal/webui
---
# Astryx + StyleX Migration Plan

**Status:** Phase 1 complete (go decision taken on the seven-gate evidence
recorded in the B13 gate record). Phase 2 domain migrations proceed under the
rules below.

**Baseline:** all counts below were measured on commit `dcabee7a`
(2026-09-25). Package facts come from the npm registry and official Astryx,
StyleX, Vite, TanStack, and React Router docs on the same date. Re-measure
before starting each phase.

## Scope

- **Owner:** the plan for replacing `web/src/frontends/classic` (Vue 3) with a
  React 19 frontend built on Astryx (`@astryxdesign/core`) and StyleX. That
  covers the coexistence architecture in `internal/webui`, the shared-layer
  extraction, the phase order, and the verification gates.
- **Authoritative for:** the target stack, pinned versions, coexistence
  design, Vue→React translation rules, component mapping, and the migration
  risk register.
- **Excludes:** product behavior. Pages, features, and acceptance workflows
  live in `.docs/db/`, and those docs serve as the parity checklists here. Also
  excluded: backend API contracts, and visual redesign decisions beyond what
  adopting Astryx implies.

## Decision Summary

1. **This is a full framework rewrite, not a styling swap.**
   `@astryxdesign/core@0.6.x` requires `react >= 19`, `react-dom >= 19`, and
   `@stylexjs/stylex ^0.19.0` as peers.
2. **Ship Astryx precompiled. Only app-authored styles need the StyleX
   compiler.** Astryx ships `astryx.css`, `reset.css`, and theme CSS; the
   docs say no build plugin is needed for Vite. `@stylexjs/unplugin` is
   required only because this plan writes app styles in StyleX (`xstyle`
   overrides and `stylex.props` on our own DOM).
3. **Coexist on the same URLs.** Keep the URLs unchanged, add a second HTML
   entry, and have Go pick one per route through a cookie plus a per-route
   rollout flag in `page_routes.json`. There is no `/v2` prefix (reasons in
   [Coexistence Architecture](#coexistence-architecture)).
4. **Do a framework-neutral extraction in classic first (Phase 0).** The
   shared layer, e2e selectors, and verify scripts are repointed while Vue is
   still the only runtime. Phase 0 is worth doing even if the migration is
   cancelled.
5. **CSP stays unchanged, which rules out runtime style injection.** Every
   theme must be built with `astryx theme build`. A CSP violation shows up
   only in the Go-served build, never in Vite dev.
6. **Pin exact versions and follow the 7-day rule.** `@astryxdesign/core@0.6.3`
   was published 2026-09-23. Pin `0.6.2` (published 2026-09-15), or wait until
   2026-09-30 to take `0.6.3`. Themes pin `core` exactly, so core and theme
   versions always move together.

### Confirmed choices (2026-09-25)

| Topic | Decision | Consequence in this plan |
| --- | --- | --- |
| Supported browsers | Chromium 125+ (Chrome and Edge) | This is Astryx Tier 1 on Chromium, so anchor-positioned layers work. `build.target: 'chrome125'` and `browserslist` pin the output. Safari and Firefox are best-effort and do not gate releases |
| Visual density | Match classic's compact metrics | The custom theme and default component sizes reproduce classic's type scale, control heights, and row heights; see [Theme and tokens](#theme-and-tokens) |
| Router | TanStack Router, code-based routes | React Router 8 is no longer evaluated |
| React Compiler | On, through the stable Babel preset | `babel-plugin-react-compiler@1.0.0` via `reactCompilerPreset`, scoped to the new frontend. Manual memoization is the exception, not the default |
| App i18n library | `react-intl` | ICU catalogs. A few classic messages must be rewritten into a syntax both runtimes accept; see [Internationalization](#internationalization) |
| Theme base | `neutral` extended by a custom `defineTheme` | Only `@astryxdesign/theme-neutral` is installed. The body font is overridden to the classic system stack, so no webfont is loaded and `font-src 'self'` holds; see [Theme and tokens](#theme-and-tokens) |

Rejected alternative: StyleX on Vue without Astryx. `stylex.attrs` plus
`@stylexjs/unplugin` would compile StyleX in `.ts` modules, but it delivers
atomic CSS without the design system. The classic styles are scoped BEM
rather than utilities (see [Baseline Inventory](#baseline-inventory)), so it
would not remove the 17.6k lines of hand-written CSS either.

## Baseline Inventory

### Size

| Area | Files | Vue SFCs | Lines (`wc -l`) | Scoped CSS lines |
| --- | ---: | ---: | ---: | ---: |
| `web/src` total | 345 | 172 | 87,204 | ~17,560 |
| `classic/components` | 75 | 70 | 11,994 | 5,338 |
| `classic/app` (router, shell, resources, controllers) | 45 | 3 | 11,168 | 184 |
| `classic/i18n` (3 locales × 8 namespaces + loader) | 26 | 0 | 10,609 | 0 |
| `classic/lib`, `api`, `styles` | 16 | 0 | 2,536 | n/a |
| `web/src/shared` (`http`, `preferences`) | 5 | 0 | 424 | 0 |

TypeScript in `classic`: 133 files and 30,235 lines. That breaks down into
10,350 lines of locale catalogs, 54 files (12,597 lines) that import `vue`,
`vue-router`, `vue-i18n`, `@tanstack/vue-query`, or `reka-ui`, and 55 files
(7,288 lines) that are already framework-free.

### Feature domains

| Domain | Routes (`page_routes.json`) | Files | SFCs | Lines | Scoped CSS | Largest SFCs (bytes) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| auth | `login` | 3 | 1 | 864 | 257 | |
| not-found | SPA fallback | 1 | 1 | 214 | 156 | |
| preferences | shell panel | 2 | 1 | 535 | 178 | |
| settings | `settings` | 10 | 7 | 3,265 | 492 | |
| model-prices | embedded in `models`, `settings` | 5 | 3 | 1,133 | 203 | |
| models | `models` | 14 | 10 | 4,682 | 1,481 | `ModelAliasEditor` 30k |
| home | `home` | 16 | 12 | 4,697 | 1,501 | `GatewayConnection` 36k |
| access-keys | `access-keys` | 20 | 12 | 4,863 | 768 | `AccessKeyDrawer` 31k |
| logs | `logs` (thin wrapper) | 2 | 1 | 131 | 14 | |
| monitor | `monitor`, `schedule`, content of `logs` | 33 | 26 | 13,956 | 3,571 | `InspectorTab` 52k, `LogsTab` 49k, `SchedulePanelDetail` 43k, `LogDetailDrawer` 42k |
| groups | `groups`, `group-detail` | 21 | 16 | 9,610 | 2,164 | `SubscriptionAccountCard` 68k, `GroupCredentialsTab` 60k, `GroupSettingsTab` 44k, `GroupModelsTab` 36k |
| import | `import` | 15 | 8 | 6,228 | 1,252 | `NewGroupImport` 66k, `SubscriptionCredentialStager` 42k |

### Framework API usage (port-cost signals)

| API | Occurrences | React target | Cost driver |
| --- | ---: | --- | --- |
| `computed(` | 768 | derived value in render / `useMemo` | low; React Compiler removes most manual memoization |
| `ref(` | 208 | `useState` / `useRef` | medium; each one needs a render-vs-non-render decision |
| `watch(` | 138 | event handler, derived state, or `useEffect` | **high**; most bugs come from here |
| `nextTick(` | 50 | ref callback / `useLayoutEffect` / rare `flushSync` | medium |
| `defineProps` / `defineEmits` | 145 / 78 | props / callback props | low |
| `v-model` / `defineModel` | 43 / 4 | controlled `value` + `onChange` | low |
| `<slot` outlets / slot usages | 67 / 198 | `children`, named element props, render props | medium |
| `provide` / `inject` | 7 / 9 | Context | low |
| `defineExpose` | 12 | `ref` prop + `useImperativeHandle` | low |
| `<Teleport>` / `<Transition>` | 3 / 7 | Astryx `Layer` / `createPortal`; CSS transitions / `useEntryAnimation` | low |
| `onBeforeRouteLeave` / `onBeforeRouteUpdate` | 4 | router `useBlocker` | medium (unsaved-changes flow) |
| `useQuery(` / `useQueryClient(` | 44 / 20 | `@tanstack/react-query` equivalents | low |
| `useRoute(` / `useRouter(` | 25 / 24 | router hooks | low |
| `reka-ui` imports | 11 files (Tabs, Switch, Popover) | Astryx components | low |
| `@lucide/vue` | 74 files, 59 icons | `lucide-react` (already a dependency of `@astryxdesign/theme-neutral`) | mechanical |

Mutations do not go through `useMutation` (0 uses). They call the API client
directly and report through `app/mutation-outcome.ts`, which ports as a plain
function.

### Styling facts

- Tailwind 4 is imported once (`styles/base.css`: `@import 'tailwindcss'`).
  In practice that supplies preflight and `sr-only`. Across all 1,713
  `class="…"` attributes, one token looks like a utility class.
- Styling lives in 156 `<style>` blocks, 146 of them `scoped`, with BEM class
  names (`logs-list__record`, `schedule-row--selected`). There are also 58
  global component classes in `components.css` and 214 custom properties in
  `tokens.css`.
- Dark mode uses `:root[data-theme='dark']` plus
  `@media (prefers-color-scheme: dark) { :root:not([data-theme]) }`. It is set
  before first paint by `web/public/theme-bootstrap.js`, which reads
  `localStorage['gpt-load.theme']`.

### Dead code (0 static importers at `dcabee7a`)

`components/charts/TrendChart.vue` (+ `trend-chart.ts`),
`components/config/ProxyConfigEditor.vue`,
`components/config/ProxyScopeIndicator.vue`,
`components/layout/PageSection.vue`, `components/ui/MobileRecordCard.vue`,
`components/ui/OperationNotice.vue`, `components/ui/SecretValue.vue`,
`components/ui/StatFigure.vue`. Re-verify these, then delete them in Phase 0.
Do not port them.

### Test and tooling surface

- **Playwright** (`web/e2e`, 7 specs) runs against the Vite dev server. It
  uses 149 `locator()` calls, mostly BEM class selectors such as
  `.schedule-row--selected` and `.logs-list__record`, alongside 77
  `getByRole`, 13 `getByText`, and 6 `getByTestId`. StyleX emits hashed atomic
  classes, so every class selector breaks.
- **Verify scripts** (`web/scripts/verify-*.mjs`, 8 files) read classic
  source by path, for example `frontends/classic/app/resources/groups.ts` and
  `frontends/classic/features/monitor/log-format.ts`.
- **Contract tests** run under `node:test`: `connection-json.test.ts` and
  `channel-contract.test.ts`.
- **Go contract tests** in `internal/webui/workflow_test.go` pin these
  literals: `"build": "pnpm run type-check && vite build"` in
  `web/package.json`; the web-ci command order `install --frozen-lockfile →
  lint → format → build`; no separate `type-check` step; and pnpm in the
  Dockerfile. The migration must preserve all of them. It keeps pnpm, not bun,
  because these tests and the Dockerfile require pnpm.
- **Tech docs whose `code.paths` point into classic:** `model-test-alias.md`,
  `reasoning-policy.md`, `usage-accounting.md`, and
  `billing-failure-attention/plan.md`. `.docs/db` semantic docs carry no code
  bindings, so the only Doc ID drift is in these tech docs.

### Browser state contract (shared by both frontends)

| Key | Storage | Owner |
| --- | --- | --- |
| `gpt-load.auth-key` | localStorage | `features/auth/auth-session.ts` |
| `gpt-load.locale` | localStorage | `i18n/index.ts`, `shared/preferences/locale.ts` |
| `gpt-load.theme` | localStorage | `features/preferences/theme.ts`, `public/theme-bootstrap.js` |
| `gpt-load.import-reauth-draft` | sessionStorage | `features/import/import-recovery.ts` |

These key names and value formats are frozen for the whole coexistence period.
That is what lets a user switch frontends without logging in again or losing
an import draft.

## Corrections To The Previous Draft

| Previous claim | Verified fact | Consequence |
| --- | --- | --- |
| StyleX unplugin is required for Astryx | Astryx ships prebuilt CSS and JS; no build plugin is needed for Vite | The StyleX compiler is an explicit app-styling choice, not a prerequisite |
| "CSP already compatible, no changes" | The default theme import injects `<style>` elements at runtime (`Theme.tsx` → `useInsertionEffect`). `style-src-elem 'self'` blocks that. Only `/built` themes skip injection | Built themes are mandatory, and custom themes need `astryx theme build`. Check CSP against the Go-served build |
| Optional Tailwind v4 bridge while both systems coexist | Classic does not use Tailwind utilities, and the two frontends are separate HTML documents, so their styles never share a cascade | No bridge. Layer order matters only inside the new entry |
| `@shared/http` is reused as-is | `shared/http/client-context.ts` imports `inject` from `vue` | Move `client-context.ts` into classic. `client.ts`, `errors.ts`, and `types.ts` are framework-free |
| `.docs/db` Doc IDs are bound to code paths | `doc-compiler.js` has no code-binding logic, and no Doc IDs appear in `web/src` | Reconcile `code.paths` in the 4 tech docs above at each domain cutover |
| The chart gap must be hand-ported | `TrendChart.vue` has no importers | Delete it. There is no chart dependency. (`@astryxdesign/charts` exists only as a `0.0.0-bootstrap.0` placeholder) |
| Router choice is TanStack Router vs React Router 7 | React Router is at 8.4.0 (peer `react >= 19.2.7`, Node ≥ 22.22) | v8 was evaluated; TanStack Router was chosen (see [Router](#router)) |
| Astryx `InternationalizationProvider` preferred for app strings | Astryx ships **English only** and recommends a dedicated i18n library for substantial localization | Use a dedicated library for app strings, and write local `zh-CN`/`ja-JP` catalogs for `@astryx.*` keys |
| `/v2` route prefix for grey release | This would duplicate every route in Go, break deep links and the `redirect` query, and place a UI under `/v2`, next to the `/v1` and `/v1beta` API namespaces | Same-URL, cookie-selected index |
| Register the Astryx MCP server | Not found in the current Astryx docs. The documented agent surfaces are the CLI (`component`, `template`, `docs`) and `astryx init` | Use the CLI |
| Domain order omitted `schedule` | `schedule` is a manifest route backed by `features/monitor/ScheduleView.vue` | Handled inside the monitor domain |
| Security `overrides` live in `package.json` | They are duplicated in `web/package.json` and `web/pnpm-workspace.yaml`, which also has `allowBuilds: vue-demi` | Re-evaluate both, and drop `vue-demi` when classic is deleted |

## Target Architecture

### Stack mapping

| Concern | Current | Target (pin) | Notes |
| --- | --- | --- | --- |
| Runtime | `vue@3.5.42` | `react@19.3.0`, `react-dom@19.3.0` (published 2026-09-09) | `<StrictMode>` in dev, so missing effect cleanups show up early |
| Design system | `reka-ui@2.10.4` + ~50 `App*` components | `@astryxdesign/core@0.6.2` (→ `0.6.3` after 2026-09-30), plus `@astryxdesign/theme-neutral` at the same version as the base for a custom theme | Beta; release history in [Risks](#risks) |
| App styling | scoped BEM CSS + `tokens.css` | `@stylexjs/stylex@0.19.1` + `@stylexjs/unplugin@0.19.1` | Astryx CSS itself is precompiled |
| Router | `vue-router@5.3.1` | `@tanstack/react-router@1.170.x` (code-based routes) | Confirmed; see [Router](#router) |
| Server state | `@tanstack/vue-query@5.103.1` | `@tanstack/react-query@5.103.x` | Query keys, fetchers, and invalidation move to shared |
| i18n | `vue-i18n@11.4.12` | `react-intl@12.1.x` for app strings; Astryx `InternationalizationProvider` for component strings | Confirmed; see [Internationalization](#internationalization) |
| Icons | `@lucide/vue@1.46.0` | `lucide-react@1.48.x` | Same icon set, so names map 1:1 |
| Dates | `@internationalized/date@3.12.4` | unchanged | Astryx date inputs are built on the same foundation |
| Build | Vite 8 + `@vitejs/plugin-vue` | Vite 8 + `@vitejs/plugin-react@6` (Oxc) + `stylex.vite()` + React Compiler (`@rolldown/plugin-babel`, `@babel/core`, `babel-plugin-react-compiler@1.0.0`) | Use the stable Babel compiler. The Rust `compiler: true` option (`oxc-transform-react`) is marked experimental |
| Browser target | Vite 8 default (`baseline-widely-available`) | `build.target: 'chrome125'`, `"browserslist": ["chrome >= 125", "edge >= 125"]` | `browserslist` also feeds StyleX's Lightning CSS output |
| Type check | `vue-tsc` | `tsc -p tsconfig.astryx.json` (`"jsx": "react-jsx"`) | `vue-tsc` stays until classic is deleted |
| Lint | `eslint-plugin-vue`, `@vue/eslint-config-typescript` | `eslint-plugin-react-hooks@7` (compiler rules as errors), `@stylexjs/eslint-plugin@0.19.1`, `typescript-eslint` | Flat config scoped by file glob during coexistence |
| Component tests | none | optional `vitest@5` + `@testing-library/react@16` | Contracts stay on `node:test` |
| E2E | Playwright 1.62 | unchanged; one project per frontend | Selectors rewritten in Phase 0 |
| Agent tooling | none | `@astryxdesign/cli` as a devDependency plus an `"astryx"` script | See [Toolchain](#toolchain-and-repository-contracts) |

These stay unchanged: the Go backend, the embed pipeline (`Makefile`
`_web-build` → `pnpm --dir web run build` → `internal/webui/dist`), dev proxy
targets, `@shared/http/client.ts`, `@shared/preferences/locale.ts`, locale
catalog content, contract tests, and the pnpm/Node toolchain.

### Directory layout (during coexistence)

```text
web/
  index.html                  classic entry (unchanged)
  astryx.html                 new entry: same <head> contract (theme-bootstrap.js, favicon)
  src/
    main.ts                   classic bootstrap loader (unchanged)
    frontends/
      classic/                Vue; shrinks as domains move
      astryx/
        main.tsx              createRoot + providers
        app/                  router, shell, guards, providers, query client
        features/<domain>/    React pages and domain components
        components/           app-level composites on Astryx primitives
        theme/                gptload.theme.ts + built artifacts (committed)
        styles/entry.css      the only global CSS: layer order + Astryx imports
    shared/                   framework-free: http, control resources, domain logic, i18n catalogs, controllers
```

Aliases: `@` keeps pointing at `frontends/classic` (unchanged), `@app` points
at `frontends/astryx`, and `@shared` points at `shared`. Scope
`@vitejs/plugin-react` to `src/frontends/astryx/**` so Fast Refresh never
touches classic modules.

### Build configuration sketch

```ts
// web/vite.config.ts (additions only)
import babel from '@rolldown/plugin-babel'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import stylex from '@stylexjs/unplugin'

const astryxSources = /src\/frontends\/astryx\/.*\.[jt]sx?$/
const reactCompiler = reactCompilerPreset()
reactCompiler.rolldown.filter.id.exclude = ['**/src/frontends/classic/**', '**/src/shared/**']

plugins: [
  stylex.vite({
    useCSSLayers: { before: ['reset', 'astryx-base', 'astryx-theme'], prefix: 'app' },
    sxPropName: false,
  }),
  vue(),
  react({ include: astryxSources }),
  babel({ presets: [reactCompiler] }),
  tailwindcss(),               // classic only; removed with classic
  frontendSelectorDevPlugin(), // see Coexistence Architecture
],
build: {
  outDir: '../internal/webui/dist',
  emptyOutDir: true,
  manifest: true,
  target: 'chrome125',
  rolldownOptions: { input: { classic: 'index.html', astryx: 'astryx.html' } },
},
```

Plugin ordering and scope:

- `stylex.vite()` must come before the React plugin so Fast Refresh keeps
  working.
- StyleX appends its CSS to the entry's emitted CSS asset, and each output
  bundle gets its own aggregated StyleX CSS.
- `build.rollupOptions` is a deprecated alias in Vite 8, so the sketch uses
  `build.rolldownOptions`.
- `build.target: 'chrome125'` applies to the classic entry too. That matches
  the confirmed browser policy, and classic loses no supported browser.
- The React Compiler preset excludes classic and shared, since shared code is
  framework-free by rule. Verify the exclude glob against the preset's
  `rolldown.filter` shape at the pinned plugin version.

React Compiler rules:

- Write idiomatic components with no manual `useMemo`, `useCallback`, or
  `memo` by default. Add them only with a measured reason, stated in a
  comment.
- `eslint-plugin-react-hooks@7` compiler diagnostics are errors, so a
  component the compiler cannot optimize fails lint instead of silently
  staying uncompiled.
- `"use no memo"` is a per-file escape hatch for proven compiler bugs only.
  Each use carries a comment linking the upstream issue, and it is removed
  once fixed.
- Spike check: the compiler and StyleX both run Babel over the same files.
  Confirm that compiled `stylex.props(...)` output is unchanged and that Fast
  Refresh still preserves state.

### CSS cascade in the new entry

```css
/* web/src/frontends/astryx/styles/entry.css */
@layer reset, astryx-base, astryx-theme;
@import '@astryxdesign/core/reset.css';
@import '@astryxdesign/core/astryx.css';
@import '../theme/gptload.theme.css'; /* output of `astryx theme build` */
/* StyleX appends: @layer reset, astryx-base, astryx-theme, app.priority1, …; */
```

Resulting precedence, lowest to highest: `reset` → `astryx-base` (component
defaults) → `astryx-theme` (theme component overrides) → `app.*` (our
`xstyle` and `stylex.props`). Rules:

- The new entry has **no unlayered CSS**. Unlayered rules beat every layer
  regardless of specificity.
- Never import `classic/styles/*` into the new entry. Token names partly
  overlap under the `--color-*` prefix.
- Spike check: assert with a computed-style test that an `xstyle` override
  beats both the Astryx default and a theme override.

### Theme and tokens

- Base the custom theme on `neutral` (confirmed) with `defineTheme` and
  `extends`, mapping by semantic
  intent from `tokens.css`: accent `['#1c4f6e', '#6fb2d6']` (light/dark
  tuple), a warm neutral style (canvas `#eeede9` / `#0b0d10`), status colors
  from `--color-success|warning|danger|info` light/dark pairs, radius base
  `7px` (control) / `10px` (sheet), and typography body family set to the
  current `system-ui, -apple-system, 'Segoe UI', sans-serif`.
- Why override the font: the neutral theme's `--font-family-body` starts with
  `Figtree` but ships no `@font-face`. The rendered font would then depend on
  whether a machine happens to have Figtree installed. Loading Figtree from
  Google Fonts would also violate `font-src 'self'`.
- Build with `pnpm exec astryx theme build src/frontends/astryx/theme/gptload.theme.ts`
  and commit the generated CSS/JS/`.d.ts`. Import the theme object from the
  built module (the `__built` flag disables runtime injection). Add a
  `lint`-stage check that rebuilds the theme and fails on diff.
- Keep dark mode single-sourced. Leave `theme-bootstrap.js` and
  `gpt-load.theme` unchanged. Drive `<Theme mode>` from the preference store.
  The root `Theme` already syncs `data-theme="light|dark"` and removes it for
  `system`, which is exactly what `theme-bootstrap.js` and the classic
  controller do. Do not run a second theme controller that writes
  `data-theme`.
- **Density: match classic's compact metrics (confirmed).** Astryx defaults
  are larger, so the theme and app defaults must reproduce these classic
  values from `tokens.css`:

  | Metric | Classic token | Target value |
  | --- | --- | --- |
  | Body text | `--text-body` | 13.5px (theme `typography.scale.base`) |
  | Meta / small / label text | `--text-meta` / `--text-sm` / `--text-label-xs` | 12px / 11.5px / 10.5px |
  | Section / panel titles | `--title-section` / `--title-panel` | 16px / 22px |
  | Controls | `--control-compact` / `--control-sm` / `--control-md` / `--control-lg` | 30 / 34 / 38 / 42px |
  | Inline setting controls | `--setting-control-height` | 26px |
  | Collection row | `--collection-row-height` | 48px |
  | Radius | `--radius-control` / `--radius-tag` / `--radius-sheet` | 7 / 6 / 10px |
  | Spacing unit | `--space-1` | 4px (the same base as Astryx `--spacing-1`, so no remapping) |
  | Touch target (narrow screens) | `--touch-target` | 44px, kept regardless of density |

  Mechanism, in order of preference:

  1. Theme scale inputs (`typography.scale`, `radius` base).
  2. The default component size through Astryx `SizeContext` at the app
     root, and `Table` `density`.
  3. Theme component overrides (`components: { … }`) for the remaining gaps.

  Do not use per-call-site `xstyle` for density. Phase 1 measures each value
  above in the browser, with a ±1px tolerance, before any domain migrates.
- Domain-only tokens that Astryx lacks, such as health-bar height, the
  collection row height, and status-bar height, go into one app `defineVars`
  group. There should be no hex or px literals in feature code.

## Coexistence Architecture

### Why same-URL selection instead of a prefix

A prefix would double every route in Go and in `page_routes.json`, and it
would require rewriting the login `redirect` target and links stored outside
the app. It also parks a UI namespace next to `/v1` and `/v1beta`. Same-URL
selection keeps one route manifest and one URL space. Deep links and bookmarks
keep working, and switching frontends means setting a cookie and reloading.

### Go server (`internal/webui`)

1. **Manifest v2.** `page_routes.json` gets an optional per-route
   `"astryx": true` flag and `version: 2`. Both strict parsers change
   together: `page_routes.go` (`DisallowUnknownFields`, version check) and
   `app/page-routes.ts` (`routeFields`). `page_routes_test.go` gets cases for
   the flag.
2. **Dual index.** `newServerWithPages` also reads `dist/astryx.html` and
   stores it as `astryxIndex`, which is `nil` when the file is absent. Builds
   without the new entry therefore behave exactly as today.
3. **Selection.** `HTTPModule` binds one handler per page instead of the
   shared `s.serveIndex`:

   ```go
   func (s *Server) indexFor(c *gin.Context, page pageRoute) []byte {
       if s.astryxIndex != nil && page.Astryx && frontendPreference(c) == "astryx" {
           return s.astryxIndex
       }
       return s.index
   }
   ```

   `frontendPreference` reads the cookie `gpt-load.frontend` (values
   `classic|astryx`; anything else means classic). The SPA not-found fallback
   uses the same preference once the new entry implements its not-found page
   in Phase 1. `Cache-Control: no-cache`, `Vary: Cookie` (the cookie selects
   the document, so caches must not serve one document for the other), CSP,
   `nosniff`, and `DENY` headers stay identical for both documents.
4. **Assets.** Both entries emit into `dist/assets/` with content hashes, so
   the existing `/assets/*filepath` handler (`immutable`) serves both
   unchanged. `favicon.svg` and `theme-bootstrap.js` are shared.
5. **Cutover.** Flip the default: a missing cookie means astryx. Delete the
   selection code and the manifest flag (back to v1 semantics) together with
   classic.

Security boundary: the cookie only chooses between two embedded static
documents. It must never influence auth, API routing, or file paths.
`RedirectTrailingSlash = false` (`internal/app/app.go`) stays, and the
trailing-slash behavior of both frontends must match it (see
[Router](#router)).

### Switching and cross-frontend navigation

- **Toggle.** Classic's preferences panel sets `gpt-load.frontend=astryx`
  (`Path=/; SameSite=Strict`) and reloads. The new shell has a matching
  "return to classic" control.
- **New → unmigrated route.** The new router registers every manifest route
  (guards, titles, and meta need the match), but an `astryx: false` path must
  never render inside the astryx document: `app/route-link.tsx` resolves the
  manifest flag per href and renders a plain `<a>` (document load) for
  unflagged targets. All navigation entry points use it — the `LinkProvider`
  adapter, shell nav, not-found links — and programmatic navigations use
  `window.location.assign/replace`. `RoutePageStub` self-heals with a document
  navigation if an unflagged path is ever reached anyway.
- **Classic → migrated route.** Classic stays complete until cutover. In-app
  navigation remains in classic, and the next reload picks up the new
  frontend. Optionally, a classic `beforeEach` can hard-navigate when the
  cookie is `astryx` and the target is migrated.
- **Shared state.** The [browser state contract](#browser-state-contract-shared-by-both-frontends)
  keys are the only state crossing the boundary. Add a contract test that
  imports the shared key constants from both frontends.

### Vite dev and Playwright

The Vite dev server serves `index.html` for every navigation. A small
`configureServer` plugin (`frontendSelectorDevPlugin`) rewrites HTML
navigations to `/astryx.html` when the cookie is `astryx` and the path matches
a flagged manifest route. It reads the same `page_routes.json` through the
existing `pageRouteManifestPath`. Playwright then defines two projects,
`classic` and `astryx`, that differ only in a `storageState` cookie. Each
migrated domain's specs run against both projects until cutover.

Dev has no CSP, so the gate also needs a Playwright project that targets the
Go binary built by `make build` and fails on any `securitypolicyviolation`
event or CSP console error.

## Phase 0 Shared-Layer Extraction (in classic, before any React)

Goal: every piece of non-UI logic ends up in `web/src/shared`, framework-free
and tested once, and classic consumes it through thin adapters. Each step is
behavior-neutral and ships on its own.

| Step | From | To | Adapter |
| --- | --- | --- | --- |
| Control API types and protocols | `classic/api/control/*` | `shared/control/` | none |
| Query keys and invalidation | `classic/app/query-keys.ts`, `resources/invalidation.ts` | `shared/control/` | none |
| Resource fetchers and DTO mapping | `classic/app/resources/*.ts` (23 files) | `shared/control/resources/*.ts` taking **plain** params and returning plain `{ queryKey, queryFn }` objects | classic wraps them with `computed`/`toValue`. Today 13 resource files import `vue` for `MaybeRefOrGetter`, and 14 import `@tanstack/vue-query` |
| Pure helpers | `classic/lib/*` (11 files), the framework-free `features/**/*.ts` (55 files, 7.3k lines) | `shared/lib`, `shared/domain/<domain>` | none |
| Controllers | theme, toast, unsaved-changes, import-recovery, ephemeral-state, auth-session | `shared/controllers/*` as plain objects with `subscribe`/`getSnapshot` | classic: `ref` + subscription; React: `useSyncExternalStore` |
| API client context | `shared/http/client-context.ts` (Vue `inject`) | `classic/app/api-client-context.ts` | React gets its own Context |
| Route rules | `pagePathMatches`, `safeRedirect`, `decodedPathSegments`, query normalization | `shared/routing/` | Both routers call them. **`safeRedirect` is the open-redirect guard: move it together with its tests** |
| Locale catalogs | `classic/i18n/locales/**` | `shared/i18n/locales/**` | Loader stays per framework |
| ICU-compatible catalog syntax | 4 message keys × 3 locales (literal JSON examples and the `<port>` token; see [Internationalization](#internationalization)) | `{example}` / `{port}` placeholders with values supplied by code | new `verify:i18n-icu` script (devDependency `@formatjs/icu-messageformat-parser`) parses every message and checks key parity across locales |
| Verify scripts | read classic paths | read `shared/...` paths | They then validate both frontends |
| E2E selectors | BEM class locators | `getByRole` / `getByLabel` / `getByText`, plus `data-testid` only where semantics are insufficient | the same specs run against both frontends |
| Dead code | 8 components listed above | deleted | none |
| Tech docs | `code.paths` into classic | updated to `shared/...` | none |

Exit criteria: `pnpm --dir web run build`, `lint`, `format`, all `verify:*`,
`test:*`, and e2e pass with unchanged behavior, and no file under
`web/src/shared` imports `vue`, `vue-router`, `vue-i18n`, `@tanstack/vue-query`,
or `reka-ui`. Enforce that last condition with an ESLint `no-restricted-imports`
rule on `src/shared/**`.

## Vue → React Translation Rules

| Vue pattern | React rule | Pitfall to avoid |
| --- | --- | --- |
| `computed(() => f(a, b))` | compute during render; the React Compiler memoizes it | storing derived data in state and syncing it with an effect; hand-written `useMemo`/`useCallback` |
| `ref` read by the template | `useState` | |
| `ref` never rendered (timers, abort pools, DOM handles) | `useRef` | putting it in state and causing extra renders |
| `watch(source, cb)` reacting to user input | run `cb` in the event handler that changed the source | effect chains that react to state set by other effects |
| `watch` resetting local state on prop change | `key` the component on the identity, or derive the value | a `useEffect` that calls `setState` from props |
| `watch(..., { immediate })` that fetches | TanStack Query with the value in `queryKey` | hand-rolled fetch in an effect without abort |
| `watch` syncing an external system (media query, storage, document title) | `useEffect` with cleanup, or `useSyncExternalStore` | missing cleanup, which StrictMode exposes by double-invoking |
| `nextTick(() => el.focus())` | ref callback or `useLayoutEffect` | `setTimeout(0)` |
| `v-model` / `defineModel` | controlled `value` + `onChange` props; Astryx inputs are controlled | uncontrolled↔controlled switches (`undefined` initial value) |
| default slot / named slots / scoped slots | `children` / element props (`header`, `actions`) / render function props | re-creating components inside render |
| `provide`/`inject` of controllers | Context providers assembled once in `main.tsx`, mirroring `bootstrap.ts` | putting frequently changing values in one big context |
| `defineExpose` | `ref` as a prop (React 19) + `useImperativeHandle` | exposing more than the parent needs |
| `:class="{ 'x--on': on }"` | `{...stylex.props(styles.x, on && styles.on)}` | string-built class names |
| `<style scoped>` | `stylex.create` in the same file, or Astryx layout props | copying BEM CSS verbatim |
| `<Transition>` | CSS transitions in StyleX + `prefers-reduced-motion`, or Astryx `useEntryAnimation` | JS animation libraries |
| `onBeforeRouteLeave` + `unsaved-changes.ts` | router `useBlocker` backed by the shared unsaved-changes controller | blocking without also handling `beforeunload` |
| `t('ns.key', { count })` | `t('ns.key', { count })` from the `useT()` helper over `useIntl().formatMessage`; same flattened keys | calling `formatMessage` with inline `defaultMessage` strings, which forks the catalog |
| `<i18n-t keypath tag="p">` with slot content (3 uses in `LoginView`) | `formatMessage({ id }, { slot: <code>…</code> })` or `<FormattedMessage values>` | ICU rich-text tags in catalogs, which classic cannot render during coexistence |
| `v-html` (vendored SVG in `ChannelIcon`) | `dangerouslySetInnerHTML` on build-time assets only, keeping the justification comment | using it for any runtime or API string |

Querying rules: query keys contain plain values, never refs or objects
recreated on every render. Keep `retry: false` for queries and mutations (see
`app/query.ts`). Port `use-visible-refetch.ts` as a `refetchOnWindowFocus` /
visibility-based `refetchInterval` policy, not as an effect.

## Component Mapping

Usage counts are the number of classic files importing each component.
Re-check each Astryx API with `pnpm exec astryx component <Name>` at the pinned
version before porting.

| Classic | Uses | Astryx target | Notes |
| --- | ---: | --- | --- |
| `AppButton` / `IconButton` | 53 / 17 | `Button` / `IconButton` | |
| `InlineFeedback`, `QueryFeedback`, `CompactFieldError` | 34 / 22 / 11 | `Banner`, `FieldStatus` | |
| `AppTooltip`, `OverflowTooltip` | 23 / 21 | `Tooltip`, `useOverflow` | anchor positioning: see [browser tiers](#risks) |
| `StatusBadge` | 22 | `Badge`, `StatusDot` | `status-presenter.ts` moves to shared |
| `SkeletonSurface`, `SkeletonBlock` | 19 / 3 | `Skeleton` | |
| `AsyncRefreshIndicator` | 17 | `Spinner` | |
| `AppTextInput`, `AppSearchInput` | 17 / 9 | `TextInput`, `InputGroup` | |
| `AppSelect`, `SearchableSelect`, `SearchableMultiSelect` | 16 / 1 / 2 | `Selector` (`hasSearch`), `MultiSelector` | dropdowns rely on anchor positioning |
| `AppConfirmDialog`, `AppTypedConfirmation` | 13 / 2 | `AlertDialog` (+ `TextInput` for typed confirmation) | |
| `PageFrame`, `LedgerSheet`, `PageHeader`, `PanelHeader`, `Surface`, `SurfaceCard` | 12 / 12 / 8 / 5 / 4 / 1 | `AppShell`, `Layout`, `Section`, `Card`, `Heading`, `Toolbar` | start from `astryx template` page layouts |
| `ChannelIcon`, `BrandMark` | 12 / 2 | keep (domain SVG) | |
| `AppSwitch` | 11 | `Switch` | |
| `FormField`, `SettingRow`, `SettingBlock` | 10 / 5 / 2 | `Field`, `FormLayout` | |
| `CopyChip`, `CopyButton`, `CopyAction`, `CopyFallbackDialog` | 10 / 3 / 1 / 3 | `Token`/`Button` + `useClipboard` | keep the fallback dialog for insecure contexts |
| `SegmentedControl` | 9 | `SegmentedControl` | |
| `AppRelativeTime`, `AppDateTime` | 8 / 6 | `Timestamp` | check zh-CN/ja-JP formatting |
| `LedgerRecordList`, `DataTable` | 8 / 3 | `Table` + plugins (`sortable`, `pagination`, `filtering`, `selection`, `stickyColumns`, `columnSettings`, `rowExpansion`, `rowStatus`, …) | collections are **server-paginated** (ADR-0001), so use the `*State` hooks in controlled mode and never the client-side `paginateData` |
| `EmptyState` | 7 | `EmptyState` | |
| `AppDrawer`, `LogDetailDrawer`, `AccessKeyDrawer` | 7 (+ domain) | no drawer: `Dialog`, `BottomSheet`, or a swizzled `Dialog` side sheet | spike item |
| `AppDialog`, `AppPopover` | 6 / 6 | `Dialog`, `Popover` | |
| `PaginationBar` | 6 | `Pagination` | |
| `StickySaveBar`, `UnsavedChangesDialog` | 6 / 1 | `Toolbar` + `AlertDialog` + router blocker | |
| `AppTabs`, `SectionNav` | 2 / 1 | `TabList`, `NavMenu` / `Outline` | |
| `DisclosurePanel` | 2 | `Collapsible` | |
| `CredentialHealthBar`, `QuotaProgressBar` | 2 / 1 | `ProgressBar` + domain wrapper | |
| `AppDateTimeRangePicker` | 1 | `DateRangeInput` / `DateTimeInput` | |
| `CodeBlock` | 1 | `CodeBlock` | |
| `AppToastViewport` + `app/toast.ts` | 1 | `Toast` / `useToast` | |
| `RouteAnnouncer` | 1 | `useAnnounce` | |
| `HeaderRulesEditor`, `ParameterOverrideRulesEditor`, `ProxyOverrideControl` | 2 / 1 / 2 | rebuild on primitives | validation logic moves to shared in Phase 0 |
| `CollectionFilterBar`, `CollectionStatusSummary` | 3 / 3 | rebuild on `Toolbar`, `Selector`, `Badge` | filter state lives in typed search params |

Customization ladder (Astryx guidance): props → `xstyle` → theme tokens /
component overrides → `swizzle`. Every swizzled component is a fork that
`astryx upgrade` will not update. Record each one in this document with the
reason and the Astryx version it was taken from.

## Router

Decision (confirmed 2026-09-25): TanStack Router with code-based routes (no
file-router plugin). The Phase 1 spike on `group-detail` and `logs` proves the
parity rules below. It does not revisit the choice.

Why:

- **Typed search params.** Classic encodes `tab`, `mode`, log filters,
  schedule targeting, and collection filters in the query string
  (`route-query.ts`, `log-filters.ts`, `group-collection-route.ts`).
  `validateSearch` makes those typed and validated in one place.
- `beforeLoad` covers the existing `beforeEach` guards (`requiresAuth`,
  `adminOnly`, `pagePathMatches` → not-found) and the `beforeResolve`
  namespace preload.
- It also provides `useBlocker`, scroll restoration, and router-level
  `caseSensitive: true` (classic uses `sensitive: true`).

Parity rules:

- Generate the route tree from `page_routes.json`, converting `:id` to `$id`
  in one adapter so the manifest stays the single source of truth.
- `trailingSlash: 'preserve'` plus an explicit not-found for trailing-slash
  paths. The default `'never'` would redirect `/groups/` to `/groups` on the
  client, while Go (`RedirectTrailingSlash = false`) answers such paths with
  the 404 fallback document. Classic's `strict: true` treats them as not
  found, so the new router must too.
- Scroll: `scrollRestoration: true` restores positions on history nav; every
  query-only `navigate` (filters, pagination, canonicalization, detail
  open/close) passes `resetScroll: false` — matching classic's "same-path
  query changes keep scroll" behavior.
- Search codec: `app/search-codec.ts` mirrors vue-router `parseQuery`/
  `stringifyQuery` exactly (`+`→space pre-split, independent key/value decode,
  bare key→`null`, duplicate keys→arrays, null-prototype result so `__proto__`
  stays an own property); `scripts/search-codec.test.ts` pins the contract.
- `validateSearch` must return a **sparse** object: TanStack writes the
  validated search back to the URL, so injecting defaults turns clean URLs
  verbose and makes component canonicalization fire on every load. Views parse
  defaults from `location.search` themselves; `validateSearch` only emits the
  canonical serialization of what was present.
- Titles and `<html lang>` update on navigation; announce route changes with
  `useAnnounce`.
- Port the existing `safeRedirect` cases verbatim as shared tests before
  either router uses them.

Rejected: React Router 8 in data mode. It shares the manifest's `:id` syntax
and has `useBlocker`, but search params would need manual parsing and typing
at every call site. Classic's query-string state is too central for that
trade.

## Internationalization

Decision (confirmed 2026-09-25): `react-intl` for app strings, and Astryx
`InternationalizationProvider` for Astryx component strings. The two
providers are kept on the same locale.

### Runtime shape

- **Provider.** `IntlProvider` with `locale`, `defaultLocale="en-US"`, and
  the merged message map. Callers outside components resolve messages
  through the controller's sync API (`t()` is render-path only); a separate
  imperative `createIntl` instance was removed as dead weight in final
  review — do not reintroduce one without a real consumer.
- **Keys.** Catalogs stay as the nested TypeScript `MessageTree` modules in
  `shared/i18n/locales`. The loader flattens them to dot paths such as
  `home.ledger.title`, which are the key strings classic already uses, so no
  call site needs a key change. Derive a `MessageId` type from the flattened
  `en-US` catalog. The spike decides whether a type over 10k catalog lines
  keeps `tsc` fast enough; the fallback is plain `string` ids plus the parity
  script.
- **Namespaces.** Keep the 8 namespaces (`core` + 7 route namespaces) and the
  existing dynamic `import()` map. The route `beforeLoad` awaits
  `loadNamespaces(route.messageNamespaces)`, which mirrors classic's
  `beforeResolve`. It merges the loaded namespaces into the message map before
  the page renders. `group-detail` still declares `group`, `import`, and
  `monitor`; a missing declaration shows raw keys, exactly as it does today.
- **Fallback.** react-intl never falls back to another locale's catalog. On a
  missing id it renders the `defaultMessage` or the id and reports
  `MISSING_TRANSLATION`. To reproduce vue-i18n's `fallbackLocale: 'en-US'`,
  every namespace load merges the `en-US` messages first and the active
  locale's messages over them.
- **Errors.** `onError` throws on `MISSING_TRANSLATION` and `FORMAT_ERROR` in
  dev and tests. In production it logs each id once and keeps rendering.
- **Call sites.** A `useT()` helper returns
  `(id, values?) => intl.formatMessage({ id }, values)`. Call sites stay
  short and grep-able, and never pass inline `defaultMessage` text.
- **Rich text.** Slot-style interpolation (`<i18n-t>`) becomes element values
  for `{placeholder}` arguments. ICU `<tag>` syntax stays out of catalogs
  while classic still reads them.
- **Dependency note.** `react-intl@12.1.3` pins `intl-messageformat@12.1.2`,
  while `@astryxdesign/core@0.6.x` depends on `intl-messageformat@^11.2.9`.
  The two majors do not dedupe, so both ship until Astryx moves to 12. Count
  both in the Phase 1 bundle measurement.

### Catalog syntax compatibility (measured; corrects the previous draft)

The earlier claim that catalogs port byte for byte was wrong. My first check
missed the backslash-escaped form in the TypeScript sources. vue-i18n and
ICU disagree in exactly two places:

| Case | Where | vue-i18n form | ICU / formatjs behavior |
| --- | --- | --- | --- |
| Literal JSON credential examples | `import.ts`, 3 keys per locale (Azure-style, `bedrock`, `vertex`), 9 messages | literal interpolation `{'…'}` | parses as an argument named with quotes → `FORMAT_ERROR` |
| `<port>` token | `import.ts` `callbackPlaceholder`, one per locale (`<port>` in en-US and ja-JP, `<端口>` in zh-CN) | literal text | parses as an unclosed rich-text tag → `FORMAT_ERROR` |

Everything else is compatible:

- There are no pipe plurals (`a | b`) and no linked messages (`@:`).
- `#` appears only outside plural blocks, where ICU treats it as a literal.
- Apostrophes appear only before letters (`instance's`), which ICU keeps
  literal.
- The 70 `{count}` placeholders and every other `{name}` argument are plain
  interpolation in both runtimes.

Fix it in Phase 0, while vue-i18n still reads the catalogs. Move the literal
JSON examples and the angle-bracketed port token out of the messages into
interpolation values supplied by code (`{example}`, `{port}`). `{name}`
interpolation is the only syntax both runtimes treat identically. The
translatable word (`port`, `端口`) becomes its own key without angle brackets,
and code adds the brackets. From then on,
`verify:i18n-icu` parses every message in all three locales with
`@formatjs/icu-messageformat-parser` and checks key parity, so a new
incompatible message fails `lint` before it reaches the new frontend.

### Astryx component strings and locale switching

- **Astryx strings.** Wrap the app in `InternationalizationProvider` with the
  same locale as `IntlProvider`. Astryx ships only `en`. Write
  `astryx/locales/zh-CN.json` and `ja-JP.json` with the shape of
  `@astryxdesign/core/locales/en.json`, and add a `verify:astryx-i18n` script
  that fails when the installed version's key inventory differs from ours.
  Run it on every Astryx upgrade. `en-US` falls back to the shipped `en`.
- **Locale switching.** One store (`gpt-load.locale`) drives `IntlProvider`,
  the imperative intl instance, the
  Astryx provider, `<html lang>`, and the `Accept-Language` header that the
  API client sets through `getLocale`.

## Toolchain And Repository Contracts

- Keep the `build` script literal. During coexistence `type-check` becomes
  `vue-tsc --noEmit -p tsconfig.app.json && tsc --noEmit -p tsconfig.astryx.json && tsc --noEmit -p tsconfig.node.json`.
  `tsconfig.app.json` must exclude `src/frontends/astryx/**`, since it
  currently includes `src/**/*.tsx` and has no `jsx` setting.
- **CI.** Fold new checks into `lint` or `build` so the web-ci command set and
  order pinned by `workflow_test.go` stays valid. If a new CI step is truly
  needed, update the Go test in the same change.
- **ESLint** flat config: Vue configs are scoped to
  `src/frontends/classic/**`; `react-hooks` (recommended, including the
  compiler rules) and `@stylexjs` (`valid-styles`, `no-unused`,
  `valid-shorthands`, `sort-keys`) apply to `src/frontends/astryx/**`;
  `no-restricted-imports` guards `src/shared/**`.
- **Dependencies.** Add each one with `pnpm --dir web add` using an exact
  version at least 7 days old. Dependabot already covers `npm /web`. Re-derive
  the `overrides` in both `package.json` and `pnpm-workspace.yaml` against the
  new tree. `@astryxdesign/cli` pulls in `postcss@^8.5.23`, `jscodeshift`,
  and Babel, but as a devDependency only. Its peers (`lab`, `charts`,
  `theme-neutral`, `core`) are optional, so install stays clean.
- **Astryx CLI.** Add `"astryx": "node node_modules/@astryxdesign/cli/clients/cli/bin/astryx.mjs"`
  as Astryx recommends. `astryx init` writes a component index into
  `AGENTS.md`/`CLAUDE.md`. Run it into a scratch location and move the
  relevant part into a web-scoped `web/AGENTS.md` linked from the root. It
  must never overwrite the root `AGENTS.md` or user-level agent files.
- **Artifacts.** `emptyOutDir: true` already deletes the tracked
  `internal/webui/dist/assets/embed-placeholder.txt` on every build. That is
  existing behavior, unchanged by this plan. Do not commit build output.

## Phased Delivery

Sizes are the classic line counts from the
[baseline](#feature-domains). They signal relative effort; they are not
schedule estimates. Each phase ends at a shippable state on `dev`.

### Phase 0: Framework-neutral preparation (classic only)

Scope: everything in [Phase 0 Shared-Layer Extraction](#phase-0-shared-layer-extraction-in-classic-before-any-react).
Exit criteria are listed there. No React dependency lands in this phase.

### Phase 1: Scaffold, coexistence, and spikes (go/no-go gate)

Scope:

- The `astryx.html` entry, `main.tsx`, providers (Query, Router, i18n,
  Astryx `Theme` + `InternationalizationProvider`, toast), the built custom
  theme, and `entry.css`.
- Go dual index, manifest v2, the dev selector plugin, and the Playwright
  project matrix, including the Go-served CSP project.
- Shell parity: `AppShell` (navigation, admin-only items, principal type),
  `AuthGate`, `login`, `not-found`, preferences (theme, locale, frontend
  toggle), and the route announcer.
- Spikes: (a) the `groups` collection on `Table` with server-driven
  sort/filter/pagination and keyboard navigation; (b) the log detail surface
  as `Dialog`/`BottomSheet`/swizzled side sheet with focus trap, Esc, and
  focus return; (c) the log time-range filter on `DateRangeInput` with
  zh-CN/ja-JP Astryx strings; (d) React Compiler + StyleX in the same Babel
  pipeline, covering compiled output, Fast Refresh, and zero compiler lint
  errors on the shell; (e) TanStack Router typed search and the parity rules
  on `logs` and `group-detail`; (f) react-intl namespace loading, `en-US`
  fallback merge, and `MessageId` typing cost; (g) compact density
  measurements for every metric in [Theme and tokens](#theme-and-tokens).

Browsers for the gate: the latest stable Chrome (Playwright's bundled
Chromium) on every run, plus a Chromium 125 binary (for example installed
with `npx @puppeteer/browsers install chrome@125`) wired into a Playwright
project through `launchOptions.executablePath`. The Chromium 125 project runs
at phase boundaries and before cutover, not on every change.

Go/no-go criteria (all must hold):

1. The Go-served build renders the shell and every spike with zero CSP
   violations in both Chromium 125 and the latest Chrome.
2. There is no theme flash on reload in light, dark, and system modes with
   `theme-bootstrap.js` unchanged.
3. The spike table holds 1,000 rows and passes the existing collection e2e
   flows after the selector rewrite.
4. Initial JS+CSS (gzip) for the new shell is measured against the classic
   shell and recorded here. Astryx CSS alone is ~30 KB gzip (`astryx.css`
   171 KB raw, not tree-shaken), plus ~4 KB for reset.
5. No required behavior needs more than three swizzled components.
6. Every density metric matches classic within ±1px, reached through theme
   inputs, `SizeContext`, `Table` density, or theme component overrides
   only.
7. `verify:i18n-icu` passes, and the shell renders zh-CN, en-US, and ja-JP
   with no `MISSING_TRANSLATION` or `FORMAT_ERROR` from react-intl.

If the gate fails, stop. Phase 0 still stands on its own. Record the reason
and delete the scaffold.

**Outcome (recorded 2026-09-26, commits `55add5be` `ef6b2779` `d0362813`
`c28a1c2c` + review-fix batch):** all seven gates passed; user chose **go**.

1. CSP: 15/15 — bundled Chromium, Chrome for Testing 125.0.6422.78, system
   Chrome 153, across classic `/` and all flagged routes. First run through
   `internal/webui/cmd/webui` (the full binary cannot compile on Windows —
   `securefile`/`catalog` are linux-only); **re-verified against the real
   linux binary built and served from WSL2** (`go build` in WSL,
   `GPT_LOAD_ORIGIN=http://localhost:<port>` connects the spec to the running
   server — 15/15 identical). The `cmd/webui` harness remains for quick local
   iteration; Unix CI continues to exercise the production binary path.
2. Theme flash: 4/4 — `data-theme` correct at DOMContentLoaded for all three
   modes; `theme-bootstrap.js` unchanged.
3. 1,000-row collection: green.
4. First-screen gzip: classic 149.9 kB vs astryx 429.2 kB (Δ +279.3 kB);
   reproducible via `scripts/measure-first-screen.mjs`. Entry-level boot
   dynamics are counted on both sides.
5. Swizzles: 0 — DetailPanel composes `Dialog` per ADR-0002.
6. Density: tokens and rendered metrics verified; two drifts found and fixed
   (table `density="balanced"` → ~47px rows; `adaptations` rule lifts narrow
   shell controls to 44px under 861px). Three classic metrics have no astryx
   counterpart: `--control-lg` 42px, `--setting-control-height` 26px,
   `--text-label-xs` 10.5px.
7. i18n: 8,132 ICU messages clean; astryx catalogs 370 keys × 3 locales with
   English fallback for upstream lag (WARN, not MISSING/FORMAT).

Final-review fixes landed alongside (route-flag navigation handoff, sparse
`validateSearch`, vue-router-parity codec, `resetScroll` on query-only navs,
`Vary: Cookie`, shared-util dedup, DetailPanel nested-overlay focus
exemption). Suites at close: astryx 53/53, classic 58/58, CSP 15/15, codec
12/12, contracts green.

### Phase 2: Low-coupling domains

`settings` (3.3k), `model-prices` (1.1k, embedded in two routes), `models`
(4.7k), `home` (4.7k). Parity checklists: `db/features/model-test-alias.md`,
`db/features/usage-timing-metrics.md`.

**Phase 2 outcome (shipped on `docs/astryx-migration-plan`):** `settings`,
`models` (model-prices embedded), and `home` are migrated; `page_routes.json`
carries `astryx: true` on `/login`, `/`, `/groups`, `/logs`, `/models`, and
`/settings`. Classic `model dialogs` bound to group-detail/monitor stay out of
the `/models` slice per the domain boundary. Evidence: astryx e2e 79/79,
classic e2e 58/58, go-csp 14/14 against a real WSL2 linux binary, ICU 8132
messages, tsc×3 + eslint clean. Three conventions landed here become binding
precedent for Phases 3–4:

- **React Compiler + mutable controllers:** shared controllers must publish a
  memoized snapshot consumed via `useSyncExternalStore`; render code reads
  `snapshot.*` and never calls `controller.get*()` — the compiler can freeze
  stable-reference method calls (root-caused in the models drawer).
- **Vue watch → render-phase adjustment:** cross-effect state mirrors
  (selection loss, config edits, route changes) use React render-phase
  reconciliation plus post-commit effects, not `setState` inside `useEffect`.
- **Sensitive-operation state machines live in shared controllers:** reveal/
  copy/quick-import identity guards, aborts, and feedback timers are
  framework-free (`gateway-actions`), with DOM-bound work injected as ops.

### Phase 3: Operational domains

`access-keys` (4.9k), then `monitor` together with its three routes
(`monitor`, `schedule`, `logs`; 14k). `LogsTab`, `InspectorTab`,
`SchedulePanelDetail`, and `LogDetailDrawer` are the four largest monitor
files. Parity checklists: `db/features/monitor-navigation-shortcuts.md`,
`db/features/dispatch-reasoning-policy.md`, plus the
`request-log-*`/`schedule-*` e2e specs.

### Phase 4: Highest-coupling domains

`groups` (9.6k; the credentials and subscription account flows are the
largest files in the repo) and `import` (6.2k, which owns the import
re-authentication recovery through sessionStorage). These two share
credential staging logic, so migrate them in adjacent releases.

### Phase 5: Cutover

The default flips to astryx, with classic still reachable through the cookie
for one release. Exit criteria: no open parity issues, and both Playwright
projects are green on the final release.

### Phase 6: Deletion

Delete `frontends/classic`, Tailwind, `@vitejs/plugin-vue`, `vue-tsc`, the Vue
ESLint configs, `vue-demi` in `allowBuilds`, the Go selection code, the
manifest flag, the dev selector plugin, and the classic Playwright project.
Update the tech docs' `code.paths` and this document's status.

### Per-domain definition of done

- Route flagged `astryx: true`, and the e2e specs for the domain pass in both
  Playwright projects.
- The `.docs/db` acceptance workflows for the domain are checked manually in
  zh-CN, en-US, and ja-JP, in light and dark, at narrow and wide widths.
- No classic-only logic left: whatever both frontends need lives in `shared`.
- Tech docs whose `code.paths` point at the domain now reference the new
  files.
- **Feature-freeze rule during coexistence:** before a domain migrates, new
  features land only in classic. After it migrates, they land only in the new
  frontend. Bug fixes land in both while the domain is still reachable in
  classic.

## Verification

| Gate | Command / check | When |
| --- | --- | --- |
| Type, build | `pnpm --dir web run build` (runs `type-check`) | every change |
| Lint, format | `pnpm --dir web run lint`, `pnpm --dir web run format` | every change |
| Shared logic | `pnpm --dir web run verify:group-collection`, `verify:health-projection`, `verify:request-log-affinity`, and the other `verify-*.mjs` scripts; `test:connection-json`, `test:channel-contract` | every change touching `shared` |
| E2E (dev) | `pnpm --dir web run e2e:request-log-affinity` and the full Playwright suite, both projects | per domain |
| E2E (Go, CSP) | `make build`, start the binary, run the CSP Playwright project | per phase, and before cutover |
| Theme artifacts | rebuild `gptload.theme.ts` and diff | every change touching the theme |
| App catalog ICU syntax and key parity | `verify:i18n-icu` (folded into `lint`) | every change touching `shared/i18n` |
| Astryx i18n parity | `verify:astryx-i18n` | every Astryx upgrade |
| Minimum browser | Playwright Chromium 125 project | phase boundaries, before cutover |
| Go contracts | `make test` (includes `internal/webui` workflow, page route, and server tests) | every change touching `internal/webui`, `web/package.json`, CI |
| Docs | `pnpm --dir web run docs:check`, `pnpm --dir web run docs:build` | each domain cutover |

## Risks

| Risk | Evidence | Mitigation |
| --- | --- | --- |
| Astryx churn | Beta; 8 releases from 2026-08-29 to 2026-09-23, including a minor bump (0.5.4 → 0.6.0) three days after the previous patch; 231 open issues | Exact pins and lockstep core/theme upgrades at phase boundaries only; `astryx upgrade --apply` codemods; keep swizzles minimal and recorded |
| Browser floor | Policy is Chromium 125+, which is Astryx Tier 1 on Chromium, so anchor positioning is available. Safari and Firefox are best-effort and not release-gating: on Tier 2 versions `Tooltip`, `Popover`, and `Selector` dropdowns render unpositioned, and their classic counterparts have 69 import sites | `build.target`/`browserslist` pinned to Chrome 125; Chromium 125 Playwright project; state the policy in the README when cutover ships |
| React Compiler miscompiles or bails out | Compiler 1.0 plus a second Babel pass (StyleX) on the same files | Compiler lint rules as errors; spike (d); `"use no memo"` only with a linked upstream issue |
| Density drift from Astryx defaults | Astryx metrics are larger than classic's | Density applied at theme and root level only; ±1px gate in Phase 1; recheck after each Astryx upgrade |
| i18n syntax divergence during coexistence | vue-i18n and ICU read the same catalogs | Phase 0 rewrite of the 12 incompatible messages; `verify:i18n-icu` in `lint` |
| CSP breaks only in production | Runtime theme injection; Vite dev has no CSP | Built themes only; Go-served CSP Playwright project |
| Behavior drift in large rewrites | Four files over 40 KB each in monitor, two over 60 KB in groups/import | Phase 0 extraction of pure logic; dual-project e2e; `.docs/db` workflows as checklists |
| Effect-driven bugs | 138 `watch` calls | [Translation rules](#vue--react-translation-rules); StrictMode; the `react-hooks` lint rules |
| Duplicate work during coexistence | Two frontends reachable for months | Feature-freeze rule; short per-domain windows |
| Missing localized Astryx strings | Astryx ships English only | Local catalogs + `verify:astryx-i18n` |
| Supply chain | Many new packages; Astryx publishes often | 7-day rule, exact pins, frozen lockfile, Dependabot |
| React lock-in | The React 19 floor comes from Astryx | Accepted in the ADR; shared logic stays framework-free |
| Doc drift | 4 tech docs reference classic paths | Update at each domain cutover; `docs:check` |

## Open Questions

None. The browser policy, density target, router, React Compiler, i18n
library, and theme base were all decided on 2026-09-25; see
[Confirmed choices](#confirmed-choices-2026-09-25). The remaining unknowns
are empirical and are settled by the Phase 1 go/no-go gate, not by further
decisions.
