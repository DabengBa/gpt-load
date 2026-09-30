import * as stylex from '@stylexjs/stylex'
import {
  Outlet,
  createRootRouteWithContext,
  createRoute,
  createRouter,
  notFound,
  redirect,
  useRouter,
  useRouterState,
} from '@tanstack/react-router'
import { useEffect, type ReactNode } from 'react'

import { pageRouteEntries } from '@shared/routing/page-routes'
import { pageRouteMetaFor, type PageRouteMeta } from '@shared/routing/route-meta'
import { sharedPageRouteNames } from '@shared/routing/route-names'
import {
  parseAccessKeyCollectionRouteQuery,
  parseAccessKeyDrawerRoute,
  serializeAccessKeyCollectionRouteQuery,
} from '@shared/routing/access-key-collection-route'
import {
  parseGroupCollectionRouteQuery,
  serializeGroupCollectionRouteQuery,
} from '@shared/routing/group-collection-route'
import {
  parseCredentialRouteQuery,
  parseCredentialRouteState,
  parseGroupModelsRouteQuery,
  serializeCredentialRouteQuery,
  serializeGroupModelsRouteQuery,
} from '@shared/routing/group-detail-route'
import { scalarRouteQuery } from '@shared/routing/route-query'
import {
  parseImportRouteQuery,
  serializeImportRouteQuery,
} from '@shared/routing/import-route'
import { parseHomeRouteQuery, serializeHomeRouteQuery } from '@shared/routing/home-route'
import { parseModelsRouteQuery, serializeModelsRouteQuery } from '@shared/routing/models-route'
import {
  parseSettingsRouteSection,
  serializeSettingsRouteQuery,
} from '@shared/routing/settings-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import {
  normalizeMonitorQuery,
  parseScheduleMonitorState,
  scheduleMonitorQuery,
} from '@shared/routing/monitor-route'

import { useT } from './i18n'
import { astryxRoutePaths } from './route-adapter'
import { parseSharedRouteSearch, stringifySharedRouteSearch } from './search-codec'

import type { AppServices } from './services'
import { ToastHost } from './ToastHost'
import { AuthedShell, PublicShell } from './shell/Shells'
import { LoginView } from './shell/LoginView'
import { NotFoundView } from './shell/NotFoundView'
import { AccessKeysView } from '../features/access-keys/AccessKeysView'
import { GroupDetailView } from '../features/groups/GroupDetailView'
import { GroupsView } from '../features/groups/GroupsView'
import { ImportView } from '../features/import/ImportView'
import { HomeView } from '../features/home/HomeView'
import { LogsView } from '../features/logs/LogsView'
import { ModelsView } from '../features/models/ModelsView'
import { MonitorView } from '../features/monitor/MonitorView'
import { ScheduleView } from '../features/monitor/ScheduleView'
import { SettingsView } from '../features/settings/SettingsView'

interface RouterContext {
  services: AppServices
}

const shellStyles = stylex.create({
  shell: {
    minHeight: '100vh',
    backgroundColor: 'var(--color-background-body)',
    color: 'var(--color-text-primary)',
    fontFamily: 'var(--font-family-body)',
    fontSize: 'var(--text-body-size)',
  },
  srOnly: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
})

// Reads the matched route's staticData meta and keeps document.title in sync;
// <html lang> is owned by the i18n controller (app/i18n.tsx emit/setLocale).
function HeadSync() {
  const router = useRouter()
  const t = useT()
  const { pathname, isNotFound } = useRouterState({
    select: (state) => ({
      pathname: state.location.pathname,
      isNotFound: state.matches.some(
        (match) =>
          match.status === 'notFound' || (match as { _notFound?: boolean })._notFound === true,
      ),
    }),
  })
  useEffect(() => {
    const [, , foundRoute] = router.getMatchedRoutes(pathname)
    const staticData = foundRoute?.options.staticData as { meta?: PageRouteMeta } | undefined
    const titleKey = staticData?.meta?.titleKey ?? (isNotFound ? 'notFound.title' : undefined)
    document.title = titleKey === undefined ? 'GPT-Load' : `${t(titleKey)} · GPT-Load`
  }, [pathname, isNotFound, router, t])
  return null
}

function RouteAnnouncer() {
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  return (
    <div aria-live="polite" {...stylex.props(shellStyles.srOnly)}>
      {pathname}
    </div>
  )
}

// Picks the frame from the deepest matched route's manifest meta, mirroring
// classic App.vue: requiresAuth -> AuthGate > AppShell, else PublicShell.
// The not-found render carries no staticData, so it lands in PublicShell.
function ShellOutlet() {
  const { meta, isNotFound } = useRouterState({
    select: (state) => ({
      meta: state.matches.at(-1)?.staticData as
        { pageName?: string; meta?: PageRouteMeta } | undefined,
      // Match-level notFound covers throws from any beforeLoad; _notFound
      // marks the boundary match for paths that never matched a route.
      isNotFound: state.matches.some(
        (match) =>
          match.status === 'notFound' || (match as { _notFound?: boolean })._notFound === true,
      ),
    }),
  })
  const routeMeta = meta?.meta
  // A not-found render (thrown in root beforeLoad or an unmatched path) must
  // land in the public frame even when the matched route was guarded —
  // otherwise AuthedShell's AuthGate swallows the 404 view.
  if (!isNotFound && routeMeta?.requiresAuth === true) {
    return (
      <AuthedShell pageName={meta?.pageName} meta={routeMeta}>
        <Outlet />
      </AuthedShell>
    )
  }
  return (
    <PublicShell>
      <Outlet />
    </PublicShell>
  )
}

const rootRoute = createRootRouteWithContext<RouterContext>()({
  beforeLoad: ({ location }) => {
    // TanStack's 'preserve' still matches '/x/' to '/x'; the classic router
    // (strict:true) rejects trailing slashes — keep parity via an explicit
    // canonical-path check. ShellOutlet swaps in the public frame whenever a
    // not-found is active, so this can live on the root.
    if (location.pathname.length > 1 && location.pathname.endsWith('/')) {
      throw notFound()
    }
  },
  component: () => (
    <div {...stylex.props(shellStyles.shell)} data-testid="astryx-shell">
      <HeadSync />
      <RouteAnnouncer />
      <ShellOutlet />
      <ToastHost />
    </div>
  ),
  notFoundComponent: NotFoundView,
})

function loginSearch(search: Record<string, unknown>): { redirect?: string; help?: 'auth' } {
  return {
    redirect: typeof search.redirect === 'string' ? search.redirect : undefined,
    help: search.help === 'auth' ? 'auth' : undefined,
  }
}

// Typed route search: the shared parser owns the groups collection contract.
// validateSearch must return a SPARSE object — TanStack writes the validated
// search back to the URL, so injecting defaults here would turn '/groups' into
// a verbose '?sort=recent&page=1&page_size=100' URL and make the component's
// canonicalization effect fire on every clean load. Sparse keys keep the URL
// untouched; GroupsView still parses defaults from location.search itself.
function groupsSearch(search: Record<string, unknown>) {
  return serializeGroupCollectionRouteQuery(
    parseGroupCollectionRouteQuery(search as SharedRouteQuery),
  )
}

// Sparse like groupsSearch: routing → {}, everything else → { section }.
// Invalid/repeated values canonicalize to the bare '/settings' URL via the
// same mechanism, matching the classic router.replace behavior.
function settingsSearch(search: Record<string, unknown>) {
  return serializeSettingsRouteQuery(parseSettingsRouteSection(search as SharedRouteQuery))
}

// Sparse canonical search: defaults (enabled/all/page 1, no drawer) serialize
// away; junk or duplicated keys normalize out on the write-back.
function modelsSearch(search: Record<string, unknown>) {
  return serializeModelsRouteQuery(parseModelsRouteQuery(search as SharedRouteQuery))
}

// Sparse canonical search: access_key_id only survives when it resolves to a
// number, client only when it's a known gateway client (default 'cc-switch'
// serializes away).
function homeSearch(search: Record<string, unknown>) {
  return serializeHomeRouteQuery(parseHomeRouteQuery(search as SharedRouteQuery))
}

// Sparse canonical search: q/status/page serialize away at defaults; the
// drawer pair (action, access_key_id) only survives a well-formed parse.
function accessKeysSearch(search: Record<string, unknown>) {
  const query = search as SharedRouteQuery
  return serializeAccessKeyCollectionRouteQuery(
    parseAccessKeyCollectionRouteQuery(query),
    parseAccessKeyDrawerRoute(query),
  )
}

// Sparse canonical search: the shared monitor codec owns the three-tab query
// contract (health/usage/inspector + schedule surface). The admin shape is the
// default; the access_key divergence is canonicalized by MonitorView (same as
// the classic deep watch).
function monitorSearch(search: Record<string, unknown>) {
  return normalizeMonitorQuery(search as SharedRouteQuery)
}

// Sparse canonical search: parse → serialize drops unknown params and
// malformed schedule state, matching the classic view-level replace.
function scheduleSearch(search: Record<string, unknown>) {
  return scheduleMonitorQuery(parseScheduleMonitorState(search as SharedRouteQuery))
}

// Sparse canonical search, scoped per tab — this must NOT call the codec's
// normalizeGroupQuery: that helper came from classic GroupTabs.vue (dead code
// never mounted), so feeding it to validateSearch would rewrite the bare
// '/groups/:id' URL to '?tab=credentials' and land users on the management
// tab, while classic renders the unified settings+models view. Classic only
// canonicalizes live for the credentials sub-query (and the models discovery
// params); an absent/unknown tab keeps the raw query and renders unified.
function groupDetailSearch(search: Record<string, unknown>) {
  const query = search as SharedRouteQuery
  const tab = scalarRouteQuery(query.tab)
  if (tab === 'credentials') {
    return serializeCredentialRouteQuery(
      parseCredentialRouteQuery(query),
      parseCredentialRouteState(query),
    )
  }
  if (tab === 'models') {
    return serializeGroupModelsRouteQuery(parseGroupModelsRouteQuery(query))
  }
  if (tab === 'settings') return { tab: 'settings' }
  return {}
}

// Sparse canonical search: mode=existing forces off the new-mode params and
// vice versa; a bare `group_id` implies existing (classic parity).
function importSearch(search: Record<string, unknown>) {
  return serializeImportRouteQuery(parseImportRouteQuery(search as SharedRouteQuery))
}

const searchValidators: Partial<
  Record<string, (search: Record<string, unknown>) => Record<string, unknown>>
> = {
  [sharedPageRouteNames.login]: loginSearch,
  [sharedPageRouteNames.home]: homeSearch,
  [sharedPageRouteNames.accessKeys]: accessKeysSearch,
  [sharedPageRouteNames.groups]: groupsSearch,
  [sharedPageRouteNames.groupDetail]: groupDetailSearch,
  [sharedPageRouteNames.import]: importSearch,
  [sharedPageRouteNames.settings]: settingsSearch,
  [sharedPageRouteNames.models]: modelsSearch,
  [sharedPageRouteNames.monitor]: monitorSearch,
  [sharedPageRouteNames.schedule]: scheduleSearch,
}

type RouteName = (typeof sharedPageRouteNames)[keyof typeof sharedPageRouteNames]

// Every manifest route must map to a real view — the type fails to compile if
// a sharedPageRouteNames entry is missing here (there is no stub fallback).
const routeViews: Record<RouteName, () => ReactNode> = {
  [sharedPageRouteNames.login]: LoginView,
  [sharedPageRouteNames.home]: HomeView,
  [sharedPageRouteNames.accessKeys]: AccessKeysView,
  [sharedPageRouteNames.groups]: GroupsView,
  [sharedPageRouteNames.groupDetail]: GroupDetailView,
  [sharedPageRouteNames.import]: ImportView,
  [sharedPageRouteNames.logs]: LogsView,
  [sharedPageRouteNames.settings]: SettingsView,
  [sharedPageRouteNames.models]: ModelsView,
  [sharedPageRouteNames.monitor]: MonitorView,
  [sharedPageRouteNames.schedule]: ScheduleView,
}

const pageRoutes = astryxRoutePaths(pageRouteEntries).map(({ name, path }) => {
  const meta = pageRouteMetaFor(name)
  return createRoute({
    getParentRoute: () => rootRoute,
    path,
    staticData: { pageName: name, meta },
    validateSearch: searchValidators[name],
    beforeLoad: async ({ context, location }) => {
      if (meta.adminOnly && context.services.authSession.getPrincipalType() === 'access_key') {
        throw redirect({ href: '/', replace: true })
      }
      if (meta.requiresAuth && !context.services.authSession.hasCredential()) {
        throw redirect({
          href: `/login?redirect=${encodeURIComponent(location.href)}`,
          replace: true,
        })
      }
      await context.services.i18n.ensureNamespaces(meta.messageNamespaces ?? [])
    },
    component: routeViews[name as RouteName],
  })
})

const routeTree = rootRoute.addChildren(pageRoutes)

export function createAppRouter(services: AppServices) {
  return createRouter({
    routeTree,
    context: { services },
    trailingSlash: 'preserve',
    caseSensitive: true,
    scrollRestoration: true,
    parseSearch: parseSharedRouteSearch,
    stringifySearch: stringifySharedRouteSearch,
    defaultNotFoundComponent: NotFoundView,
  })
}

export type AppRouter = ReturnType<typeof createAppRouter>

declare module '@tanstack/react-router' {
  interface Register {
    router: AppRouter
  }
}
