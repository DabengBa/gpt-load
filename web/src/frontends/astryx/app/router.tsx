import * as stylex from '@stylexjs/stylex'
import {
  Outlet,
  createRootRouteWithContext,
  createRoute,
  createRouter,
  notFound,
  redirect,
  useBlocker,
  useRouter,
  useRouterState,
} from '@tanstack/react-router'
import { useEffect } from 'react'

import { pageRouteEntries } from '@shared/routing/page-routes'
import { pageRouteMetaFor, type PageRouteMeta } from '@shared/routing/route-meta'
import { sharedPageRouteNames } from '@shared/routing/route-names'

import { useT } from './i18n'
import { astryxRoutePaths } from './route-adapter'
import { RoutePageStub } from './pages'
import { useAppServices, type AppServices } from './services'
import { AuthedShell, PublicShell } from './shell/Shells'
import { LoginView } from './shell/LoginView'
import { NotFoundView } from './shell/NotFoundView'

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
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  useEffect(() => {
    const [, , foundRoute] = router.getMatchedRoutes(pathname)
    const staticData = foundRoute?.options.staticData as
      | { meta?: PageRouteMeta }
      | undefined
    const titleKey = staticData?.meta?.titleKey
    document.title = titleKey === undefined ? 'GPT-Load' : `${t(titleKey)} · GPT-Load`
  }, [pathname, router, t])
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
        | { pageName?: string; meta?: PageRouteMeta }
        | undefined,
      // Match-level notFound covers throws from any beforeLoad; _notFound
      // marks the boundary match for paths that never matched a route.
      isNotFound: state.matches.some(
        (match) =>
          match.status === 'notFound' ||
          (match as { _notFound?: boolean })._notFound === true,
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
    </div>
  ),
  notFoundComponent: NotFoundView,
})

function loginSearch(
  search: Record<string, unknown>,
): { redirect?: string; help?: 'auth' } {
  return {
    redirect: typeof search.redirect === 'string' ? search.redirect : undefined,
    help: search.help === 'auth' ? 'auth' : undefined,
  }
}

const pageRoutes = astryxRoutePaths(pageRouteEntries).map(({ name, path }) => {
  const meta = pageRouteMetaFor(name)
  return createRoute({
    getParentRoute: () => rootRoute,
    path,
    staticData: { pageName: name, meta },
    validateSearch: name === sharedPageRouteNames.login ? loginSearch : undefined,
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
    component:
      name === sharedPageRouteNames.login ? LoginView : () => <RoutePageStub name={name} />,
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
    defaultNotFoundComponent: NotFoundView,
  })
}

export type AppRouter = ReturnType<typeof createAppRouter>

declare module '@tanstack/react-router' {
  interface Register {
    router: AppRouter
  }
}

// Bridges the shared unsaved-changes controller onto TanStack's blocker;
// the confirmation dialog itself lands with the B9 shell.
export function useUnsavedGuard(active: boolean): void {
  const { unsavedChanges } = useAppServices()
  useBlocker({
    shouldBlockFn: () => {
      if (!active || unsavedChanges.consumeBypass()) return false
      return unsavedChanges.requestConfirmation()
    },
    enableBeforeUnload: active,
  })
}
