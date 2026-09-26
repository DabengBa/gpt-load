import { Button } from '@astryxdesign/core/Button'
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

import { getBrowserLocale } from '@shared/preferences/locale'
import { pageRouteEntries } from '@shared/routing/page-routes'
import { pageRouteMetaFor, type PageRouteMeta } from '@shared/routing/route-meta'
import { sharedPageRouteNames } from '@shared/routing/route-names'

import { astryxRoutePaths } from './route-adapter'
import { LoginPageStub, NotFoundPageStub, RoutePageStub } from './pages'
import { useAppServices, type AppServices } from './services'

interface RouterContext {
  services: AppServices
}

const shellStyles = stylex.create({
  shell: {
    display: 'grid',
    minHeight: '100vh',
    gridTemplateRows: 'auto 1fr',
    backgroundColor: 'var(--color-background-body)',
    color: 'var(--color-text-primary)',
    fontFamily: 'var(--font-family-body)',
    fontSize: 'var(--text-body-size)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 16px',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle, rgba(0,0,0,0.08))',
  },
  outlet: {
    padding: '16px',
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

// Reads the matched route's staticData meta and keeps document.title plus
// <html lang> in sync; react-intl lands in B8 and will translate titleKey.
function HeadSync() {
  const router = useRouter()
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  useEffect(() => {
    const [, , foundRoute] = router.getMatchedRoutes(pathname)
    const staticData = foundRoute?.options.staticData as
      | { meta?: PageRouteMeta }
      | undefined
    const titleKey = staticData?.meta?.titleKey
    document.title = titleKey === undefined ? 'GPT-Load' : `${titleKey} · GPT-Load`
    document.documentElement.lang = getBrowserLocale()
  }, [pathname, router])
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

const rootRoute = createRootRouteWithContext<RouterContext>()({
  component: () => (
    <div {...stylex.props(shellStyles.shell)} data-testid="astryx-shell">
      <HeadSync />
      <RouteAnnouncer />
      <header {...stylex.props(shellStyles.header)}>
        <Button label="GPT-Load" />
        <Button label="Override" xstyle={probeStyles.overrideProbe} />
      </header>
      <div {...stylex.props(shellStyles.outlet)}>
        <Outlet />
      </div>
    </div>
  ),
  notFoundComponent: NotFoundPageStub,
})

const probeStyles = stylex.create({
  overrideProbe: {
    borderRadius: '2px',
  },
})

function loginSearch(search: Record<string, unknown>): { redirect?: string } {
  return { redirect: typeof search.redirect === 'string' ? search.redirect : undefined }
}

const pageRoutes = astryxRoutePaths(pageRouteEntries).map(({ name, path }) => {
  const meta = pageRouteMetaFor(name)
  return createRoute({
    getParentRoute: () => rootRoute,
    path,
    staticData: { pageName: name, meta },
    validateSearch: name === sharedPageRouteNames.login ? loginSearch : undefined,
    beforeLoad: ({ context, location }) => {
      // TanStack's 'preserve' still matches '/x/' to '/x'; the classic router
      // (strict:true) rejects trailing slashes — keep parity via an explicit
      // canonical-path check before the auth guards.
      if (location.pathname.length > 1 && location.pathname.endsWith('/')) {
        throw notFound()
      }
      if (meta.adminOnly && context.services.authSession.getPrincipalType() === 'access_key') {
        throw redirect({ href: '/', replace: true })
      }
      if (meta.requiresAuth && !context.services.authSession.hasCredential()) {
        throw redirect({
          href: `/login?redirect=${encodeURIComponent(location.href)}`,
          replace: true,
        })
      }
    },
    component:
      name === sharedPageRouteNames.login
        ? LoginPageStub
        : () => <RoutePageStub name={name} />,
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
    defaultNotFoundComponent: NotFoundPageStub,
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
