import type { AnyRouter } from '@tanstack/react-router'

import { sharedPageRouteNames } from '@shared/routing/route-names'
import type { PageRouteMeta } from '@shared/routing/route-meta'
import { safeRedirectTarget, type SafeRedirectResolution } from '@shared/routing/safe-redirect'

// Adapts the shared safeRedirect contract to TanStack matching; blocklist
// mirrors the classic wrapper (login + the not-found catch-all).
export function safeRedirect(router: Pick<AnyRouter, 'getMatchedRoutes'>, raw: unknown): string {
  return safeRedirectTarget(
    raw,
    (value): SafeRedirectResolution => {
      let pathname = value
      try {
        pathname = new URL(value, 'http://localhost').pathname
      } catch {
        pathname = value
      }
      const [matchedRoutes, , foundRoute] = router.getMatchedRoutes(pathname)
      const staticData = (foundRoute?.options.staticData ?? {}) as {
        pageName?: string
        meta?: PageRouteMeta
      }
      return {
        matched: matchedRoutes,
        name: staticData.pageName,
        path: pathname,
        fullPath: value,
        meta: { requiresAuth: staticData.meta?.requiresAuth },
      }
    },
    [sharedPageRouteNames.login, 'not-found'],
  )
}
