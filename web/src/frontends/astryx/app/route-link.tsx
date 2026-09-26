import { Link } from '@tanstack/react-router'
import type { ComponentProps, ReactNode } from 'react'

import { pageRouteEntryForPath } from '@shared/routing/page-routes'

const externalHrefPattern = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i

// Manifest-flagged paths stay in-app navigations. An unflagged path belongs
// to the classic document, so the same URL must be reached with a full
// document navigation for the server-side frontend selection to run —
// rendering a stub for it inside the astryx document would strand the user.
// External or non-path hrefs bypass the router the same way.
export function isAstryxNavigable(href: string): boolean {
  if (externalHrefPattern.test(href)) return false
  const path = href.split(/[?#]/, 1)[0] ?? href
  return pageRouteEntryForPath(path)?.astryx === true
}

export function RouteLink({
  to,
  children,
  ...rest
}: { to: string; children?: ReactNode } & Omit<ComponentProps<'a'>, 'href'>) {
  if (!isAstryxNavigable(to)) {
    return (
      <a href={to} {...rest}>
        {children}
      </a>
    )
  }
  return (
    <Link to={to as never} {...rest}>
      {children}
    </Link>
  )
}
