import { Link } from '@tanstack/react-router'
import type { ComponentProps, ReactNode } from 'react'

const externalHrefPattern = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i

// Every internal path — manifest route or not — is served by this document
// and handled by the router (unknown paths land on the not-found view), so
// in-app links always stay SPA navigations. External or non-path hrefs keep
// a plain anchor so the browser performs a real document navigation.
export function RouteLink({
  to,
  children,
  ...rest
}: { to: string; children?: ReactNode } & Omit<ComponentProps<'a'>, 'href'>) {
  if (externalHrefPattern.test(to)) {
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
