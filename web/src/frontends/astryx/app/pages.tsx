import * as stylex from '@stylexjs/stylex'
import { useRouterState } from '@tanstack/react-router'
import { useEffect } from 'react'

import { pageRouteEntryForPath } from '@shared/routing/page-routes'
import { pageRouteMetaFor } from '@shared/routing/route-meta'

import { useT } from './i18n'
import { RouteLink } from './route-link'

const styles = stylex.create({
  page: {
    display: 'grid',
    gap: 'var(--space-4, 16px)',
    padding:
      'var(--stage-padding-top) var(--stage-padding-inline) var(--stage-padding-bottom)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-heading-2-size, 22px)',
    fontWeight: 600,
  },
  meta: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta, 12px)',
  },
})

// Placeholder body for manifest routes whose business pages migrate in later
// phases; the shell (nav/auth/title/i18n) around them is real.
export function RoutePageStub({ name }: { name: string }) {
  const t = useT()
  const titleKey = pageRouteMetaFor(name).titleKey
  const href = useRouterState({ select: (state) => state.location.href })

  // Stubs only exist for astryx-flagged routes. An unflagged path landing here
  // means a missed handoff (e.g. a programmatic navigate that bypassed
  // RouteLink) — recover with the document navigation the click path took.
  const unflagged = pageRouteEntryForPath(href.split(/[?#]/, 1)[0] ?? href)
  useEffect(() => {
    if (unflagged !== undefined && unflagged.astryx !== true) {
      window.location.assign(href)
    }
  }, [unflagged, href])
  if (unflagged !== undefined && unflagged.astryx !== true) return null

  return (
    <div {...stylex.props(styles.page)} data-route={name}>
      <h1 {...stylex.props(styles.title)}>{titleKey === undefined ? name : t(titleKey)}</h1>
      <p {...stylex.props(styles.meta)}>React/Astryx preview shell — this page is a stub.</p>
      <p {...stylex.props(styles.meta)}>
        <RouteLink to="/">{t('notFound.backHome')}</RouteLink>
      </p>
    </div>
  )
}
