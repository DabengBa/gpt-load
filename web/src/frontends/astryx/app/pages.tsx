import * as stylex from '@stylexjs/stylex'
import { Link } from '@tanstack/react-router'

import { pageRouteMetaFor } from '@shared/routing/route-meta'

import { useT } from './i18n'

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
  return (
    <div {...stylex.props(styles.page)} data-route={name}>
      <h1 {...stylex.props(styles.title)}>{titleKey === undefined ? name : t(titleKey)}</h1>
      <p {...stylex.props(styles.meta)}>React/Astryx preview shell — this page is a stub.</p>
      <p {...stylex.props(styles.meta)}>
        <Link to="/">{t('notFound.backHome')}</Link>
      </p>
    </div>
  )
}
