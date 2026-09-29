import { Button } from '@astryxdesign/core/Button'
import * as stylex from '@stylexjs/stylex'
import { useLocation, useRouter } from '@tanstack/react-router'

import { decodedPathSegments } from '@shared/routing/safe-redirect'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../i18n'
import { RouteLink } from '../route-link'

const narrow = '@media (max-width: 860px)'
const small = '@media (max-width: 760px)'

const styles = stylex.create({
  frame: {
    padding: {
      default:
        'var(--stage-padding-top) var(--stage-padding-inline) var(--stage-padding-bottom)',
      [narrow]:
        'var(--stage-padding-top-compact) var(--stage-padding-inline-compact) var(--stage-padding-bottom-compact)',
    },
  },
  sheet: {
    display: 'grid',
    width: 'min(100%, var(--page-max))',
    marginLeft: 'auto',
    marginRight: 'auto',
    minHeight:
      'calc(100vh - var(--topbar-height) - var(--stage-padding-top) - var(--stage-padding-bottom))',
    gridTemplateColumns: { default: '230px minmax(0, 1fr)', [small]: '1fr' },
    gridTemplateRows: { default: null, [small]: 'auto minmax(0, 1fr)' },
    alignItems: 'stretch',
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-page, 10px)',
    backgroundColor: 'var(--color-surface)',
  },
  code: {
    display: 'flex',
    height: { default: '100%', [small]: 'auto' },
    minHeight: { default: null, [small]: '140px' },
    flexDirection: 'column',
    justifyContent: 'space-between',
    borderRightWidth: { default: '1px', [small]: 0 },
    borderRightStyle: 'solid',
    borderRightColor: 'var(--color-border-subtle)',
    borderBottomWidth: { default: null, [small]: '1px' },
    borderBottomStyle: { default: null, [small]: 'solid' },
    borderBottomColor: { default: null, [small]: 'var(--color-border-subtle)' },
    backgroundColor: 'var(--color-surface-sunken)',
    padding: { default: '30px', [small]: '22px' },
  },
  codeNumber: {
    fontFamily: 'var(--font-serif)',
    fontSize: { default: '72px', [small]: '48px' },
    fontWeight: 500,
    letterSpacing: '-0.07em',
    lineHeight: 1,
  },
  codeMeta: {
    margin: '12px 0 0',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 'var(--line-relaxed)',
    display: { default: null, [small]: 'none' },
  },
  codeFooter: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: '9.5px',
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
  },
  content: {
    display: 'flex',
    width: '100%',
    maxWidth: { default: '650px', [small]: 'none' },
    flexDirection: 'column',
    justifyContent: 'center',
    padding: { default: '48px 56px', [small]: '32px 22px' },
  },
  eyebrow: {
    margin: '0 0 6px',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    letterSpacing: '0.075em',
    textTransform: 'uppercase',
  },
  title: {
    maxWidth: 'none',
    margin: 0,
    fontSize: '24px',
    fontWeight: 680,
    letterSpacing: '-0.025em',
    lineHeight: 1.25,
  },
  description: {
    maxWidth: '560px',
    margin: '10px 0 0',
    color: 'var(--color-text-muted)',
    lineHeight: 1.75,
  },
  path: {
    display: 'flex',
    maxWidth: '100%',
    alignItems: 'center',
    gap: '9px',
    marginTop: '20px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control, 7px)',
    backgroundColor: 'var(--color-surface-sunken)',
    padding: '10px 12px',
  },
  pathLabel: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    whiteSpace: 'nowrap',
  },
  pathValue: {
    minWidth: 0,
    overflow: 'hidden',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  actions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    marginTop: 'var(--space-6)',
  },
  homeLink: {
    display: 'inline-flex',
    height: 'var(--control-compact)',
    alignItems: 'center',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-action)',
    borderRadius: 'var(--radius-control, 7px)',
    backgroundColor: 'var(--color-action)',
    color: 'var(--color-action-ink)',
    padding: '0 12px',
    fontSize: 'var(--text-meta)',
    fontWeight: 560,
    textDecoration: 'none',
  },
})

export function NotFoundView() {
  const router = useRouter()
  const t = useT()
  const pathname = useLocation({ select: (location) => location.pathname })
  const decoded = decodedPathSegments(pathname).join('/')
  const displayPath = `/${decoded}`

  function goBack(): void {
    if (window.history.length > 1) {
      router.history.back()
      return
    }
    // Home is astryx-owned — stay in the SPA.
    void router.navigate({ to: pagePath('home') })
  }

  return (
    <div {...stylex.props(styles.frame)}>
      <article {...stylex.props(styles.sheet)}>
        <aside {...stylex.props(styles.code)}>
          <div>
            <strong {...stylex.props(styles.codeNumber)}>404</strong>
            <p {...stylex.props(styles.codeMeta)}>
              ROUTE_NOT_FOUND
              <br />
              MANAGEMENT_PLANE
            </p>
          </div>
          <span {...stylex.props(styles.codeFooter)}>{t('common.appName')} · Console</span>
        </aside>

        <section {...stylex.props(styles.content)} aria-labelledby="not-found-title">
          <p {...stylex.props(styles.eyebrow)}>{t('notFound.eyebrow')}</p>
          <h1 id="not-found-title" tabIndex={-1} {...stylex.props(styles.title)}>
            {t('notFound.title')}
          </h1>
          <p {...stylex.props(styles.description)}>{t('notFound.description')}</p>

          <div {...stylex.props(styles.path)}>
            <span {...stylex.props(styles.pathLabel)}>{t('notFound.requestedPath')}</span>
            <code {...stylex.props(styles.pathValue)} title={displayPath}>
              {displayPath}
            </code>
          </div>

          <div {...stylex.props(styles.actions)}>
            <RouteLink to={pagePath('home')} {...stylex.props(styles.homeLink)}>
              {t('notFound.backHome')}
            </RouteLink>
            <Button variant="secondary" label={t('notFound.backPrevious')} onClick={goBack} />
          </div>
        </section>
      </article>
    </div>
  )
}
