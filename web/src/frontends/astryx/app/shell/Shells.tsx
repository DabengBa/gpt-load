import * as stylex from '@stylexjs/stylex'
import { useRouter } from '@tanstack/react-router'
import {
  Activity,
  Check,
  FileText,
  Home,
  KeyRound,
  LockKeyhole,
  Megaphone,
  Settings,
  SlidersHorizontal,
  UsersRound,
  type LucideIcon,
} from 'lucide-react'
import { useSyncExternalStore, type CSSProperties, type ReactNode } from 'react'

import type { MessageId } from '@shared/i18n/message-ids'
import { pagePath } from '@shared/routing/page-routes'
import type { PageRouteMeta } from '@shared/routing/route-meta'

import { useT } from '../i18n'
import { RouteLink } from '../route-link'
import { useAppServices } from '../services'
import { AuthGate } from './AuthGate'
import { BrandMark } from './BrandMark'
import { PreferencesControl } from './PreferencesControl'

const narrow = '@media (max-width: 860px)'

const styles = stylex.create({
  skipLink: {
    position: 'absolute',
    top: { default: '-40px', ':focus': 0 },
    left: '8px',
    zIndex: 20, // --z-sticky (stylex requires a numeric literal)
    backgroundColor: 'var(--color-action)',
    color: 'var(--color-action-ink)',
    padding: '6px 10px',
    borderRadius: '0 0 var(--radius-control, 7px) var(--radius-control, 7px)',
    fontSize: 'var(--text-meta)',
    textDecoration: 'none',
  },
  topbar: {
    position: 'sticky',
    zIndex: 20, // --z-sticky (stylex requires a numeric literal)
    top: 0,
    display: 'flex',
    height: 'var(--topbar-height)',
    alignItems: 'center',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    backgroundColor: 'var(--color-surface)',
    paddingLeft: { default: 'var(--topbar-padding-inline)', [narrow]: 'var(--space-4)' },
    paddingRight: { default: 'var(--topbar-padding-inline)', [narrow]: 'var(--space-4)' },
  },
  topbarInner: {
    display: 'flex',
    width: '100%',
    height: '100%',
    alignItems: 'center',
    gap: { default: '28px', [narrow]: 'var(--space-2)' },
  },
  brand: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '9px',
    color: 'var(--color-text)',
    fontFamily: 'var(--font-serif)',
    fontSize: 'var(--title-section, 16px)',
    fontWeight: 650,
    letterSpacing: '-0.01em',
    whiteSpace: 'nowrap',
    textDecoration: 'none',
  },
  sidebar: {
    display: { default: 'flex', [narrow]: 'none' },
    position: 'fixed',
    top: 0,
    bottom: 0,
    left: 0,
    width: '216px',
    overflowY: 'auto',
    flexDirection: 'column',
    backgroundColor: 'var(--color-surface-sunken)',
    borderRightWidth: '1px',
    borderRightStyle: 'solid',
    borderRightColor: 'var(--color-border-subtle)',
    padding: '24px 14px',
  },
  sidebarBrand: {
    padding: '0 8px',
    minHeight: '28px',
    marginBottom: '32px',
  },
  workspace: {
    marginLeft: { default: '216px', [narrow]: 0 },
    minWidth: 0,
  },

  mobileBrand: {
    display: { default: 'none', [narrow]: 'inline-flex' },
  },
  currentPage: {
    display: { default: 'block', [narrow]: 'none' },
    color: 'var(--color-text)',
    fontSize: '14px',
    fontWeight: 600,
  },
  desktopNav: {
    display: { default: 'flex', [narrow]: 'none' },
    flexDirection: 'column',
    gap: '4px',
  },
  navLink: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    minHeight: '42px',
    borderRadius: 'var(--radius-control)',
    color: { default: 'var(--color-text-muted)', ':hover': 'var(--color-text)' },
    backgroundColor: { default: 'transparent', ':hover': 'var(--color-interactive-hover)' },
    padding: '8px 12px',
    fontSize: '13px',
    fontWeight: 400,
    textDecoration: 'none',
  },
  navLinkActive: {
    color: { default: 'var(--color-action-ink)', ':hover': 'var(--color-action-ink)' },
    backgroundColor: { default: 'var(--color-action)', ':hover': 'var(--color-action-hover)' },
    fontWeight: 560,
  },
  badge: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    gap: '5px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control, rgba(0,0,0,0.16))',
    borderRadius: 'var(--radius-tag, 7px)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    padding: '3px 7px',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 560,
    lineHeight: 1.2,
    whiteSpace: 'nowrap',
  },
  actions: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    marginLeft: 'auto',
  },
  importAction: {
    display: 'inline-flex',
    height: 'var(--control-compact)',
    minHeight: 'var(--control-compact)',
    alignItems: 'center',
    gap: '6px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-action)',
    borderRadius: 'var(--radius-control, 7px)',
    backgroundColor: 'var(--color-action)',
    color: 'var(--color-action-ink)',
    padding: { default: '0 9px', [narrow]: 0 },
    fontSize: 'var(--text-meta)',
    fontWeight: 560,
    textDecoration: 'none',
    width: { default: null, [narrow]: 'var(--touch-target)' },
    justifyContent: { default: null, [narrow]: 'center' },
  },
  importLabel: {
    letterSpacing: '0.01em',
    display: { default: null, [narrow]: 'none' },
  },
  content: {
    minHeight: 'calc(100vh - var(--topbar-height))',
    outlineStyle: 'none',
  },
  workspaceContent: {
    maxWidth: 'var(--content-max)',
    marginInline: 'auto',
  },
  mobileNav: {
    display: 'grid',
    gap: 'var(--space-1)',
  },
  mobileNavLabel: {
    margin: 0,
    color: 'var(--color-text-faint)',
    padding: '0 2px 2px',
    fontSize: 'var(--text-label-xs)',
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
  },
  mobileNavLink: {
    display: 'flex',
    minHeight: 'var(--touch-target)',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'transparent',
    borderRadius: 'var(--radius-control, 7px)',
    color: { default: 'var(--color-text-muted)', ':hover': 'var(--color-text)' },
    backgroundColor: { default: null, ':hover': 'var(--color-surface-sunken)' },
    padding: '0 10px',
    fontSize: 'var(--text-sm)',
    textDecoration: 'none',
  },
  mobileNavLinkActive: {
    borderColor: 'var(--color-action)',
    backgroundColor: 'var(--color-action)',
    color: 'var(--color-action-ink)',
    fontWeight: 560,
  },
})

interface NavItem {
  key: string
  to: string
  labelKey: MessageId
  icon: LucideIcon
}

function navItems(isAccessKey: boolean): readonly NavItem[] {
  // Schedule is the single model entry for both principals; the retired
  // models page no longer gets a nav slot.
  const shared: NavItem[] = [
    { key: 'home', to: pagePath('home'), labelKey: 'shell.home', icon: Home },
    {
      key: 'schedule',
      to: pagePath('schedule'),
      labelKey: 'shell.schedule',
      icon: SlidersHorizontal,
    },
    { key: 'monitor', to: pagePath('monitor'), labelKey: 'shell.monitor', icon: Activity },
    { key: 'logs', to: pagePath('logs'), labelKey: 'shell.logs', icon: FileText },
  ]
  if (isAccessKey) return shared
  return [
    shared[0]!,
    { key: 'groups', to: pagePath('groups'), labelKey: 'shell.groups', icon: UsersRound },
    {
      key: 'announcements',
      to: pagePath('announcements'),
      labelKey: 'nav.announcements',
      icon: Megaphone,
    },
    shared[1]!,
    shared[2]!,
    shared[3]!,
    { key: 'settings', to: pagePath('settings'), labelKey: 'shell.settings', icon: Settings },
  ]
}

function MobileNavItems({
  items,
  activeKey,
  onNavigate,
}: {
  items: readonly NavItem[]
  activeKey?: string
  onNavigate?(): void
}) {
  const t = useT()
  return (
    <nav {...stylex.props(styles.mobileNav)} aria-label={t('shell.primaryNavigation')}>
      <p {...stylex.props(styles.mobileNavLabel)}>{t('shell.primaryNavigation')}</p>
      {items.map((item) => {
        const active = item.key === activeKey
        return (
          <RouteLink
            key={item.key}
            to={item.to}
            aria-current={active ? 'page' : undefined}
            onClick={onNavigate}
            {...stylex.props(styles.mobileNavLink, active && styles.mobileNavLinkActive)}
          >
            <span>{t(item.labelKey)}</span>
            {active && <Check size={15} aria-hidden />}
          </RouteLink>
        )
      })}
    </nav>
  )
}

// Authenticated application frame: AuthGate wraps the topbar shell and only
// releases content for a validated session. Mirrors classic App.vue's
// AuthGate > AppShell composition.
export function AuthedShell({
  pageName,
  meta,
  children,
}: {
  pageName?: string
  meta?: PageRouteMeta
  children: ReactNode
}) {
  const services = useAppServices()
  const session = services.authSession
  const state = useSyncExternalStore(session.subscribe, session.getState)
  const router = useRouter()
  const t = useT()
  const isAccessKey = state.principalType === 'access_key'
  const items = navItems(isAccessKey)
  const currentItem = items.find((item) => item.key === meta?.primaryNav)

  async function logout(): Promise<void> {
    const recovery = services.importRecovery
    const unsavedChanges = services.unsavedChanges
    if (pageName === 'import') {
      recovery.clear()
      unsavedChanges.bypassNext()
      session.clear()
      try {
        await router.navigate({ href: pagePath('login'), replace: true })
      } finally {
        unsavedChanges.consumeBypass()
      }
      return
    }
    await router.navigate({ href: pagePath('login'), replace: true })
    recovery.clear()
    session.clear()
  }

  return (
    <AuthGate adminOnly={meta?.adminOnly === true}>
      <a {...stylex.props(styles.skipLink)} href="#main-content">
        {t('shell.skip')}
      </a>
      <aside data-testid="desktop-sidebar" {...stylex.props(styles.sidebar)}>
        <RouteLink
          to={pagePath('home')}
          aria-label={`${t('common.appName')} · ${t('shell.home')}`}
          {...stylex.props(styles.brand, styles.sidebarBrand)}
        >
          <BrandMark size={24} />
          <span>{t('common.appName')}</span>
        </RouteLink>

        <nav
          data-testid="desktop-nav"
          aria-label={t('shell.primaryNavigation')}
          {...stylex.props(styles.desktopNav)}
        >
          {items.map((item) => {
            const active = meta?.primaryNav === item.key
            return (
              <RouteLink
                key={item.key}
                to={item.to}
                aria-current={active ? 'page' : undefined}
                {...stylex.props(styles.navLink, active && styles.navLinkActive)}
              >
                <item.icon size={17} aria-hidden />
                {t(item.labelKey)}
              </RouteLink>
            )
          })}
        </nav>
      </aside>
      <div
        {...stylex.props(styles.workspace)}
        style={{ '--topbar-height': '68px' } as CSSProperties}
      >
        <header {...stylex.props(styles.topbar)}>
          <div {...stylex.props(styles.topbarInner)}>
            <RouteLink
              to={pagePath('home')}
              aria-label={`${t('common.appName')} · ${t('shell.home')}`}
              {...stylex.props(styles.brand, styles.mobileBrand)}
            >
              <BrandMark size={24} />
              <span>{t('common.appName')}</span>
            </RouteLink>
            {currentItem && (
              <span {...stylex.props(styles.currentPage)}>{t(currentItem.labelKey)}</span>
            )}
            {isAccessKey && (
              <span {...stylex.props(styles.badge)} title={t('shell.accessKeyReadOnlyDescription')}>
                <LockKeyhole size={13} aria-hidden />
                <span>{t('shell.accessKeyReadOnly')}</span>
              </span>
            )}

            <div {...stylex.props(styles.actions)}>
              {!isAccessKey && (
                <RouteLink
                  to={pagePath('import')}
                  aria-label={t('shell.import')}
                  {...stylex.props(styles.importAction)}
                >
                  <KeyRound size={15} aria-hidden />
                  <span {...stylex.props(styles.importLabel)}>{t('shell.import')}</span>
                </RouteLink>
              )}
              <PreferencesControl
                triggerLabel={t('shell.menu')}
                showSignOut
                mobileNav={<MobileNavItems items={items} activeKey={meta?.primaryNav} />}
                onSignOut={() => void logout()}
              />
            </div>
          </div>
        </header>
        <main
          id="main-content"
          tabIndex={-1}
          {...stylex.props(styles.content, styles.workspaceContent)}
        >
          {children}
        </main>
      </div>
    </AuthGate>
  )
}

// Public frame for unauthenticated routes (login, not-found): brand + compact
// preferences, no navigation.
export function PublicShell({ children }: { children: ReactNode }) {
  const t = useT()
  return (
    <>
      <a {...stylex.props(styles.skipLink)} href="#main-content">
        {t('shell.skip')}
      </a>
      <header {...stylex.props(styles.topbar)}>
        <RouteLink
          to={pagePath('home')}
          aria-label={`${t('common.appName')} · ${t('shell.home')}`}
          {...stylex.props(styles.brand)}
        >
          <BrandMark size={24} />
          <span>{t('common.appName')}</span>
        </RouteLink>
        <div {...stylex.props(styles.actions)}>
          <PreferencesControl />
        </div>
      </header>
      <div id="main-content" tabIndex={-1} {...stylex.props(styles.content)}>
        {children}
      </div>
    </>
  )
}
