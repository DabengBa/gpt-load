import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'

import { pagePath } from '@shared/routing/page-routes'
import {
  normalizeAccessKeyMonitorQuery,
  normalizeMonitorQuery,
  sameMonitorQuery,
} from '@shared/routing/monitor-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'

import { useAppServices } from '../../app/services'
import { useT } from '../../app/i18n'
import { UsageTab, type UsageTabHandle } from './UsageTab'

const styles = stylex.create({
  page: {
    display: 'grid',
    minWidth: 0,
  },
  sheet: {
    display: 'grid',
    minWidth: 0,
    alignContent: 'start',
    gap: 0,
    minHeight: {
      default: '760px',
      '@media (max-width: 800px)': '0',
    },
  },
  headerRow: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-title)',
    fontWeight: 650,
  },
  usageActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  panel: {
    minWidth: 0,
    paddingTop: {
      default: 'var(--detail-panel-padding-top, var(--space-5))',
      '@media (max-width: 800px)': 'var(--detail-panel-padding-top-compact, var(--space-4))',
    },
  },
})

export function MonitorView() {
  const t = useT()
  const navigate = useNavigate()
  const { authSession } = useAppServices()
  const sessionState = useSyncExternalStore(authSession.subscribe, authSession.getState)
  const isAccessKey = sessionState.principalType === 'access_key'

  const { rawSearch, searchStr, pathname } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
      pathname: state.location.pathname,
    }),
  })
  const monitorPath = pagePath('monitor')

  const canonicalQuery = useMemo(
    () =>
      isAccessKey ? normalizeAccessKeyMonitorQuery(rawSearch) : normalizeMonitorQuery(rawSearch),
    [rawSearch, isAccessKey],
  )
  const isCanonicalQuery = sameMonitorQuery(rawSearch, canonicalQuery)

  // Classic deep watch on route.query (immediate): canonicalize by replace.
  useEffect(() => {
    // Pending transition: the outgoing route still renders while location has
    // moved — a late canonicalization must not resurrect this page.
    if (pathname !== monitorPath) return
    if (!isCanonicalQuery) {
      void navigate({
        to: monitorPath,
        search: canonicalQuery,
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the URL string
  }, [searchStr])

  const usageTabRef = useRef<UsageTabHandle | null>(null)
  const [usageRefreshPending, setUsageRefreshPending] = useState(false)

  async function refreshUsage(): Promise<void> {
    if (!usageTabRef.current || usageRefreshPending) return
    setUsageRefreshPending(true)
    try {
      await usageTabRef.current.refresh()
    } finally {
      setUsageRefreshPending(false)
    }
  }

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="monitor-title">
      <div {...stylex.props(styles.sheet)}>
        <div {...stylex.props(styles.headerRow)}>
          <h1 id="monitor-title" {...stylex.props(styles.title)}>
            {t('monitor.title')}
          </h1>

          <div {...stylex.props(styles.usageActions)}>
            <Button
              variant="secondary"
              size="sm"
              isLoading={usageRefreshPending}
              icon={<RefreshCw size={14} aria-hidden />}
              label={t('monitor.usage.filters.refresh')}
              onClick={() => void refreshUsage()}
            />
          </div>
        </div>

        {isCanonicalQuery && (
          <div {...stylex.props(styles.panel)}>
            <UsageTab handleRef={usageTabRef} />
          </div>
        )}
      </div>
    </section>
  )
}
