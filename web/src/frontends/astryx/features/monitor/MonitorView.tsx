import * as stylex from '@stylexjs/stylex'
import { Button, Selector, Tab, TabList } from '@astryxdesign/core'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { ListFilter, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'

import { isTimeRange } from '@shared/lib/time'
import { usageRanges } from '@shared/control/resources/usage'
import { pagePath } from '@shared/routing/page-routes'
import {
  normalizeAccessKeyMonitorQuery,
  normalizeMonitorQuery,
  normalizeMonitorTab,
  parseHealthMonitorState,
  parseUsageMonitorState,
  sameMonitorQuery,
  scopeAccessKeyUsageFilters,
  usageMonitorQuery,
} from '@shared/routing/monitor-route'
import { parseAppliedUsageFilters } from '@shared/domain/monitor/usage-filters'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import type { MessageId } from '@shared/i18n/message-ids'

import { useAppServices } from '../../app/services'
import { useT } from '../../app/i18n'
import { HealthTab, type HealthTabHandle } from './HealthTab'
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
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-title)',
    fontWeight: 650,
  },
  tabRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    minWidth: 0,
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border)',
  },
  usageActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    paddingBottom: 'var(--space-1)',
  },
  filterCount: {
    display: 'inline-grid',
    minWidth: '17px',
    height: '17px',
    placeItems: 'center',
    borderRadius: '999px',
    background: 'var(--color-action-soft)',
    color: 'var(--color-action)',
    paddingInline: '4px',
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
  },
  panel: {
    minWidth: 0,
    paddingTop: {
      default: 'var(--detail-panel-padding-top, var(--space-5))',
      '@media (max-width: 800px)': 'var(--detail-panel-padding-top-compact, var(--space-4))',
    },
  },
  stub: {
    paddingBlock: 'var(--space-6)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
})

export function MonitorView() {
  const t = useT()
  const navigate = useNavigate()
  const { authSession } = useAppServices()
  const sessionState = useSyncExternalStore(authSession.subscribe, authSession.getState)
  const isAccessKey = sessionState.principalType === 'access_key'
  const isAdmin = sessionState.principalType === 'admin'

  const { rawSearch, searchStr } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
    }),
  })
  const monitorPath = pagePath('monitor')

  const canonicalQuery = useMemo(
    () =>
      isAccessKey ? normalizeAccessKeyMonitorQuery(rawSearch) : normalizeMonitorQuery(rawSearch),
    [rawSearch, isAccessKey],
  )
  const activeTab = normalizeMonitorTab(canonicalQuery.tab)
  const isCanonicalQuery = sameMonitorQuery(rawSearch, canonicalQuery)

  // Classic deep watch on route.query (immediate): canonicalize by replace.
  useEffect(() => {
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

  const usageFilters = useMemo(() => {
    const filters = parseAppliedUsageFilters(rawSearch)
    return isAccessKey ? scopeAccessKeyUsageFilters(filters) : filters
  }, [rawSearch, isAccessKey])
  const usageFilterCount =
    Number(!isAccessKey && usageFilters.group_id !== undefined) +
    Number(!isAccessKey && usageFilters.channel_id !== undefined) +
    Number(!isAccessKey && usageFilters.credential_id !== undefined) +
    Number(usageFilters.upstream_model !== undefined)

  const usageRangeOptions = useMemo(
    () =>
      usageRanges.map((value) => ({
        value,
        label: t(`monitor.usage.filters.ranges.${value}` as MessageId),
      })),
    [t],
  )

  const tabs = isAdmin ? (['health', 'usage', 'inspector'] as const) : (['usage'] as const)

  function selectTab(value: string): void {
    const tab = normalizeMonitorTab(value)
    if (tab === activeTab) return
    if (isAccessKey && tab !== 'usage') return
    void navigate({ to: monitorPath, search: { tab }, resetScroll: false })
  }

  const healthTabRef = useRef<HealthTabHandle | null>(null)
  const usageTabRef = useRef<UsageTabHandle | null>(null)
  const [healthRefreshPending, setHealthRefreshPending] = useState(false)
  const [usageRefreshPending, setUsageRefreshPending] = useState(false)

  async function refreshHealth(): Promise<void> {
    if (!healthTabRef.current || healthRefreshPending) return
    setHealthRefreshPending(true)
    try {
      await healthTabRef.current.refresh()
    } finally {
      setHealthRefreshPending(false)
    }
  }

  async function refreshUsage(): Promise<void> {
    if (!usageTabRef.current || usageRefreshPending) return
    setUsageRefreshPending(true)
    try {
      await usageTabRef.current.refresh()
    } finally {
      setUsageRefreshPending(false)
    }
  }

  function selectUsageRange(value: string | number): void {
    if (!isTimeRange(value)) return
    const state = parseUsageMonitorState(rawSearch)
    void navigate({
      to: monitorPath,
      search: usageMonitorQuery(
        { ...usageFilters, range: value },
        { filtersOpen: false, seriesExpanded: false, metric: state.metric },
      ),
      resetScroll: false,
    })
  }

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="monitor-title">
      <div {...stylex.props(styles.sheet)}>
        <div {...stylex.props(styles.headerRow)}>
          <h1 id="monitor-title" {...stylex.props(styles.title)}>
            {t('monitor.title')}
          </h1>
        </div>

        <div {...stylex.props(styles.tabRow)}>
          <TabList
            value={activeTab}
            onChange={selectTab}
            size="sm"
            role="tablist"
            aria-label={t('monitor.tabs.label')}
          >
            {tabs.map((tab) => (
              <Tab key={tab} value={tab} label={t(`monitor.tabs.${tab}` as MessageId)} />
            ))}
          </TabList>

          {activeTab === 'health' && (
            <Button
              variant="secondary"
              size="sm"
              isLoading={healthRefreshPending}
              icon={<RefreshCw size={14} aria-hidden />}
              label={t('monitor.health.refresh')}
              onClick={() => void refreshHealth()}
            />
          )}
          {activeTab === 'usage' && (
            <div {...stylex.props(styles.usageActions)}>
              <Selector
                label={t('monitor.usage.filters.range')}
                isLabelHidden
                variant="ghost"
                size="sm"
                value={usageFilters.range}
                options={usageRangeOptions}
                onChange={selectUsageRange}
              />
              <Button
                variant="secondary"
                size="sm"
                icon={<ListFilter size={14} aria-hidden />}
                label={t('monitor.usage.filters.button')}
                onClick={() => usageTabRef.current?.openFilters()}
              />
              {usageFilterCount > 0 && (
                <span {...stylex.props(styles.filterCount)}>{usageFilterCount}</span>
              )}
              <Button
                variant="secondary"
                size="sm"
                isLoading={usageRefreshPending}
                icon={<RefreshCw size={14} aria-hidden />}
                label={t('monitor.usage.filters.refresh')}
                onClick={() => void refreshUsage()}
              />
            </div>
          )}
        </div>

        {isCanonicalQuery && (
          <div {...stylex.props(styles.panel)} role="tabpanel">
            {activeTab === 'health' && (
              <HealthTab
                handleRef={healthTabRef}
                groupsExpanded={parseHealthMonitorState(canonicalQuery).groupsExpanded}
              />
            )}
            {activeTab === 'usage' && <UsageTab handleRef={usageTabRef} />}
            {activeTab === 'inspector' && (
              // InspectorTab lands in Task D.
              <div {...stylex.props(styles.stub)}>{t('monitor.tabs.inspector')}</div>
            )}
          </div>
        )}
      </div>
    </section>
  )
}
