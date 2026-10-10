import * as stylex from '@stylexjs/stylex'
import { Banner, Button, EmptyState, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { Database, TriangleAlert } from 'lucide-react'
import { useImperativeHandle, useMemo, useState, useSyncExternalStore, type Ref } from 'react'

import { controlQueryKeys } from '@shared/control/query-keys'
import { listChannels } from '@shared/control/resources/channels'
import { groupOptionsQueryOptions } from '@shared/control/resources/groups'
import {
  normalizeUsageBreakdownSort,
  normalizeUsageBreakdownSortDirection,
  usageQueryOptions,
  type UsageBreakdownSort,
  type UsageBreakdownSortDirection,
  type UsageFilters,
} from '@shared/control/resources/usage'
import {
  applyUsageFilterDraft,
  createUsageFilterDraft,
  parseAppliedUsageFilters,
  validateUsageFilterDraft,
  type UsageFilterDraft,
  type UsageFilterErrors,
} from '@shared/domain/monitor/usage-filters'
import { pagePath } from '@shared/routing/page-routes'
import { scopeAccessKeyUsageFilters, usageMonitorQuery } from '@shared/routing/monitor-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'

import { useAppServices } from '../../app/services'
import { useCollectionLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { MonitorSectionHeading } from './MonitorSectionHeading'
import { UsageBreakdownTable } from './UsageBreakdownTable'
import { UsageFilterBar } from './UsageFilterBar'

const styles = stylex.create({
  root: {
    display: 'grid',
    minWidth: 0,
    gap: '22px',
  },
  refreshing: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  skeleton: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  section: {
    display: 'grid',
    minWidth: 0,
    gap: '12px',
  },
})

export interface UsageTabHandle {
  refresh(): Promise<void>
}

export function UsageTab({ handleRef }: { handleRef?: Ref<UsageTabHandle> }) {
  const t = useT()
  const navigate = useNavigate()
  const { apiClient, authSession } = useAppServices()
  const sessionState = useSyncExternalStore(authSession.subscribe, authSession.getState)
  const isAccessKey = sessionState.principalType === 'access_key'

  const rawSearch = useRouterState({
    select: (state) => state.location.search as SharedRouteQuery,
  })
  const monitorPath = pagePath('monitor')

  const appliedFilters = useMemo(() => {
    const filters = parseAppliedUsageFilters(rawSearch)
    return isAccessKey ? scopeAccessKeyUsageFilters(filters) : filters
  }, [rawSearch, isAccessKey])
  const breakdownSort = normalizeUsageBreakdownSort(appliedFilters.breakdown_sort)
  const breakdownSortDirection = normalizeUsageBreakdownSortDirection(
    appliedFilters.breakdown_sort_direction,
    breakdownSort,
  )
  // Resets the draft whenever the applied filter fields change — carried here
  // as a render adjustment on a signature key.
  const draftSignature = `${appliedFilters.range}|${appliedFilters.group_id ?? ''}|${
    appliedFilters.upstream_model ?? ''
  }`
  const [draftState, setDraftState] = useState<{
    signature: string
    draft: UsageFilterDraft
    errors: UsageFilterErrors
  }>(() => ({
    signature: draftSignature,
    draft: createUsageFilterDraft(appliedFilters),
    errors: {},
  }))
  if (draftState.signature !== draftSignature) {
    setDraftState({
      signature: draftSignature,
      draft: createUsageFilterDraft(appliedFilters),
      errors: {},
    })
  }
  const draft = draftState.draft
  const filterErrors = draftState.errors

  const groupsQuery = useQuery(groupOptionsQueryOptions(apiClient, !isAccessKey))
  const channelsQuery = useQuery({
    queryKey: controlQueryKeys.channels.list(''),
    queryFn: ({ signal }) => listChannels(apiClient, '', signal),
    enabled: !isAccessKey,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
  const usageQuery = useQuery(usageQueryOptions(apiClient, appliedFilters))
  const report = usageQuery.data

  const loading = useCollectionLoading(
    {
      pending: usageQuery.isPending,
      placeholder: usageQuery.isPlaceholderData,
      fetching: usageQuery.isFetching,
      hasData: report !== undefined,
      itemCount: report?.breakdown.rows.length ?? 0,
    },
    { fallbackRows: 5 },
  )
  const usageRefreshing =
    loading.refreshing ||
    (!isAccessKey && groupsQuery.data !== undefined && groupsQuery.isFetching) ||
    (!isAccessKey && channelsQuery.data !== undefined && channelsQuery.isFetching)

  const hasData = (report?.summary.request_count ?? 0) > 0

  function navigateUsage(filters: UsageFilters): Promise<void> {
    const scopedFilters = isAccessKey ? scopeAccessKeyUsageFilters(filters) : filters
    return navigate({
      to: monitorPath,
      search: usageMonitorQuery(scopedFilters),
      resetScroll: false,
    }) as Promise<void>
  }

  function updateDraftField(field: keyof UsageFilterDraft, value: string): void {
    setDraftState((prev) => ({
      ...prev,
      draft: { ...prev.draft, [field]: value },
    }))
  }

  async function applyFilters(): Promise<void> {
    const errors = validateUsageFilterDraft(draft)
    setDraftState((prev) => ({ ...prev, errors }))
    if (Object.keys(errors).length > 0) return
    await navigateUsage({
      ...applyUsageFilterDraft(draft),
      breakdown_sort: breakdownSort,
      breakdown_sort_direction: breakdownSortDirection,
    })
  }

  async function resetFilters(): Promise<void> {
    setDraftState((prev) => ({
      ...prev,
      draft: createUsageFilterDraft({ range: appliedFilters.range }),
      errors: {},
    }))
    await navigateUsage({
      range: appliedFilters.range,
      breakdown_sort: breakdownSort,
      breakdown_sort_direction: breakdownSortDirection,
    })
  }

  async function setBreakdownPage(page: number): Promise<void> {
    await navigateUsage({ ...appliedFilters, breakdown_page: page })
  }

  async function setBreakdownPageSize(pageSize: 20 | 50 | 100): Promise<void> {
    await navigateUsage({
      ...appliedFilters,
      breakdown_page: 1,
      breakdown_page_size: pageSize,
    })
  }

  async function setBreakdownSort(
    sort: UsageBreakdownSort,
    direction: UsageBreakdownSortDirection,
  ): Promise<void> {
    await navigateUsage({
      ...appliedFilters,
      breakdown_page: 1,
      breakdown_sort: sort,
      breakdown_sort_direction: direction,
    })
  }

  async function refresh(): Promise<void> {
    await Promise.all([
      usageQuery.refetch(),
      ...(!isAccessKey ? [groupsQuery.refetch(), channelsQuery.refetch()] : []),
    ])
  }

  useImperativeHandle(handleRef, () => ({ refresh }))

  return (
    <div {...stylex.props(styles.root)}>
      {usageRefreshing && (
        <span aria-live="polite" {...stylex.props(styles.refreshing)}>
          {t('monitor.usage.loading')}
        </span>
      )}

      <UsageFilterBar
        draft={draft}
        errors={filterErrors}
        groups={groupsQuery.data ?? []}
        groupsFailed={groupsQuery.isError}
        selfScoped={isAccessKey}
        onFieldChange={updateDraftField}
        onApply={() => void applyFilters()}
        onReset={() => void resetFilters()}
      />

      {usageQuery.isPending || loading.initial || loading.transition ? (
        <div {...stylex.props(styles.skeleton)} aria-label={t('monitor.usage.loading')}>
          <Skeleton height={38} radius={2} />
          <Skeleton height={320} radius={2} />
        </div>
      ) : usageQuery.isError && !report ? (
        <div role="alert">
          <EmptyState
            title={t('monitor.usage.loadFailed')}
            icon={<TriangleAlert size={20} />}
            actions={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void usageQuery.refetch()}
              />
            }
          />
        </div>
      ) : report ? (
        <>
          {usageQuery.isError && (
            <Banner
              status="warning"
              title={t('monitor.usage.stale')}
              endContent={
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('common.retry')}
                  onClick={() => void usageQuery.refetch()}
                />
              }
            />
          )}

          {!isAccessKey && (groupsQuery.isError || channelsQuery.isError) && (
            <Banner status="warning" title={t('monitor.usage.options.partialFailed')} />
          )}

          {!hasData ? (
            <EmptyState
              title={t('monitor.usage.empty.title')}
              description={t('monitor.usage.empty.description')}
              icon={<Database size={20} aria-hidden />}
            />
          ) : (
            <section {...stylex.props(styles.section)} aria-labelledby="usage-breakdown-title">
              <MonitorSectionHeading
                id="usage-breakdown-title"
                title={t('monitor.usage.breakdown.title')}
                description={t('monitor.usage.breakdown.description')}
                meta={t('monitor.usage.breakdown.rowCount', {
                  count: report.breakdown.pagination.total_items,
                })}
              />
              <UsageBreakdownTable
                breakdown={report.breakdown}
                groups={groupsQuery.data ?? []}
                sort={breakdownSort}
                sortDirection={breakdownSortDirection}
                onPage={(page) => void setBreakdownPage(page)}
                onPageSize={(size) => void setBreakdownPageSize(size)}
                onSort={(sort, direction) => void setBreakdownSort(sort, direction)}
              />
            </section>
          )}
        </>
      ) : null}
    </div>
  )
}
