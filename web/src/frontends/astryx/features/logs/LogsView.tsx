import * as stylex from '@stylexjs/stylex'
import { Banner, Button, EmptyState, IconButton, Selector, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useRouter, useRouterState } from '@tanstack/react-router'
import { ChevronLeft, ChevronRight, Search, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'

import { controlQueryKeys } from '@shared/control/query-keys'
import { accessKeyOptionsQueryOptions } from '@shared/control/resources/access-keys'
import { listChannels, type ChannelDto } from '@shared/control/resources/channels'
import { groupOptionsQueryOptions } from '@shared/control/resources/groups'
import {
  requestLogQueryOptions,
  type RequestLogFilters,
  type RequestLogPageSize,
} from '@shared/control/resources/request-logs'
import {
  applyLogFilterDraft,
  createLogFilterDraft,
  defaultRequestLogFilters,
  parseAppliedLogFilterState,
  serializeAppliedLogFilters,
  validateLogFilterDraft,
  type LogFilterDraft,
  type LogFilterErrors,
} from '@shared/domain/monitor/log-filters'
import { isValidRequestLogAffinityKey } from '@shared/domain/monitor/request-log-affinity'
import {
  logsMonitorQuery,
  parseLogsMonitorState,
  scopeAccessKeyLogFilters,
  type LogsMonitorState,
} from '@shared/routing/logs-route'
import { pagePath } from '@shared/routing/page-routes'
import type { SharedRouteQueryRaw } from '@shared/routing/route-query'
import type { MessageId } from '@shared/i18n/message-ids'

import { useCollectionLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { stringifySharedRouteSearch } from '../../app/search-codec'
import { useAppServices } from '../../app/services'
import { LogDetailDrawer } from '../monitor/LogDetailDrawer'
import { LogsFilterForm, type AppliedChip } from '../monitor/LogsFilterForm'
import { LogsTable } from './LogsTable'

const TABLET = '@media (max-width: 860px)'

const styles = stylex.create({
  page: {
    display: 'grid',
    minWidth: 0,
  },
  sheet: {
    display: 'grid',
    minWidth: 0,
    alignContent: 'start',
    gap: 'var(--space-4)',
    minHeight: {
      default: '760px',
      '@media (max-width: 800px)': '0',
    },
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-title)',
    fontWeight: 650,
  },
  summary: {
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  skeleton: {
    display: 'grid',
    gap: 8,
  },
  state: {
    paddingBlock: 24,
  },
  // Classic AsyncRefreshIndicator — assertive only while active.
  refreshing: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clipPath: 'inset(50%)',
  },
  pagination: {
    display: 'flex',
    minHeight: 32,
    alignItems: 'center',
    gap: 10,
    paddingBlock: 4,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    justifyContent: { default: 'flex-end', [TABLET]: 'center' },
  },
  pageNumber: {
    minWidth: 24,
    textAlign: 'center',
    fontVariantNumeric: 'tabular-nums',
  },
  pageSize: {
    minWidth: 108,
  },
})

const logPageSizes: readonly RequestLogPageSize[] = [20, 50, 100]

const allAdvancedFilterKeys: readonly (keyof RequestLogFilters)[] = [
  'channel_id',
  'credential_id',
  'upstream_model',
  'model_consistency',
  'access_key_id',
  'request_id',
  'protocol',
  'operation',
  'stream',
  'final_status_code',
  'usage_state',
  'cost_state',
  'pricing_completeness',
  'cache_present',
  'attempt_status_code',
  'failure_category',
  'error_code',
  'retry_state',
  'retry_count_min',
  'retry_count_max',
  'first_response_min_ms',
  'first_response_max_ms',
  'duration_min_ms',
  'duration_max_ms',
  'input_tokens_min',
  'input_tokens_max',
  'output_tokens_min',
  'output_tokens_max',
  'cost_min_nano_usd',
  'cost_max_nano_usd',
  'affinity_key',
]

const accessKeyForbiddenFilterKeys: ReadonlySet<keyof RequestLogFilters> = new Set([
  'group_id',
  'channel_id',
  'credential_id',
  'upstream_model',
  'model_consistency',
  'access_key_id',
  'attempt_status_code',
  'failure_category',
  'error_code',
  'retry_state',
  'retry_count_min',
  'retry_count_max',
  'affinity_key',
])

function LogsPagination({
  page,
  pageSize,
  hasPrevious,
  hasNext,
  busy,
  onPrevious,
  onNext,
  onPageSize,
}: {
  page: number
  pageSize: number
  hasPrevious: boolean
  hasNext: boolean
  busy: boolean
  onPrevious: () => void
  onNext: () => void
  onPageSize: (size: RequestLogPageSize) => void
}) {
  const t = useT()
  const pageSizeOptions = useMemo(
    () =>
      logPageSizes.map((size) => ({
        value: String(size),
        label: t('common.pagination.pageSize', { size }),
      })),
    [t],
  )
  return (
    <nav
      {...stylex.props(styles.pagination)}
      aria-label={t('common.pagination.label')}
      aria-busy={busy || undefined}
    >
      <span {...stylex.props(styles.pageSize)}>
        <Selector
          label={t('common.pagination.pageSizeLabel')}
          isLabelHidden
          variant="ghost"
          size="sm"
          value={String(pageSize)}
          options={pageSizeOptions}
          isDisabled={busy}
          onChange={(value) => {
            const size = Number(value)
            if (size === 20 || size === 50 || size === 100) {
              onPageSize(size)
            }
          }}
        />
      </span>
      <IconButton
        variant="secondary"
        size="sm"
        label={t('common.pagination.previous')}
        icon={<ChevronLeft size={16} aria-hidden />}
        isDisabled={busy || !hasPrevious}
        onClick={onPrevious}
      />
      <span {...stylex.props(styles.pageNumber)}>{page}</span>
      <IconButton
        variant="secondary"
        size="sm"
        label={t('common.pagination.next')}
        icon={<ChevronRight size={16} aria-hidden />}
        isDisabled={busy || !hasNext}
        onClick={onNext}
      />
    </nav>
  )
}

export function LogsView() {
  const t = useT()
  const { apiClient, authSession } = useAppServices()
  const sessionState = useSyncExternalStore(authSession.subscribe, authSession.getState)
  const isAccessKey = sessionState.principalType === 'access_key'
  const navigate = useNavigate()
  const router = useRouter()
  const rawSearch = useRouterState({
    select: (state) => state.location.search as Record<string, unknown>,
  })
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  const logsPath = pagePath('logs')

  const appliedFilterState = useMemo(() => parseAppliedLogFilterState(rawSearch), [rawSearch])
  const invalidAffinityKey = isAccessKey
    ? undefined
    : appliedFilterState.invalidAffinityKey
  const appliedFilters = useMemo(
    () =>
      isAccessKey
        ? scopeAccessKeyLogFilters(appliedFilterState.filters)
        : appliedFilterState.filters,
    [appliedFilterState, isAccessKey],
  )
  const routeState = useMemo(() => parseLogsMonitorState(rawSearch), [rawSearch])
  const selectedRequestID = routeState.selectedRequestID
  const advancedOpen = routeState.filtersOpen
  const currentCursor = routeState.cursorHistory.at(-1)
  const currentPage = routeState.cursorHistory.length + 1

  const [detailClosing, setDetailClosing] = useState(false)
  const [paginationPending, setPaginationPending] = useState(false)
  const [pageTransitionOrigin, setPageTransitionOrigin] =
    useState<LogsMonitorState | null>(null)
  const [draft, setDraft] = useState<LogFilterDraft>(() => createLogFilterDraft(appliedFilters))
  const [filterErrors, setFilterErrors] = useState<LogFilterErrors>(() =>
    invalidAffinityKey !== undefined
      ? { affinity_key: 'monitor.logs.errors.affinityKey' }
      : {},
  )
  const detailFocusTimerRef = useRef<number | undefined>(undefined)
  const pendingDetailNavRef = useRef<Promise<unknown> | undefined>(undefined)

  // Classic watch 1 (filterSignature, immediate): the applied filters own the
  // draft. Carried as a render adjustment keyed on the serialized signature.
  const filterSignature = JSON.stringify([
    serializeAppliedLogFilters(appliedFilters),
    invalidAffinityKey,
  ])
  const [appliedSync, setAppliedSync] = useState(filterSignature)
  if (appliedSync !== filterSignature) {
    setAppliedSync(filterSignature)
    setDraft(createLogFilterDraft(appliedFilters))
    setFilterErrors(
      invalidAffinityKey !== undefined
        ? { affinity_key: 'monitor.logs.errors.affinityKey' }
        : {},
    )
    setPaginationPending(false)
    setPageTransitionOrigin(null)
  }

  // Classic watch 2 (selectedRequestID): a new selection clears the closing
  // flag so the drawer can reopen for another row.
  const [detailSync, setDetailSync] = useState(selectedRequestID)
  if (detailSync !== selectedRequestID) {
    setDetailSync(selectedRequestID)
    setDetailClosing(false)
  }

  const groupsQuery = useQuery(groupOptionsQueryOptions(apiClient, !isAccessKey))
  const channelsQuery = useQuery({
    queryKey: controlQueryKeys.channels.list(''),
    queryFn: ({ signal }) => listChannels(apiClient, '', signal),
    enabled: !isAccessKey,
    staleTime: 5 * 60 * 1_000,
  })
  const accessKeyOptionsQuery = useQuery(
    accessKeyOptionsQueryOptions(apiClient, !isAccessKey),
  )
  const logsQuery = useQuery({
    ...requestLogQueryOptions(apiClient, appliedFilters, currentCursor),
    enabled: invalidAffinityKey === undefined,
  })
  const logs = logsQuery.data?.items ?? []

  // Classic watch 3 (dataUpdatedAt): fresh data ends the pagination pending
  // state and the transition origin used for error rollback.
  const [dataSync, setDataSync] = useState(0)
  if (logsQuery.dataUpdatedAt > 0 && logsQuery.dataUpdatedAt !== dataSync) {
    setDataSync(logsQuery.dataUpdatedAt)
    setPaginationPending(false)
    setPageTransitionOrigin(null)
  }

  // Classic watch 4 (isError): a failed page transition restores the origin
  // URL so back/forward state stays consistent with what the user sees. The
  // pending flags clear as a render adjustment; the replace navigation runs in
  // the effect below, and clearing the captured origin happens in the
  // navigation callback (setState inside an effect body is lint-forbidden).
  const [pendingRestore, setPendingRestore] = useState<LogsMonitorState | null>(null)
  if (logsQuery.isError && pageTransitionOrigin !== null && pendingRestore === null) {
    setPendingRestore(pageTransitionOrigin)
    setPaginationPending(false)
    setPageTransitionOrigin(null)
  }
  useEffect(() => {
    // Pending transition: the outgoing route still renders while location has
    // moved — a late restore must not resurrect this page.
    if (pathname !== logsPath) return
    if (pendingRestore === null) return
    const origin = pendingRestore
    void navigateLogs(logsMonitorQuery(appliedFilters, origin), true).finally(() =>
      setPendingRestore((current) => (current === origin ? null : current)),
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mirrors the classic isError watch edge
  }, [pendingRestore])

  const loading = useCollectionLoading(
    {
      pending: logsQuery.isPending,
      placeholder: logsQuery.isPlaceholderData,
      fetching: logsQuery.isFetching,
      hasData: logsQuery.data !== undefined,
      itemCount: logs.length,
    },
    { fallbackRows: 20 },
  )
  const logsRefreshing =
    loading.refreshing ||
    (!isAccessKey && groupsQuery.data !== undefined && groupsQuery.isFetching) ||
    (!isAccessKey && channelsQuery.data !== undefined && channelsQuery.isFetching) ||
    (!isAccessKey &&
      accessKeyOptionsQuery.data !== undefined &&
      accessKeyOptionsQuery.isFetching)
  const paginationBusy = paginationPending || logsQuery.isFetching

  const groupNames = useMemo<Record<number, string>>(
    () => Object.fromEntries((groupsQuery.data ?? []).map((group) => [group.id, group.name])),
    [groupsQuery.data],
  )
  const groupProviderUrls = useMemo<Record<number, string>>(
    () =>
      Object.fromEntries(
        (groupsQuery.data ?? [])
          .filter((group) => group.provider_url !== null)
          .map((group) => [group.id, group.provider_url as string]),
      ),
    [groupsQuery.data],
  )
  const channelsByID = useMemo<Record<string, ChannelDto>>(
    () =>
      Object.fromEntries(
        (channelsQuery.data?.items ?? []).map((channel) => [channel.channel_id, channel]),
      ),
    [channelsQuery.data],
  )

  const advancedFilterKeys = isAccessKey
    ? allAdvancedFilterKeys.filter((key) => !accessKeyForbiddenFilterKeys.has(key))
    : allAdvancedFilterKeys
  const advancedCount = advancedFilterKeys.filter(
    (key) => appliedFilters[key] !== undefined,
  ).length
  const hasNonTimeFilters = Object.keys(appliedFilters).some(
    (key) => key !== 'from_ms' && key !== 'to_ms' && key !== 'limit',
  )
  // Chips only cover what the quick form cannot show (advanced drawer
  // fields); time/group/model/status already render in visible controls.
  const appliedChips = useMemo<AppliedChip[]>(() => {
    const values: AppliedChip[] = []
    for (const key of advancedFilterKeys) {
      const value = appliedFilters[key]
      if (value === undefined) continue
      values.push({ key, label: chipLabel(key, value) })
    }
    return values
    // eslint-disable-next-line react-hooks/exhaustive-deps -- chipLabel reads t + option data
  }, [appliedFilters, isAccessKey, groupsQuery.data, channelsQuery.data, accessKeyOptionsQuery.data, t])

  function chipLabel(key: keyof RequestLogFilters, value: unknown): string {
    if (key === 'access_key_id') {
      const accessKey = accessKeyOptionsQuery.data?.find(({ id }) => id === value)
      return t('monitor.logs.filters.appliedAccessKey', {
        value: accessKey?.name ?? `#${value}`,
      })
    }
    if (key === 'channel_id') {
      const channel = channelsQuery.data?.items.find(
        ({ channel_id }) => channel_id === value,
      )
      return t('monitor.logs.filters.appliedChannel', {
        value: channel?.name ?? String(value),
      })
    }
    if (key === 'credential_id') {
      return t('monitor.logs.filters.appliedCredential', { value: String(value) })
    }
    if (key === 'upstream_model') {
      return t('monitor.logs.filters.appliedUpstreamModel', { value: String(value) })
    }
    if (key === 'model_consistency') {
      return t('monitor.logs.filters.appliedModelConsistency', {
        value: t(`monitor.logs.filters.modelConsistency.${String(value)}` as MessageId),
      })
    }
    if (key === 'request_id') {
      return t('monitor.logs.filters.appliedRequestId', { value: String(value) })
    }
    if (key === 'affinity_key') {
      return t('monitor.logs.filters.appliedAffinityKey', { value: String(value) })
    }
    if (key === 'protocol') return String(value)
    if (key === 'operation') {
      return t('monitor.logs.filters.appliedOperation', {
        value: t(`monitor.logs.operation.${String(value)}` as MessageId),
      })
    }
    if (key === 'failure_category') {
      return t(`monitor.logs.failureCategory.${String(value)}` as MessageId)
    }
    if (key === 'retry_state') {
      return t(`monitor.logs.filters.retryState.${String(value)}` as MessageId)
    }
    if (key === 'usage_state') {
      return t('monitor.logs.filters.appliedUsageState', {
        value: t(`monitor.logs.filters.usageState.${String(value)}` as MessageId),
      })
    }
    if (key === 'cost_state') {
      return t('monitor.logs.filters.appliedCostState', {
        value: t(`monitor.logs.filters.costState.${String(value)}` as MessageId),
      })
    }
    if (key === 'pricing_completeness') {
      return t('monitor.logs.filters.appliedCompleteness', {
        value: t(`monitor.logs.filters.completeness.${String(value)}` as MessageId),
      })
    }
    const labelKeys: Partial<Record<keyof RequestLogFilters, string>> = {
      stream: 'stream',
      final_status_code: 'finalStatusCode',
      usage_state: 'usageStateLabel',
      cost_state: 'costStateLabel',
      pricing_completeness: 'completenessLabel',
      cache_present: 'cachePresent',
      channel_id: 'channel',
      credential_id: 'credential',
      attempt_status_code: 'attemptStatusCode',
      error_code: 'errorCode',
    }
    const rangeKey = key.replace(/_nano_usd$/u, '_usd')
    const label = labelKeys[key]
      ? t(`monitor.logs.filters.${labelKeys[key]}` as MessageId)
      : t(`monitor.logs.filters.rangeFields.${rangeKey}` as MessageId)
    const display =
      typeof value === 'boolean' ? t(value ? 'monitor.logs.yes' : 'monitor.logs.no') : value
    return `${label} ${String(display)}`
  }

  function navigateLogs(query: SharedRouteQueryRaw, replace = false): Promise<unknown> {
    return navigate({
      href: `${logsPath}${stringifySharedRouteSearch(query)}`,
      resetScroll: false,
      replace,
    })
  }

  function updateDraftField(field: keyof LogFilterDraft, value: string): void {
    // Functional update — preset clicks emit two back-to-back field writes and
    // a stale-closure spread would drop the first one.
    setDraft((prev) => ({ ...prev, [field]: value }))
  }

  async function commitFilters(filters: RequestLogFilters): Promise<void> {
    // The awaited detail navigation can have already committed a new URL —
    // classic reads the live route state after the await; re-parse the live
    // location here for the same semantics.
    if (pendingDetailNavRef.current) await pendingDetailNavRef.current
    const liveSearch = router.state.location.search as Record<string, unknown>
    const liveState = parseLogsMonitorState(liveSearch)
    const liveParsed = parseAppliedLogFilterState(liveSearch)
    const liveInvalid = isAccessKey ? undefined : liveParsed.invalidAffinityKey
    const liveFilters = isAccessKey
      ? scopeAccessKeyLogFilters(liveParsed.filters)
      : liveParsed.filters
    const liveSignature = JSON.stringify([
      serializeAppliedLogFilters(liveFilters),
      liveInvalid,
    ])

    const scoped = isAccessKey ? scopeAccessKeyLogFilters(filters) : filters
    const serialized = serializeAppliedLogFilters(scoped)
    const nextSignature = JSON.stringify([serialized, undefined])
    setDraft(createLogFilterDraft(scoped))
    setFilterErrors({})

    if (
      nextSignature === liveSignature &&
      liveState.cursorHistory.length === 0 &&
      liveState.selectedRequestID === undefined &&
      !liveState.filtersOpen
    ) {
      await logsQuery.refetch()
      return
    }

    await navigateLogs(logsMonitorQuery(scoped))
  }

  // In-place narrowing, not a jump: triage needs other requests on the same
  // dimension, and the target entity may already be deleted.
  function filterByGroup(groupID: number): void {
    void commitFilters({ ...appliedFilters, group_id: groupID })
  }
  function filterByCredential(credentialID: number): void {
    void commitFilters({ ...appliedFilters, credential_id: credentialID })
  }
  function filterByClientModel(clientModel: string): void {
    void commitFilters({ ...appliedFilters, client_model: clientModel })
  }
  function filterByAffinityKey(affinityKey: string | null): void {
    if (isAccessKey || affinityKey === null || !isValidRequestLogAffinityKey(affinityKey)) {
      return
    }
    void commitFilters({ ...appliedFilters, affinity_key: affinityKey })
  }

  async function applyFilters(): Promise<void> {
    const errors = validateLogFilterDraft(draft)
    if (invalidAffinityKey !== undefined) {
      errors.affinity_key = 'monitor.logs.errors.affinityKey'
    }
    setFilterErrors(errors)
    if (Object.keys(errors).length > 0) return
    await commitFilters({
      ...applyLogFilterDraft(draft),
      limit: appliedFilters.limit ?? 20,
    })
  }

  async function resetFilters(): Promise<void> {
    await commitFilters({
      ...defaultRequestLogFilters(),
      limit: appliedFilters.limit ?? 20,
    })
  }

  function setPageSize(pageSize: RequestLogPageSize): void {
    if (paginationBusy) return
    void commitFilters({ ...appliedFilters, limit: pageSize })
  }

  function removeFilter(key: string): void {
    const filters = { ...appliedFilters }
    delete filters[key as keyof RequestLogFilters]
    void commitFilters(filters)
  }

  function nextPage(): void {
    if (paginationBusy) return
    const cursor = logsQuery.data?.next_cursor
    if (!cursor || cursor === currentCursor) return
    if (routeState.cursorHistory.includes(cursor)) return
    setPageTransitionOrigin({ ...routeState, cursorHistory: [...routeState.cursorHistory] })
    setPaginationPending(true)
    void navigateLogs(
      logsMonitorQuery(appliedFilters, {
        filtersOpen: false,
        cursorHistory: [...routeState.cursorHistory, cursor],
      }),
    )
  }

  function previousPage(): void {
    if (paginationBusy || routeState.cursorHistory.length === 0) return
    setPageTransitionOrigin({ ...routeState, cursorHistory: [...routeState.cursorHistory] })
    setPaginationPending(true)
    void navigateLogs(
      logsMonitorQuery(appliedFilters, {
        filtersOpen: false,
        cursorHistory: routeState.cursorHistory.slice(0, -1),
      }),
    )
  }

  function setAdvancedOpen(open: boolean): void {
    void navigateLogs(
      logsMonitorQuery(appliedFilters, {
        ...routeState,
        filtersOpen: open,
        selectedRequestID: undefined,
      }),
    )
  }

  async function setDetailOpen(requestID: string | undefined, open: boolean): Promise<void> {
    const closingID = selectedRequestID
    setDetailClosing(!open)
    const navigation = navigateLogs(
      logsMonitorQuery(appliedFilters, {
        ...routeState,
        filtersOpen: false,
        selectedRequestID: open ? requestID : undefined,
      }),
    )
    pendingDetailNavRef.current = navigation
    try {
      await navigation
    } finally {
      if (pendingDetailNavRef.current === navigation) pendingDetailNavRef.current = undefined
    }
    if (open || !closingID) return
    window.clearTimeout(detailFocusTimerRef.current)
    detailFocusTimerRef.current = window.setTimeout(() => {
      if (document.activeElement && document.activeElement !== document.body) return
      document.getElementById(`log-details-${closingID}`)?.focus()
    }, 30)
  }

  const optionsPartiallyFailed =
    !isAccessKey &&
    (groupsQuery.isError || channelsQuery.isError || accessKeyOptionsQuery.isError)
  const listBlocked = invalidAffinityKey !== undefined
  const skeletonRows = loading.rows

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="logs-title">
      <div {...stylex.props(styles.sheet)} data-testid="logs-tab">
        <h1 id="logs-title" {...stylex.props(styles.title)}>
          {t('shell.logs')}
        </h1>

        <LogsFilterForm
        draft={draft}
        errors={filterErrors}
        groups={groupsQuery.data ?? []}
        channels={channelsQuery.data?.items ?? []}
        accessKeys={accessKeyOptionsQuery.data ?? []}
        groupsFailed={groupsQuery.isError}
        channelsFailed={channelsQuery.isError}
        accessKeysFailed={accessKeyOptionsQuery.isError}
        appliedChips={appliedChips}
        advancedCount={advancedCount}
        advancedOpen={advancedOpen}
        selfScoped={isAccessKey}
        onAdvancedOpenChange={setAdvancedOpen}
        onUpdateField={updateDraftField}
        onRemoveFilter={removeFilter}
        onApply={() => void applyFilters()}
        onReset={() => void resetFilters()}
      />

      {optionsPartiallyFailed && (
        <Banner status="warning" title={t('monitor.logs.options.partialFailed')} />
      )}

      <span aria-live="polite" {...stylex.props(styles.refreshing)}>
        {logsRefreshing ? t('monitor.logs.loading') : ''}
      </span>

      {listBlocked || logsQuery.isPending || loading.initial ? (
        !listBlocked && (
          <div
            role="status"
            aria-label={t('monitor.logs.loading')}
            {...stylex.props(styles.skeleton)}
          >
            {Array.from({ length: appliedFilters.limit ?? 20 }, (_, index) => (
              <Skeleton key={index} height={52} radius={2} />
            ))}
          </div>
        )
      ) : logsQuery.isError && !logsQuery.data ? (
        <div role="alert" {...stylex.props(styles.state)}>
          <EmptyState
            title={t('monitor.logs.loadFailed')}
            icon={<TriangleAlert size={20} aria-hidden />}
            actions={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void logsQuery.refetch()}
              />
            }
          />
        </div>
      ) : logsQuery.data ? (
        <>
          {logsQuery.isError && (
            <Banner
              status="warning"
              title={t('monitor.logs.stale')}
              endContent={
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('common.retry')}
                  onClick={() => void logsQuery.refetch()}
                />
              }
            />
          )}
          {loading.transition ? (
            <div
              role="status"
              aria-label={t('monitor.logs.loading')}
              {...stylex.props(styles.skeleton)}
            >
              {Array.from({ length: skeletonRows }, (_, index) => (
                <Skeleton key={index} height={52} radius={2} />
              ))}
            </div>
          ) : (
            <>
              {logs.length > 0 && (
                <p {...stylex.props(styles.summary)} data-testid="logs-result-summary">
                  {t('monitor.logs.resultSummary', { count: logs.length })}
                </p>
              )}
              {logs.length > 0 ? (
                <LogsTable
                  logs={logs}
                  isAccessKey={isAccessKey}
                  groupNames={groupNames}
                  groupProviderUrls={groupProviderUrls}
                  groupsLoaded={groupsQuery.isSuccess}
                  channelsByID={channelsByID}
                  onFilterGroup={filterByGroup}
                  onFilterCredential={filterByCredential}
                  onFilterClientModel={filterByClientModel}
                  onFilterAffinityKey={filterByAffinityKey}
                  onOpenDetail={(id) => void setDetailOpen(id, true)}
                />
              ) : (
                <EmptyState
                  title={t(
                    hasNonTimeFilters
                      ? 'monitor.logs.empty.filteredTitle'
                      : 'monitor.logs.empty.title',
                  )}
                  description={t(
                    hasNonTimeFilters
                      ? 'monitor.logs.empty.filteredDescription'
                      : 'monitor.logs.empty.description',
                  )}
                  icon={<Search size={20} aria-hidden />}
                />
              )}
              <LogsPagination
                page={currentPage}
                pageSize={appliedFilters.limit ?? 20}
                hasPrevious={routeState.cursorHistory.length > 0}
                hasNext={Boolean(logsQuery.data.next_cursor)}
                busy={paginationBusy}
                onPrevious={previousPage}
                onNext={nextPage}
                onPageSize={setPageSize}
              />
            </>
          )}
        </>
      ) : null}

        <LogDetailDrawer
          open={
            Boolean(selectedRequestID) && !detailClosing && invalidAffinityKey === undefined
          }
          requestId={invalidAffinityKey === undefined ? selectedRequestID : undefined}
          selfScoped={isAccessKey}
          groupNames={groupNames}
          groupsLoaded={groupsQuery.isSuccess}
          providerUrls={groupProviderUrls}
          channels={channelsByID}
          onOpenChange={(open: boolean) => {
            if (!open) void setDetailOpen(undefined, false)
          }}
        />
      </div>
    </section>
  )
}
