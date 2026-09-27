import * as stylex from '@stylexjs/stylex'
import {
  Banner,
  Button,
  EmptyState,
  SegmentedControl,
  SegmentedControlItem,
  Selector,
  Skeleton,
} from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { Database, TriangleAlert } from 'lucide-react'
import {
  useImperativeHandle,
  useMemo,
  useState,
  useSyncExternalStore,
  type Ref,
} from 'react'
import { useIntl } from 'react-intl'

import { controlQueryKeys } from '@shared/control/query-keys'
import { listAccessKeyOptions } from '@shared/control/resources/access-keys'
import { listChannels } from '@shared/control/resources/channels'
import { groupOptionsQueryOptions } from '@shared/control/resources/groups'
import {
  normalizeUsageBreakdownSort,
  normalizeUsageBreakdownSortDirection,
  usageQueryOptions,
  type UsageAggregateDto,
  type UsageBreakdownSort,
  type UsageBreakdownSortDirection,
  type UsageDistributionDimension,
  type UsageDistributionMetric,
  type UsageFilters,
  type UsageRange,
} from '@shared/control/resources/usage'
import {
  applyUsageFilterDraft,
  createUsageFilterDraft,
  parseAppliedUsageFilters,
  validateUsageFilterDraft,
  type UsageFilterDraft,
  type UsageFilterErrors,
} from '@shared/domain/monitor/usage-filters'
import type { UsageBarDatum } from '@shared/domain/monitor/usage-bar-chart'
import {
  formatEstimatedCost,
  formatInteger,
  formatLocalInstant,
  formatTokens,
} from '@shared/lib/format'
import { pagePath } from '@shared/routing/page-routes'
import {
  parseUsageMonitorState,
  scopeAccessKeyUsageFilters,
  usageMonitorQuery,
  type UsageMonitorState,
  type UsageTrendMetric,
} from '@shared/routing/monitor-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import type { MessageId } from '@shared/i18n/message-ids'

import { useAppServices } from '../../app/services'
import { useCollectionLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { MonitorSectionHeading } from './MonitorSectionHeading'
import { UsageBarChart } from './UsageBarChart'
import { UsageBreakdownTable } from './UsageBreakdownTable'
import { UsageDistribution } from './UsageDistribution'
import { UsageFilterForm } from './UsageFilterForm'
import { UsageSummary } from './UsageSummary'

const COMPACT = '@media (max-width: 620px)'

const styles = stylex.create({
  root: {
    display: 'grid',
    minWidth: 0,
    gap: '22px',
  },
  refreshing: {
    minHeight: 'var(--space-4)',
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
  trendPanel: {
    display: 'grid',
    minWidth: 0,
    gap: '12px',
  },
  trendChart: {
    minWidth: 0,
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface)',
    paddingTop: { default: '18px', [COMPACT]: '14px' },
    paddingBottom: { default: '14px', [COMPACT]: '10px' },
    paddingInline: { default: '20px', [COMPACT]: '12px' },
  },
  qualityGrid: {
    display: 'grid',
    overflow: 'hidden',
    margin: 0,
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-border-subtle)',
    gap: '1px',
    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
  },
  qualityCell: {
    minWidth: 0,
    minHeight: '78px',
    backgroundColor: 'var(--color-surface)',
    paddingBlock: '13px',
    paddingInline: '15px',
  },
  qualityTerm: {
    display: 'flex',
    alignItems: 'center',
    gap: '7px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  qualityDot: {
    width: '6px',
    height: '6px',
    flex: '0 0 6px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-warning)',
  },
  qualityDotDanger: {
    backgroundColor: 'var(--color-danger)',
  },
  qualityValue: {
    marginTop: '7px',
    marginBottom: 0,
    marginInline: 0,
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: '1.1rem',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 560,
    letterSpacing: '-0.035em',
  },
  buckets: {
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface)',
  },
  bucketsSummary: {
    display: 'flex',
    minHeight: '48px',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingInline: '16px',
    color: 'var(--color-text-muted)',
    cursor: 'pointer',
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
  },
  bucketsSummaryOpen: {
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
  },
  bucketsBody: {
    paddingBlock: '14px',
    paddingInline: '16px',
  },
  bucketsTable: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: 'var(--text-sm)',
  },
  bucketsTh: {
    paddingBlock: '8px',
    paddingInline: '10px',
    textAlign: 'left',
    color: 'var(--color-text-faint)',
    fontWeight: 500,
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    whiteSpace: 'nowrap',
  },
  bucketsTd: {
    paddingBlock: '8px',
    paddingInline: '10px',
    color: 'var(--color-text-muted)',
    fontVariantNumeric: 'tabular-nums',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    verticalAlign: 'top',
  },
  metricSelect: {
    minWidth: '104px',
  },
})

export interface UsageTabHandle {
  openFilters(): void
  refresh(): Promise<void>
}

const costChartResolution = 1_000_000_000n

export function UsageTab({ handleRef }: { handleRef?: Ref<UsageTabHandle> }) {
  const t = useT()
  const intl = useIntl()
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
  const routeState = useMemo(() => parseUsageMonitorState(rawSearch), [rawSearch])
  const filterOpen = routeState.filtersOpen

  // Classic resets the draft whenever the applied filter fields change or the
  // panel toggles — carried here as a render adjustment on a signature key.
  const draftSignature = `${appliedFilters.range}|${appliedFilters.group_id ?? ''}|${
    appliedFilters.channel_id ?? ''
  }|${appliedFilters.credential_id ?? ''}|${appliedFilters.upstream_model ?? ''}|${filterOpen}`
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
  const accessKeysQuery = useQuery({
    queryKey: controlQueryKeys.accessKeys.options(),
    queryFn: ({ signal }) => listAccessKeyOptions(apiClient, signal),
    enabled: !isAccessKey,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
  const usageQuery = useQuery(usageQueryOptions(apiClient, appliedFilters))
  const report = usageQuery.data

  const [distributionDimension, setDistributionDimension] =
    useState<UsageDistributionDimension>('model')
  const [distributionMetric, setDistributionMetric] =
    useState<UsageDistributionMetric>('cost')

  const distribution = useMemo(() => {
    const distributions = report?.distributions
    if (distributions === undefined) return undefined
    if (isAccessKey || distributionDimension === 'model') {
      return distributions.model[distributionMetric]
    }
    if (distributionDimension === 'access_key') {
      return (
        distributions.access_key?.[distributionMetric] ??
        distributions.model[distributionMetric]
      )
    }
    return (
      distributions.group?.[distributionMetric] ??
      distributions.model[distributionMetric]
    )
  }, [report, isAccessKey, distributionDimension, distributionMetric])

  const loading = useCollectionLoading(
    {
      pending: usageQuery.isPending,
      placeholder: usageQuery.isPlaceholderData,
      fetching: usageQuery.isFetching,
      hasData: report !== undefined,
      itemCount:
        (distribution?.items.length ?? 0) + (distribution?.other == null ? 0 : 1),
    },
    { fallbackRows: 5 },
  )
  const usageRefreshing =
    loading.refreshing ||
    (!isAccessKey && groupsQuery.data !== undefined && groupsQuery.isFetching) ||
    (!isAccessKey && channelsQuery.data !== undefined && channelsQuery.isFetching) ||
    (!isAccessKey && accessKeysQuery.data !== undefined && accessKeysQuery.isFetching)

  const hasData = (report?.summary.request_count ?? 0) > 0

  const trendPresentation =
    routeState.metric === 'cost'
      ? {
          title: t('monitor.usage.trend.costTitle'),
          description: t('monitor.usage.trend.costDescription'),
          accessibleDescription: t('monitor.usage.trend.costAccessibleDescription'),
          valueLabel: t('monitor.usage.columns.estimatedCost'),
          secondaryLabel: undefined,
        }
      : {
          title: t('monitor.usage.trend.tokensTitle'),
          description: t('monitor.usage.trend.tokensDescription'),
          accessibleDescription: t('monitor.usage.trend.tokensAccessibleDescription'),
          valueLabel: t('monitor.usage.columns.totalTokens'),
          secondaryLabel: t('monitor.usage.tokens.cacheRead'),
        }

  const barTrendSeries = useMemo<UsageBarDatum[]>(() => {
    const buckets = report?.series ?? []
    if (routeState.metric === 'tokens') {
      return buckets.map((bucket) => ({
        bucket_start_ms: bucket.bucket_start_ms,
        bucket_end_ms: bucket.bucket_end_ms,
        primary_value: bucket.total_tokens,
        secondary_value: bucket.cache_read_tokens,
        primary_display: formatTokens(bucket.total_tokens, intl.locale),
        secondary_display: formatTokens(bucket.cache_read_tokens, intl.locale),
        details: [
          {
            label: t('monitor.usage.trend.inputTokens'),
            display: formatTokens(inputTokens(bucket), intl.locale),
          },
          {
            label: t('monitor.usage.trend.outputTokens'),
            display: formatTokens(bucket.output_tokens, intl.locale),
          },
        ],
      }))
    }
    const costs = buckets.map((bucket) => BigInt(bucket.estimated_cost_nano_usd))
    const maximum = costs.reduce(
      (current, value) => (value > current ? value : current),
      0n,
    )
    return buckets.map((bucket, index) => ({
      bucket_start_ms: bucket.bucket_start_ms,
      bucket_end_ms: bucket.bucket_end_ms,
      primary_value: normalizeTrendCost(costs[index]!, maximum),
      secondary_value: 0,
      primary_display: formatEstimatedCost(
        bucket.estimated_cost_nano_usd,
        intl.locale,
      ),
    }))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- t is stable per locale
  }, [report, routeState.metric, intl.locale])

  function navigateUsage(
    filters: UsageFilters,
    state: UsageMonitorState = {
      filtersOpen: false,
      seriesExpanded: false,
      metric: routeState.metric,
    },
  ): Promise<void> {
    const scopedFilters = isAccessKey ? scopeAccessKeyUsageFilters(filters) : filters
    return navigate({
      to: monitorPath,
      search: usageMonitorQuery(scopedFilters, state),
      resetScroll: false,
    }) as Promise<void>
  }

  function updateDraftField(field: keyof UsageFilterDraft, value: string): void {
    setDraftState((prev) => ({
      ...prev,
      draft: { ...prev.draft, [field]: value },
    }))
  }

  function setFilterOpen(open: boolean): void {
    if (!open) {
      setDraftState({
        signature: draftSignature,
        draft: createUsageFilterDraft(appliedFilters),
        errors: {},
      })
    }
    void navigateUsage(appliedFilters, { ...routeState, filtersOpen: open })
  }

  function openFilters(): void {
    setDraftState({
      signature: draftSignature,
      draft: createUsageFilterDraft(appliedFilters),
      errors: {},
    })
    void navigateUsage(appliedFilters, { ...routeState, filtersOpen: true })
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

  function updateDistributionDimension(value: string): void {
    if (value !== 'group' && value !== 'model' && value !== 'access_key') return
    if (
      isAccessKey ||
      (value === 'group' && report?.distributions.group === undefined) ||
      (value === 'access_key' && report?.distributions.access_key === undefined)
    ) {
      return
    }
    setDistributionDimension(value)
  }

  function updateDistributionMetric(value: string): void {
    if (value !== 'requests' && value !== 'tokens' && value !== 'cost') return
    setDistributionMetric(value)
  }

  async function updateTrendMetric(value: string): Promise<void> {
    if (value !== 'tokens' && value !== 'cost') return
    await navigateUsage(appliedFilters, {
      ...routeState,
      metric: value as UsageTrendMetric,
    })
  }

  function setSeriesExpanded(event: React.SyntheticEvent<HTMLDetailsElement>): void {
    const expanded = event.currentTarget.open
    if (expanded === routeState.seriesExpanded) return
    void navigateUsage(appliedFilters, { ...routeState, seriesExpanded: expanded })
  }

  async function refresh(): Promise<void> {
    await Promise.all([
      usageQuery.refetch(),
      ...(!isAccessKey
        ? [groupsQuery.refetch(), channelsQuery.refetch(), accessKeysQuery.refetch()]
        : []),
    ])
  }

  useImperativeHandle(handleRef, () => ({ openFilters, refresh }))

  function rangeLabel(range: UsageRange): string {
    return t(`monitor.usage.filters.ranges.${range}` as MessageId)
  }

  function granularityLabel(): string {
    const bucketWidthMS = report?.bucket_width_ms
    if (bucketWidthMS === 60 * 60 * 1000) return t('monitor.usage.trend.hourly')
    if (bucketWidthMS === 24 * 60 * 60 * 1000) return t('monitor.usage.trend.daily')
    return t('monitor.usage.trend.everyHours', {
      count: (bucketWidthMS ?? 0) / (60 * 60 * 1000),
    })
  }

  const qualityCells: {
    key: string
    label: string
    value: number
    danger?: boolean
  }[] = report
    ? [
        {
          key: 'missing',
          label: t('monitor.usage.quality.missing'),
          value: report.summary.usage_missing_count,
        },
        {
          key: 'partial',
          label: t('monitor.usage.quality.partial'),
          value: report.summary.partial_count,
        },
        {
          key: 'unpriced',
          label: t('monitor.usage.quality.unpriced'),
          value: report.summary.unpriced_request_count,
          danger: true,
        },
        {
          key: 'pricingPartial',
          label: t('monitor.usage.quality.pricingPartial'),
          value: report.summary.pricing_partial_count,
        },
      ]
    : []

  return (
    <div {...stylex.props(styles.root)}>
      <span aria-live="polite" {...stylex.props(styles.refreshing)}>
        {usageRefreshing ? t('monitor.usage.loading') : ''}
      </span>

      {usageQuery.isPending || loading.initial || loading.transition ? (
        <div {...stylex.props(styles.skeleton)} aria-label={t('monitor.usage.loading')}>
          <Skeleton height={96} radius={2} />
          <Skeleton height={180} radius={2} />
          <Skeleton height={140} radius={2} />
          <Skeleton height={220} radius={2} />
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

          {!isAccessKey &&
            (groupsQuery.isError ||
              channelsQuery.isError ||
              accessKeysQuery.isError) && (
              <Banner status="warning" title={t('monitor.usage.options.partialFailed')} />
            )}

          <UsageSummary summary={report.summary} />

          {!isAccessKey &&
            (report.collection_health.dropped_total > 0 ||
              report.collection_health.write_failure_total > 0) && (
              <Banner
                status="error"
                title={t('monitor.usage.process.warning', {
                  dropped: formatInteger(
                    report.collection_health.dropped_total,
                    intl.locale,
                  ),
                  failures: formatInteger(
                    report.collection_health.write_failure_total,
                    intl.locale,
                  ),
                })}
              />
            )}

          {!hasData ? (
            <EmptyState
              title={t('monitor.usage.empty.title')}
              description={t('monitor.usage.empty.description')}
              icon={<Database size={20} aria-hidden />}
            />
          ) : (
            <>
              <section
                {...stylex.props(styles.trendPanel)}
                aria-labelledby="usage-trend-title"
              >
                <MonitorSectionHeading
                  id="usage-trend-title"
                  title={trendPresentation.title}
                  description={trendPresentation.description}
                  meta={`${rangeLabel(report.range)} · ${granularityLabel()}`}
                  actions={
                    <SegmentedControl
                      value={routeState.metric}
                      label={t('monitor.usage.trend.metrics.label')}
                      size="sm"
                      onChange={(value) => void updateTrendMetric(value)}
                    >
                      <SegmentedControlItem
                        value="tokens"
                        label={t('monitor.usage.trend.metrics.tokens')}
                      />
                      <SegmentedControlItem
                        value="cost"
                        label={t('monitor.usage.trend.metrics.cost')}
                      />
                    </SegmentedControl>
                  }
                />
                <div {...stylex.props(styles.trendChart)}>
                  <UsageBarChart
                    series={barTrendSeries}
                    title={trendPresentation.title}
                    description={trendPresentation.accessibleDescription}
                    emptyLabel={t('monitor.usage.trend.empty')}
                    primaryLabel={trendPresentation.valueLabel}
                    secondaryLabel={trendPresentation.secondaryLabel}
                    primaryZeroDisplay={
                      routeState.metric === 'cost'
                        ? formatEstimatedCost('0', intl.locale)
                        : formatTokens(0, intl.locale)
                    }
                    secondaryZeroDisplay={formatTokens(0, intl.locale)}
                    detailZeroDisplay={formatTokens(0, intl.locale)}
                    rangeStart={report.from_ms}
                    rangeEnd={report.to_ms}
                    locale={intl.locale}
                    grouped={routeState.metric === 'tokens'}
                  />
                </div>
              </section>

              <section {...stylex.props(styles.section)}>
                <MonitorSectionHeading
                  title={t('monitor.usage.quality.title')}
                  description={t('monitor.usage.quality.description')}
                />
                <dl {...stylex.props(styles.qualityGrid)}>
                  {qualityCells.map((cell) => (
                    <div key={cell.key} {...stylex.props(styles.qualityCell)}>
                      <dt {...stylex.props(styles.qualityTerm)}>
                        <span
                          {...stylex.props(
                            styles.qualityDot,
                            cell.danger && styles.qualityDotDanger,
                          )}
                          aria-hidden="true"
                        />
                        {cell.label}
                      </dt>
                      <dd {...stylex.props(styles.qualityValue)}>
                        {formatInteger(cell.value, intl.locale)}
                      </dd>
                    </div>
                  ))}
                </dl>
              </section>

              <section
                {...stylex.props(styles.section)}
                aria-labelledby="usage-distribution-title"
              >
                <MonitorSectionHeading
                  id="usage-distribution-title"
                  title={t('monitor.usage.distribution.title')}
                  description={t('monitor.usage.distribution.description')}
                  meta={t('monitor.usage.distribution.limit')}
                  actions={
                    <>
                      {!isAccessKey && (
                        <SegmentedControl
                          value={distributionDimension}
                          label={t('monitor.usage.distribution.dimensionLabel')}
                          size="sm"
                          onChange={updateDistributionDimension}
                        >
                          <SegmentedControlItem
                            value="model"
                            label={t('monitor.usage.distribution.dimensions.model')}
                          />
                          <SegmentedControlItem
                            value="group"
                            label={t('monitor.usage.distribution.dimensions.group')}
                          />
                          <SegmentedControlItem
                            value="access_key"
                            label={t('monitor.usage.distribution.dimensions.accessKey')}
                          />
                        </SegmentedControl>
                      )}
                      <Selector
                        label={t('monitor.usage.distribution.metricLabel')}
                        isLabelHidden
                        variant="ghost"
                        size="sm"
                        value={distributionMetric}
                        options={[
                          {
                            value: 'requests',
                            label: t('monitor.usage.distribution.metrics.requests'),
                          },
                          {
                            value: 'tokens',
                            label: t('monitor.usage.distribution.metrics.tokens'),
                          },
                          {
                            value: 'cost',
                            label: t('monitor.usage.distribution.metrics.cost'),
                          },
                        ]}
                        onChange={updateDistributionMetric}
                      />
                    </>
                  }
                />
                {distribution && (
                  <UsageDistribution
                    distribution={distribution}
                    summary={report.summary}
                    groups={groupsQuery.data ?? []}
                    channels={channelsQuery.data?.items ?? []}
                    accessKeys={accessKeysQuery.data ?? []}
                  />
                )}
              </section>

              <section
                {...stylex.props(styles.section)}
                aria-labelledby="usage-breakdown-title"
              >
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
                  channels={channelsQuery.data?.items ?? []}
                  sort={breakdownSort}
                  sortDirection={breakdownSortDirection}
                  onPage={(page) => void setBreakdownPage(page)}
                  onPageSize={(size) => void setBreakdownPageSize(size)}
                  onSort={(sort, direction) => void setBreakdownSort(sort, direction)}
                />
              </section>

              <details
                {...stylex.props(styles.buckets)}
                open={routeState.seriesExpanded}
                onToggle={setSeriesExpanded}
              >
                <summary
                  {...stylex.props(
                    styles.bucketsSummary,
                    routeState.seriesExpanded && styles.bucketsSummaryOpen,
                  )}
                >
                  {t('monitor.usage.series.disclosure')}
                </summary>
                <div {...stylex.props(styles.bucketsBody)}>
                  <table
                    {...stylex.props(styles.bucketsTable)}
                    aria-label={t('monitor.usage.series.caption')}
                  >
                    <thead>
                      <tr>
                        <th scope="col" {...stylex.props(styles.bucketsTh)}>
                          {t('monitor.usage.columns.window')}
                        </th>
                        <th scope="col" {...stylex.props(styles.bucketsTh)}>
                          {t('monitor.usage.columns.requests')}
                        </th>
                        <th scope="col" {...stylex.props(styles.bucketsTh)}>
                          {t('monitor.usage.columns.success')}
                        </th>
                        <th scope="col" {...stylex.props(styles.bucketsTh)}>
                          {t('monitor.usage.columns.failure')}
                        </th>
                        <th scope="col" {...stylex.props(styles.bucketsTh)}>
                          {t('monitor.usage.columns.totalTokens')}
                        </th>
                        <th scope="col" {...stylex.props(styles.bucketsTh)}>
                          {t('monitor.usage.columns.estimatedCost')}
                        </th>
                        <th scope="col" {...stylex.props(styles.bucketsTh)}>
                          {t('monitor.usage.columns.quality')}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.series.map((bucket) => (
                        <tr key={bucket.bucket_start_ms}>
                          <td {...stylex.props(styles.bucketsTd)}>
                            <time dateTime={new Date(bucket.bucket_start_ms).toISOString()}>
                              {formatLocalInstant(bucket.bucket_start_ms, intl.locale)}
                            </time>
                          </td>
                          <td {...stylex.props(styles.bucketsTd)}>
                            {formatInteger(bucket.request_count, intl.locale)}
                          </td>
                          <td {...stylex.props(styles.bucketsTd)}>
                            {formatInteger(bucket.success_count, intl.locale)}
                          </td>
                          <td {...stylex.props(styles.bucketsTd)}>
                            {formatInteger(bucket.failure_count, intl.locale)}
                          </td>
                          <td {...stylex.props(styles.bucketsTd)}>
                            {formatTokens(bucket.total_tokens, intl.locale)}
                          </td>
                          <td {...stylex.props(styles.bucketsTd)}>
                            {formatEstimatedCost(
                              bucket.estimated_cost_nano_usd,
                              intl.locale,
                            )}
                          </td>
                          <td {...stylex.props(styles.bucketsTd)}>
                            {t('monitor.usage.columns.qualityCompact', {
                              missing: formatInteger(
                                bucket.usage_missing_count,
                                intl.locale,
                              ),
                              partial: formatInteger(bucket.partial_count, intl.locale),
                              unpriced: formatInteger(
                                bucket.unpriced_request_count,
                                intl.locale,
                              ),
                              pricingPartial: formatInteger(
                                bucket.pricing_partial_count,
                                intl.locale,
                              ),
                            })}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </>
          )}
        </>
      ) : null}

      <UsageFilterForm
        open={filterOpen}
        draft={draft}
        errors={filterErrors}
        groups={groupsQuery.data ?? []}
        channels={channelsQuery.data?.items ?? []}
        groupsFailed={groupsQuery.isError}
        channelsFailed={channelsQuery.isError}
        selfScoped={isAccessKey}
        onOpenChange={setFilterOpen}
        onFieldChange={updateDraftField}
        onApply={() => void applyFilters()}
        onReset={() => void resetFilters()}
      />
    </div>
  )
}

function inputTokens(aggregate: UsageAggregateDto): number {
  return (
    aggregate.uncached_input_tokens +
    aggregate.cache_read_tokens +
    aggregate.cache_write_5m_tokens +
    aggregate.cache_write_1h_tokens +
    aggregate.cache_write_unknown_tokens
  )
}

function normalizeTrendCost(value: bigint, maximum: bigint): number {
  if (maximum === 0n) return 0
  return Number((value * costChartResolution + maximum / 2n) / maximum)
}
