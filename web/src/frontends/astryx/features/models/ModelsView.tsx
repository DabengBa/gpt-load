import * as stylex from '@stylexjs/stylex'
import {
  Badge,
  Button,
  EmptyState,
  Pagination,
  Selector,
  Skeleton,
  TextInput,
} from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { RefreshCw, Search, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { useIntl } from 'react-intl'

import {
  modelCollectionQueryOptions,
  type ModelCollectionGroupStatus,
  type ModelCollectionPricingStatus,
  type ModelUpstreamDto,
} from '@shared/control/resources/models'
import { formatLocalInstant } from '@shared/lib/format'
import { constrainCollectionSearch, type SharedRouteQuery } from '@shared/routing/route-query'
import { pagePath } from '@shared/routing/page-routes'
import {
  isCanonicalModelsRouteQuery,
  parseModelsRouteQuery,
  serializeModelsRouteQuery,
  type ModelsRouteState,
} from '@shared/routing/models-route'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { useCollectionLoading } from '../../app/collection-loading'
import { useDebouncedAction } from '../../app/use-debounced-action'
import { useModelPriceSync } from '../../app/use-model-price-sync'
import { useVisibleRefetch } from '../../app/use-visible-refetch'
import { ModelTree } from './ModelTree'
import {
  ModelUpstreamDrawer,
  type ModelUpstreamDrawerHandle,
} from './ModelUpstreamDrawer'

const FILTER_MID = '@media (max-width: 980px)'
const FILTER_NARROW = '@media (max-width: 680px)'
const RESULT_NARROW = '@media (max-width: 560px)'

const styles = stylex.create({
  page: {
    width: '100%',
    paddingTop: 'var(--stage-padding-top)',
    paddingBottom: 'var(--stage-padding-bottom)',
    paddingInline: 'var(--stage-padding-inline)',
  },
  pageInner: {
    width: 'min(100%, 1240px)',
    marginInline: 'auto',
  },
  sheet: {
    position: 'relative',
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-4-5)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-sheet)',
    backgroundColor: 'var(--color-surface)',
    boxShadow: 'var(--shadow-sheet)',
    paddingTop: 'var(--sheet-padding-top)',
    paddingBottom: 'var(--sheet-padding-bottom)',
    paddingInline: 'var(--sheet-padding-inline)',
  },
  headerRow: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    flexWrap: 'wrap',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-heading-2-size, 20px)',
    fontWeight: 650,
  },
  refreshing: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  status: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-1-75)',
    margin: 0,
    marginTop: 'calc(var(--space-4-5) * -0.35)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  statusSeparator: {
    color: 'var(--color-text-faint)',
  },
  statusTime: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  filterBar: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(240px, 1fr) repeat(2, minmax(142px, 0.42fr))',
      [FILTER_MID]: 'repeat(3, minmax(0, 1fr))',
      [FILTER_NARROW]: 'minmax(0, 1fr)',
    },
    alignItems: 'end',
    gap: '10px',
    paddingTop: 'var(--space-1)',
    paddingBottom: '13px',
  },
  filterBarScoped: {
    gridTemplateColumns: {
      default: 'minmax(240px, 1fr) minmax(142px, 0.42fr)',
      [FILTER_MID]: 'repeat(3, minmax(0, 1fr))',
      [FILTER_NARROW]: 'minmax(0, 1fr)',
    },
  },
  filterField: {
    display: 'grid',
    minWidth: 0,
    gap: '5px',
  },
  filterFieldSearch: {
    gridColumn: { default: 'auto', [FILTER_MID]: '1 / -1', [FILTER_NARROW]: 'auto' },
  },
  filterLabel: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  filterResult: {
    gridColumn: '1 / -1',
    display: 'flex',
    minHeight: '32px',
    alignItems: { default: 'center', [RESULT_NARROW]: 'flex-start' },
    flexDirection: { default: 'row', [RESULT_NARROW]: 'column' },
    justifyContent: 'space-between',
    gap: { default: '14px', [RESULT_NARROW]: 'var(--space-1)' },
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    color: 'var(--color-text-faint)',
    paddingBottom: '9px',
    fontSize: 'var(--text-sm)',
  },
  banner: {
    borderRadius: 'var(--radius-control, 6px)',
    borderWidth: 1,
    borderStyle: 'solid',
    padding: '9px 12px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: 'var(--text-meta)',
  },
  bannerSuccess: {
    borderColor: 'var(--color-success)',
    backgroundColor:
      'var(--color-success-bg, color-mix(in srgb, var(--color-success) 12%, transparent))',
  },
  bannerDanger: {
    borderColor: 'var(--color-danger)',
    backgroundColor:
      'var(--color-danger-bg, color-mix(in srgb, var(--color-danger) 10%, transparent))',
  },
  bannerWarning: {
    borderColor: 'var(--color-warning)',
    backgroundColor:
      'var(--color-warning-bg, color-mix(in srgb, var(--color-warning) 12%, transparent))',
  },
  skeletonGrid: {
    display: 'grid',
    gap: '10px',
  },
  pagination: {
    display: 'flex',
    justifyContent: 'flex-end',
  },
})

const groupStatusIds: Record<ModelCollectionGroupStatus, Parameters<ReturnType<typeof useT>>[0]> = {
  enabled: 'models.filters.groupStatus.enabled',
  all: 'models.filters.groupStatus.all',
}
const pricingStatusIds: Record<
  ModelCollectionPricingStatus,
  Parameters<ReturnType<typeof useT>>[0]
> = {
  all: 'models.filters.pricingStatus.all',
  pending: 'models.filters.pricingStatus.pending',
  configured: 'models.filters.pricingStatus.configured',
}

export function ModelsView() {
  const t = useT()
  const intl = useIntl()
  const navigate = useNavigate()
  const { apiClient, authSession } = useAppServices()
  const sessionState = useSyncExternalStore(authSession.subscribe, authSession.getState)
  const isAccessKey = sessionState.principalType === 'access_key'

  const { rawSearch, searchStr } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
    }),
  })
  const routeState = useMemo<ModelsRouteState>(() => {
    const state = parseModelsRouteQuery(rawSearch)
    if (!isAccessKey) return state
    return {
      filters: { ...state.filters, group_status: 'enabled' },
      selectedPriceID: undefined,
    }
  }, [rawSearch, isAccessKey])
  const filters = routeState.filters
  const modelsPath = pagePath('models')

  const [searchDraft, setSearchDraft] = useState(filters.q ?? '')
  const searchDraftRef = useRef(searchDraft)
  useEffect(() => {
    searchDraftRef.current = searchDraft
  }, [searchDraft])
  const debounce = useDebouncedAction(250)
  const sync = useModelPriceSync()
  const syncSnapshot = sync.snapshot

  const modelsQuery = useQuery(modelCollectionQueryOptions(apiClient, filters, isAccessKey))
  const data = modelsQuery.data
  const collectionBusy = data !== undefined && modelsQuery.isFetching
  const {
    initial: initialLoading,
    transition: collectionTransition,
    refreshing: collectionRefreshing,
    rows: skeletonRows,
  } = useCollectionLoading(
    {
      pending: modelsQuery.isPending,
      placeholder: modelsQuery.isPlaceholderData,
      fetching: modelsQuery.isFetching,
      hasData: data !== undefined,
      itemCount:
        data?.items.reduce((count, item) => count + item.upstream_models.length + 1, 0) ?? 0,
    },
    { fallbackRows: 10 },
  )

  const drawerRef = useRef<ModelUpstreamDrawerHandle>(null)
  const drawerOpen = routeState.selectedPriceID !== undefined
  const activePriceID = routeState.selectedPriceID ?? null

  // Route → draft sync via the render-adjustment pattern.
  const [lastSearchStr, setLastSearchStr] = useState(searchStr)
  if (lastSearchStr !== searchStr) {
    setLastSearchStr(searchStr)
    debounce.cancel()
    setSearchDraft(filters.q ?? '')
  }

  // Canonicalize non-canonical query params — drops junk/duplicated keys.
  useEffect(() => {
    if (!isCanonicalModelsRouteQuery(rawSearch, routeState)) {
      void navigate({
        to: modelsPath,
        search: serializeModelsRouteQuery(routeState),
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the URL string
  }, [searchStr])

  // Page correction: when the server clamps a deep page, mirror it into the URL.
  const totalPages = data?.pagination.total_pages
  const requestedPage = filters.page
  const isPlaceholder = modelsQuery.isPlaceholderData
  useEffect(() => {
    if (isPlaceholder || totalPages === undefined) return
    const lastPage = Math.max(1, totalPages)
    if (requestedPage > lastPage) {
      void navigate({
        to: modelsPath,
        search: serializeModelsRouteQuery({ filters: { ...filters, page: lastPage } }),
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on server pagination
  }, [totalPages, requestedPage, isPlaceholder])

  useVisibleRefetch([modelsQuery.refetch])

  function navigateState(patch: Partial<ModelsRouteState>, replace = false): void {
    const next = { ...routeState, ...patch }
    void navigate({
      to: modelsPath,
      search: serializeModelsRouteQuery(next),
      replace,
      resetScroll: false,
    })
  }

  /** 列表条件变化会让抽屉里的草稿失去上下文,先确认再放弃并关闭。 */
  async function confirmDiscard(): Promise<boolean> {
    if (!drawerOpen || !drawerRef.current) return true
    if (!(await drawerRef.current.confirmDiscardSwitch())) return false
    drawerRef.current.discardChanges()
    return true
  }

  async function applySearch(): Promise<void> {
    const normalized = constrainCollectionSearch(searchDraftRef.current)
    if (normalized === filters.q) return
    if (!(await confirmDiscard())) {
      setSearchDraft(filters.q ?? '')
      return
    }
    navigateState({
      filters: { ...filters, q: normalized, page: 1 },
      selectedPriceID: undefined,
    })
  }

  function scheduleSearch(): void {
    debounce.schedule(() => {
      void applySearch()
    })
  }

  function onSearchChange(value: string): void {
    setSearchDraft(value)
    if (value === '') {
      debounce.cancel()
      void (async () => {
        if (filters.q === undefined) return
        if (!(await confirmDiscard())) {
          setSearchDraft(filters.q ?? '')
          return
        }
        navigateState({
          filters: { ...filters, q: undefined, page: 1 },
          selectedPriceID: undefined,
        })
      })()
      return
    }
    scheduleSearch()
  }

  async function setGroupStatus(value: string | null): Promise<void> {
    if (value === null || value === '' || value === filters.group_status) return
    if (value !== 'enabled' && value !== 'all') return
    if (!(await confirmDiscard())) return
    navigateState({
      filters: { ...filters, group_status: value, page: 1 },
      selectedPriceID: undefined,
    })
  }

  async function setPricingStatus(value: string | null): Promise<void> {
    if (value === null || value === '' || value === filters.pricing_status) return
    if (value !== 'all' && value !== 'pending' && value !== 'configured') return
    if (!(await confirmDiscard())) return
    navigateState({
      filters: { ...filters, pricing_status: value, page: 1 },
      selectedPriceID: undefined,
    })
  }

  const hasConditions =
    filters.q !== undefined ||
    filters.group_status !== 'enabled' ||
    filters.pricing_status !== 'all'

  async function resetConditions(): Promise<void> {
    if (!hasConditions || !(await confirmDiscard())) return
    debounce.cancel()
    setSearchDraft('')
    navigateState({
      filters: { group_status: 'enabled', pricing_status: 'all', page: 1, page_size: 10 },
      selectedPriceID: undefined,
    })
  }

  async function changePage(nextPage: number): Promise<void> {
    if (nextPage === filters.page || !(await confirmDiscard())) return
    navigateState({ filters: { ...filters, page: nextPage }, selectedPriceID: undefined })
  }

  async function openUpstream(upstream: ModelUpstreamDto): Promise<void> {
    if (isAccessKey) return
    if (upstream.price.id === activePriceID && drawerOpen) return
    if (drawerOpen && drawerRef.current && !(await drawerRef.current.confirmDiscardSwitch()))
      return
    drawerRef.current?.discardChanges()
    navigateState({ selectedPriceID: upstream.price.id })
  }

  function closeDrawer(): void {
    navigateState({ selectedPriceID: undefined })
  }

  const groupStatusOptions = (['enabled', 'all'] as const).map((value) => ({
    value,
    label: t(groupStatusIds[value]),
  }))
  const pricingStatusOptions = (['all', 'pending', 'configured'] as const).map((value) => ({
    value,
    label: t(pricingStatusIds[value]),
  }))

  const catalogTone = (() => {
    if (!data || !data.catalog.available) return 'neutral' as const
    return data.catalog.error_code ? ('warning' as const) : ('success' as const)
  })()
  const catalogLabel = (() => {
    if (!data) return ''
    if (!data.catalog.available) return t('models.catalog.unavailable')
    return data.catalog.error_code ? t('models.catalog.stale') : t('models.catalog.available')
  })()

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="models-title">
      <div {...stylex.props(styles.pageInner)}>
        <div {...stylex.props(styles.sheet)} aria-busy={collectionBusy || undefined}>
          <div {...stylex.props(styles.headerRow)}>
            <h1 id="models-title" {...stylex.props(styles.title)}>
              {t('models.title')}
            </h1>
            {!isAccessKey && (
              <Button
                size="sm"
                isLoading={syncSnapshot.pending}
                icon={<RefreshCw size={15} aria-hidden />}
                label={t('models.actions.sync')}
                onClick={() => void sync.controller.run()}
              />
            )}
          </div>
          <span aria-live="polite" {...stylex.props(styles.refreshing)}>
            {collectionRefreshing ? t('models.loading') : ''}
          </span>

          {!isAccessKey && syncSnapshot.succeeded && (
            <div {...stylex.props(styles.banner, styles.bannerSuccess)} role="status">
              {t('models.sync.succeeded')}
            </div>
          )}
          {!isAccessKey && syncSnapshot.failed && (
            <div {...stylex.props(styles.banner, styles.bannerDanger)} role="alert">
              <span>{t('models.sync.failed')}</span>
              <Button
                variant="ghost"
                size="sm"
                label={t('common.retry')}
                onClick={() => void sync.controller.run()}
              />
            </div>
          )}

          {modelsQuery.isPending || initialLoading ? (
            <div
              {...stylex.props(styles.skeletonGrid)}
              role="status"
              aria-label={t('models.loading')}
            >
              {Array.from({ length: filters.page_size }, (_, index) => (
                <Skeleton key={index} height={72} radius={2} />
              ))}
            </div>
          ) : modelsQuery.isError && !data ? (
            <div role="alert">
              <EmptyState
                title={t('models.loadFailed')}
                icon={<TriangleAlert size={20} />}
                actions={
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={() => void modelsQuery.refetch()}
                  />
                }
              />
            </div>
          ) : data ? (
            <>
              {modelsQuery.isError && (
                <div {...stylex.props(styles.banner, styles.bannerWarning)} role="status">
                  <TriangleAlert size={13} aria-hidden />
                  <span>{t('models.stale')}</span>
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={() => void modelsQuery.refetch()}
                  />
                </div>
              )}

              <p {...stylex.props(styles.status)} aria-live="polite">
                <span>
                  {t('models.status.models', {
                    count: intl.formatNumber(data.summary.client_model_count),
                  })}
                </span>
                <span {...stylex.props(styles.statusSeparator)} aria-hidden>
                  ·
                </span>
                <span>
                  {t('models.status.upstreams', {
                    count: intl.formatNumber(data.summary.upstream_model_count),
                  })}
                </span>
                <span {...stylex.props(styles.statusSeparator)} aria-hidden>
                  ·
                </span>
                {data.summary.pending_price_count > 0 ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    label={t('models.status.pending', {
                      count: intl.formatNumber(data.summary.pending_price_count),
                    })}
                    onClick={() => void setPricingStatus('pending')}
                  />
                ) : (
                  <span>
                    {t('models.status.pending', {
                      count: intl.formatNumber(data.summary.pending_price_count),
                    })}
                  </span>
                )}
                <span {...stylex.props(styles.statusSeparator)} aria-hidden>
                  ·
                </span>
                <span title={t('models.context')}>{t('models.status.unit')}</span>
                {!isAccessKey && (
                  <>
                    <span {...stylex.props(styles.statusSeparator)} aria-hidden>
                      ·
                    </span>
                    <Badge variant={catalogTone} label={catalogLabel} />
                    {data.catalog.successful_fetch_at_ms > 0 && (
                      <span {...stylex.props(styles.statusTime)}>
                        {formatLocalInstant(data.catalog.successful_fetch_at_ms, intl.locale)}
                      </span>
                    )}
                  </>
                )}
              </p>

              <div
                {...stylex.props(styles.filterBar, isAccessKey && styles.filterBarScoped)}
                role="group"
                aria-label={t('models.filters.region')}
              >
                <label {...stylex.props(styles.filterField, styles.filterFieldSearch)}>
                  <span {...stylex.props(styles.filterLabel)}>
                    {t('models.filters.searchLabel')}
                  </span>
                  <TextInput
                    size="sm"
                    label={t('models.filters.searchLabel')}
                    placeholder={t('models.filters.searchPlaceholder')}
                    startIcon={<Search size={14} />}
                    value={searchDraft}
                    hasClear
                    onChange={onSearchChange}
                  />
                </label>
                {!isAccessKey && (
                  <label {...stylex.props(styles.filterField)}>
                    <span {...stylex.props(styles.filterLabel)}>
                      {t('models.filters.groupStatusLabel')}
                    </span>
                    <Selector
                      size="sm"
                      variant="input"
                      label={t('models.filters.groupStatusLabel')}
                      options={groupStatusOptions}
                      value={filters.group_status}
                      onChange={(value) => void setGroupStatus(value)}
                    />
                  </label>
                )}
                <label {...stylex.props(styles.filterField)}>
                  <span {...stylex.props(styles.filterLabel)}>
                    {t('models.filters.pricingStatusLabel')}
                  </span>
                  <Selector
                    size="sm"
                    variant="input"
                    label={t('models.filters.pricingStatusLabel')}
                    options={pricingStatusOptions}
                    value={filters.pricing_status}
                    onChange={(value) => void setPricingStatus(value)}
                  />
                </label>
                {hasConditions && (
                  <span {...stylex.props(styles.filterResult)}>
                    <span aria-live="polite">
                      {t('models.result', {
                        shown: intl.formatNumber(data.items.length),
                        total: intl.formatNumber(data.pagination.total_items),
                      })}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      label={t('models.filters.reset')}
                      onClick={() => void resetConditions()}
                    />
                  </span>
                )}
              </div>

              {collectionTransition ? (
                <div
                  {...stylex.props(styles.skeletonGrid)}
                  role="status"
                  aria-label={t('models.loading')}
                >
                  {Array.from({ length: Math.min(skeletonRows, 12) }, (_, index) => (
                    <Skeleton key={index} height={44} radius={2} />
                  ))}
                </div>
              ) : data.pagination.total_items === 0 ? (
                <EmptyState
                  title={
                    hasConditions ? t('models.empty.noResultsTitle') : t('models.empty.title')
                  }
                  description={
                    hasConditions
                      ? t('models.empty.noResultsDescription')
                      : t('models.empty.description')
                  }
                  icon={hasConditions ? <Search size={20} /> : <RefreshCw size={20} />}
                  actions={
                    hasConditions ? (
                      <Button
                        variant="secondary"
                        size="sm"
                        label={t('models.filters.reset')}
                        onClick={() => void resetConditions()}
                      />
                    ) : !isAccessKey ? (
                      <Button
                        size="sm"
                        label={t('models.actions.configureGroups')}
                        href={pagePath('groups')}
                      />
                    ) : undefined
                  }
                />
              ) : (
                <>
                  <ModelTree items={data.items} readOnly={isAccessKey} onOpen={openUpstream} />
                  <div {...stylex.props(styles.pagination)}>
                    <Pagination
                      page={data.pagination.page}
                      totalItems={data.pagination.total_items}
                      totalPages={data.pagination.total_pages}
                      pageSize={data.pagination.page_size}
                      onChange={(page) => void changePage(page)}
                      isDisabled={collectionBusy}
                      size="sm"
                    />
                  </div>
                </>
              )}
            </>
          ) : null}
        </div>
      </div>

      {!isAccessKey && (
        <ModelUpstreamDrawer
          ref={drawerRef}
          isOpen={drawerOpen}
          priceId={activePriceID}
          onClose={closeDrawer}
        />
      )}
    </section>
  )
}
