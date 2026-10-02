import * as stylex from '@stylexjs/stylex'
import {
  Banner,
  Button,
  EmptyState,
  Pagination,
  Selector,
  Skeleton,
  TextInput,
} from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { KeyRound, Plus, Search, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useIntl } from 'react-intl'

import {
  accessKeyCollectionQueryOptions,
  updateAccessKey,
} from '@shared/control/resources/access-keys'
import { channelsQueryOptions } from '@shared/control/resources/channels'
import { groupOptionsQueryOptions } from '@shared/control/resources/groups'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { accessKeyResources } from '@shared/control/resources/access-keys'
import type {
  AccessKeyCollectionFilters,
  AccessKeyCollectionStatus,
  AccessKeyDto,
} from '@shared/control/types'
import type { PendingAccessKeyCreateOperation } from '@shared/domain/access-keys/access-key-create-operation'
import type { PendingAccessKeyEditOperation } from '@shared/domain/access-keys/access-key-edit-operation'
import type { PendingAccessKeyRotateOperation } from '@shared/domain/access-keys/access-key-rotate-operation'
import { RequestCancelledError } from '@shared/http/errors'
import type { MessageId } from '@shared/i18n/message-ids'
import { pagePath } from '@shared/routing/page-routes'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import {
  constrainAccessKeyCollectionSearchQuery,
  parseAccessKeyCollectionRouteQuery,
  parseAccessKeyDrawerRoute,
  type AccessKeyDrawerRoute,
} from '@shared/routing/access-key-collection-route'
import {
  parseSettingsRouteSection,
  serializeSettingsRouteQuery,
} from '@shared/routing/settings-route'

import { useT } from '../../app/i18n'
import { useCollectionLoading } from '../../app/collection-loading'
import { useAppServices } from '../../app/services'
import { useDebouncedAction } from '../../app/use-debounced-action'
import {
  AccessKeyCollection,
  type AccessKeyCollectionHandle,
} from '../access-keys/AccessKeyCollection'
import { AccessKeyDrawer } from '../access-keys/AccessKeyDrawer'

const styles = stylex.create({
  panel: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-3)',
  },
  headerRow: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  heading: {
    margin: 0,
    fontSize: 'var(--text-meta)',
    fontWeight: 650,
  },
  headingDescription: {
    margin: 0,
    marginTop: 'var(--space-1)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  refreshing: {
    minHeight: 'var(--space-4)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  loginNotice: {
    marginTop: 'var(--space-1)',
  },
  summaryStrip: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    fontSize: 'var(--text-sm)',
  },
  summaryLabel: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  filterBar: {
    display: 'grid',
    gridTemplateColumns: 'minmax(220px, 320px) auto',
    alignItems: 'end',
    gap: 'var(--space-3)',
  },
  filterField: {
    display: 'grid',
    gap: 'var(--space-1)',
    minWidth: 0,
  },
  filterLabel: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  filterResult: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 'var(--space-2)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    whiteSpace: 'nowrap',
  },
  skeletonGrid: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  srOnly: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
})

const statusFilterIds: Record<AccessKeyCollectionStatus, MessageId> = {
  active: 'accessKeys.collection.status.active',
  disabled: 'accessKeys.collection.status.disabled',
}

// The access-key collection + drawer moved here from the standalone
// /access-keys page. Route state now rides on the settings query string inside
// ?section=credentials (the shared settings codec owns (de)serialization).
export function AccessKeysPanel() {
  const t = useT()
  const intl = useIntl()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { apiClient, toast, unsavedChanges } = useAppServices()

  const { rawSearch, searchStr, pathname } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
      pathname: state.location.pathname,
    }),
  })
  const settingsPath = pagePath('settings')
  const credentialsActive = parseSettingsRouteSection(rawSearch) === 'credentials'
  const filters = useMemo(() => parseAccessKeyCollectionRouteQuery(rawSearch), [rawSearch])
  const drawerRoute = useMemo(() => parseAccessKeyDrawerRoute(rawSearch), [rawSearch])
  const drawerOpen = credentialsActive && drawerRoute !== undefined

  const [searchDraft, setSearchDraft] = useState(filters.q ?? '')
  const searchDraftRef = useRef(searchDraft)
  useEffect(() => {
    searchDraftRef.current = searchDraft
  }, [searchDraft])
  const debounce = useDebouncedAction(300)

  const [createOperation, setCreateOperation] = useState<PendingAccessKeyCreateOperation | null>(
    null,
  )
  const [editOperation, setEditOperation] = useState<PendingAccessKeyEditOperation | null>(null)
  const [rotateOperation, setRotateOperation] = useState<PendingAccessKeyRotateOperation | null>(
    null,
  )
  const [pendingStatusIDs, setPendingStatusIDs] = useState<ReadonlySet<number>>(new Set())
  const [deletionAnnouncement, setDeletionAnnouncement] = useState('')
  const statusControllersRef = useRef(new Map<number, AbortController>())
  const restoreFocusRef = useRef<HTMLElement | null>(null)
  const viewRootRef = useRef<HTMLDivElement | null>(null)
  const collectionRef = useRef<AccessKeyCollectionHandle | null>(null)
  const mountedRef = useRef(true)

  const accessKeysQuery = useQuery(accessKeyCollectionQueryOptions(apiClient, filters))
  const groupsQuery = useQuery(groupOptionsQueryOptions(apiClient))
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const data = accessKeysQuery.data
  const collectionBusy = data !== undefined && accessKeysQuery.isFetching
  const {
    initial: initialLoading,
    transition: collectionTransition,
    refreshing: collectionRefreshing,
    rows: skeletonRows,
  } = useCollectionLoading(
    {
      pending: accessKeysQuery.isPending,
      placeholder: accessKeysQuery.isPlaceholderData,
      fetching: accessKeysQuery.isFetching,
      hasData: data !== undefined,
      itemCount: data?.items.length ?? 0,
    },
    { fallbackRows: 20 },
  )
  const pageRefreshing =
    collectionRefreshing ||
    (groupsQuery.data !== undefined && groupsQuery.isFetching) ||
    (channelsQuery.data !== undefined && channelsQuery.isFetching)
  const hasFilterCriteria = filters.q !== undefined || filters.status !== undefined

  const groupCatalogState =
    groupsQuery.isError || channelsQuery.isError
      ? groupsQuery.data && channelsQuery.data
        ? 'stale'
        : 'error'
      : groupsQuery.isPending || channelsQuery.isPending
        ? 'loading'
        : 'ready'

  const lockedAccessKeyIDs = useMemo<ReadonlySet<number>>(
    () => (rotateOperation ? new Set([rotateOperation.base.id]) : new Set()),
    [rotateOperation],
  )

  // Route → draft sync + canonicalization. Render-phase adjustment mirrors the
  // classic deep watch on route.query (immediate: true).
  const [lastSearchStr, setLastSearchStr] = useState(searchStr)
  if (lastSearchStr !== searchStr) {
    setLastSearchStr(searchStr)
    debounce.cancel()
    setSearchDraft(filters.q ?? '')
  }

  // Selected-item resolution is fully derived from (route, data, pending ops) —
  // the classic `selected` ref only ever mirrors those inputs. The auto-close
  // below is navigation-only, so it stays a pure side effect.
  const selected = useMemo<AccessKeyDto | null>(() => {
    if (drawerRoute === undefined || drawerRoute.mode === 'create') return null
    const id = drawerRoute.accessKeyID
    return (
      data?.items.find((accessKey) => accessKey.id === id) ??
      (editOperation?.base.id === id ? editOperation.base : null) ??
      (rotateOperation?.base.id === id ? rotateOperation.base : null) ??
      null
    )
  }, [drawerRoute, data, editOperation, rotateOperation])

  const isPlaceholder = accessKeysQuery.isPlaceholderData
  useEffect(() => {
    if (pathname !== settingsPath || !credentialsActive) return
    if (drawerRoute === undefined || drawerRoute.mode !== 'edit') return
    if (selected !== null || !data || isPlaceholder) return
    void navigate({
      to: settingsPath,
      search: serializeSettingsRouteQuery('credentials', { collection: filters }),
      replace: true,
      resetScroll: false,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on resolution inputs
  }, [drawerRoute, selected, data, isPlaceholder, credentialsActive])

  // Page correction: clamp an out-of-range page to the final valid page.
  const totalPages = data?.pagination.total_pages
  const requestedPage = filters.page
  useEffect(() => {
    if (pathname !== settingsPath || !credentialsActive) return
    if (isPlaceholder || totalPages === undefined) return
    const lastPage = Math.max(1, totalPages)
    if (requestedPage > lastPage) {
      void navigate({
        to: settingsPath,
        search: serializeSettingsRouteQuery('credentials', {
          collection: { ...filters, page: lastPage },
          drawer: drawerRoute,
        }),
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on server pagination
  }, [totalPages, requestedPage, isPlaceholder, credentialsActive])

  useEffect(() => {
    const statusControllers = statusControllersRef.current
    return () => {
      mountedRef.current = false
      for (const controller of statusControllers.values()) controller.abort()
      statusControllers.clear()
      queryClient.removeQueries({ queryKey: accessKeyResources.collection.queryKey })
    }
  }, [queryClient])

  // Classic `watch(filters, () => collection.conceal())` — any change to the
  // route-derived filters conceals keys and aborts in-flight reveals, even
  // when the change is external (browser navigation) rather than via
  // navigateCollection below.
  useEffect(() => {
    collectionRef.current?.conceal()
  }, [filters])

  function navigateCollection(
    next: AccessKeyCollectionFilters,
    // `undefined` keeps the current drawer (default parameter), so `null` is
    // the explicit "clear the drawer" sentinel — passing `undefined` would
    // silently serialize the drawer params back into the same URL.
    drawer: AccessKeyDrawerRoute | null | undefined = drawerRoute,
    replace = false,
  ): Promise<unknown> {
    // Classic routeWithFilters conceals before navigating.
    collectionRef.current?.conceal()
    return navigate({
      to: settingsPath,
      search: serializeSettingsRouteQuery('credentials', {
        collection: next,
        drawer: drawer ?? undefined,
      }),
      replace,
      resetScroll: false,
    })
  }

  function setDrawerRoute(drawer: AccessKeyDrawerRoute | undefined, replace = false): void {
    void navigateCollection(filters, drawer ?? null, replace)
  }

  function updateConditions(
    patch: Partial<Pick<AccessKeyCollectionFilters, 'q' | 'status'>>,
  ): void {
    void navigateCollection({
      ...filters,
      q: constrainAccessKeyCollectionSearchQuery(searchDraftRef.current),
      ...patch,
      page: 1,
    })
  }

  function onSearchChange(value: string): void {
    setSearchDraft(value)
    if (value === '') {
      // TextInput's hasClear only fires onChange('') — treat it as the classic
      // @clear path: cancel the pending debounce and navigate immediately.
      debounce.cancel()
      if (filters.q !== undefined) updateConditions({ q: undefined })
      return
    }
    debounce.schedule(() => {
      updateConditions({ q: constrainAccessKeyCollectionSearchQuery(searchDraftRef.current) })
    })
  }

  function setStatusFilter(status: AccessKeyCollectionStatus | undefined): void {
    updateConditions({ status })
  }

  function resetConditions(): void {
    debounce.cancel()
    setSearchDraft('')
    void navigateCollection({ page: 1, page_size: 20 })
  }

  function retryGroupCatalog(): void {
    void Promise.allSettled([groupsQuery.refetch(), channelsQuery.refetch()])
  }

  function setPage(page: number): void {
    void navigateCollection({ ...filters, page })
  }

  function createKey(): void {
    restoreFocusRef.current = null
    setDrawerRoute({ mode: 'create' })
  }

  function openKey(accessKey: AccessKeyDto, trigger: HTMLElement): void {
    if (editOperation && editOperation.base.id !== accessKey.id) {
      checkEditOperation()
      return
    }
    if (rotateOperation && rotateOperation.base.id !== accessKey.id) {
      checkRotateOperation()
      return
    }
    restoreFocusRef.current = trigger
    setDrawerRoute({ mode: 'edit', accessKeyID: accessKey.id })
  }

  function checkCreateOperation(): void {
    if (!createOperation) return
    restoreFocusRef.current = null
    setDrawerRoute({ mode: 'create' })
  }

  function checkEditOperation(): void {
    if (!editOperation) return
    restoreFocusRef.current = null
    setDrawerRoute({ mode: 'edit', accessKeyID: editOperation.base.id })
  }

  function checkRotateOperation(): void {
    if (!rotateOperation) return
    restoreFocusRef.current = null
    setDrawerRoute({ mode: 'edit', accessKeyID: rotateOperation.base.id })
  }

  async function setDrawerOpen(open: boolean): Promise<void> {
    if (open) return
    // Every path reaching this close has already resolved the unsaved
    // question — confirmDiscard inside the drawer, or a completed
    // save/delete mutation. The route blocker reads dirty/blocked through
    // props that lag one commit behind the controller, so bypass the close
    // navigation like classic's live computed does.
    unsavedChanges.bypassNext()
    try {
      await navigateCollection(filters, null)
    } finally {
      unsavedChanges.consumeBypass()
    }
    const target = restoreFocusRef.current
    restoreFocusRef.current = null
    // Classic awaits nextTick then focuses — rAF lands after the commit that
    // removed the drawer from the DOM.
    requestAnimationFrame(() => target?.focus())
  }

  async function handleSaved(kind: 'created' | 'updated', name: string): Promise<void> {
    await setDrawerOpen(false)
    toast.show({ message: t(`accessKeys.toast.${kind}` as MessageId, { name }) })
  }

  async function handleDeleted(name: string): Promise<void> {
    setDeletionAnnouncement('')
    restoreFocusRef.current = null
    await setDrawerOpen(false)
    await applyInvalidationPlan(queryClient, mutationInvalidationPlans.accessKey.delete)
    if (!mountedRef.current) return
    setDeletionAnnouncement(t('accessKeys.delete.deletedAnnouncement', { name }))
    toast.show({ message: t('accessKeys.toast.deleted', { name }) })
    requestAnimationFrame(() => {
      const target = viewRootRef.current?.querySelector('button.access-key-create')
      if (target instanceof HTMLButtonElement && target.isConnected) target.focus()
    })
  }

  async function handleCostLimitsReset(name: string): Promise<void> {
    try {
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.accessKey.reset)
    } catch {
      void queryClient.invalidateQueries({ queryKey: accessKeyResources.collection.queryKey })
    }
    if (mountedRef.current) {
      toast.show({ message: t('accessKeys.toast.reset', { name }) })
    }
  }

  function handleRotated(name: string): void {
    toast.show({ message: t('accessKeys.toast.rotated', { name }) })
  }

  async function toggleStatus(accessKey: AccessKeyDto): Promise<void> {
    const statusControllers = statusControllersRef.current
    if (statusControllers.has(accessKey.id)) return
    const status = accessKey.status === 'active' ? 'disabled' : 'active'
    const controller = new AbortController()
    statusControllers.set(accessKey.id, controller)
    setPendingStatusIDs((previous) => new Set(previous).add(accessKey.id))
    try {
      try {
        await updateAccessKey(apiClient, accessKey.id, { status }, controller.signal)
      } catch (error: unknown) {
        if (!(error instanceof RequestCancelledError) && mountedRef.current) {
          toast.show({ message: t('accessKeys.actions.updateFailed'), tone: 'danger' })
        }
        return
      }
      if (!mountedRef.current) return
      try {
        await applyInvalidationPlan(queryClient, mutationInvalidationPlans.accessKey.update)
      } catch {
        void queryClient.invalidateQueries({ queryKey: accessKeyResources.collection.queryKey })
      }
      if (!mountedRef.current) return
      toast.show({
        message: t(status === 'active' ? 'accessKeys.toast.enabled' : 'accessKeys.toast.disabled', {
          name: accessKey.name,
        }),
      })
    } finally {
      if (statusControllers.get(accessKey.id) === controller) {
        statusControllers.delete(accessKey.id)
        setPendingStatusIDs((previous) => {
          const next = new Set(previous)
          next.delete(accessKey.id)
          return next
        })
      }
    }
  }

  const statusSummaryItems =
    data?.summary === undefined
      ? []
      : ([
          { value: undefined, id: 'accessKeys.collection.status.all', count: data.summary.total },
          { value: 'active' as const, id: statusFilterIds.active, count: data.summary.active },
          {
            value: 'disabled' as const,
            id: statusFilterIds.disabled,
            count: data.summary.disabled,
          },
        ] as const)

  return (
    <div ref={viewRootRef} {...stylex.props(styles.panel)} aria-busy={collectionBusy || undefined}>
      <div {...stylex.props(styles.headerRow)}>
        <div>
          <h3 {...stylex.props(styles.heading)}>{t('settings.credentials.accessTitle')}</h3>
          <p {...stylex.props(styles.headingDescription)}>
            {t('settings.credentials.accessDescription')}
          </p>
        </div>
        <Button
          className="access-key-create"
          size="sm"
          icon={<Plus size={15} aria-hidden />}
          label={t('accessKeys.create')}
          onClick={createKey}
        />
      </div>
      <span aria-live="polite" {...stylex.props(styles.refreshing)}>
        {pageRefreshing ? t('accessKeys.collection.loading') : ''}
      </span>

      <div {...stylex.props(styles.loginNotice)}>
        <Banner
          status="info"
          icon={<KeyRound size={13} aria-hidden />}
          title={t('accessKeys.loginNotice.title')}
          description={t('accessKeys.loginNotice.description')}
        />
      </div>

      {drawerOpen && (drawerRoute?.mode === 'create' || selected !== null) && (
        <AccessKeyDrawer
          open={drawerOpen}
          accessKey={drawerRoute?.mode === 'create' ? null : selected}
          groups={groupsQuery.data ?? []}
          channels={channelsQuery.data?.items ?? []}
          total={data?.summary.total ?? 0}
          groupCatalogState={groupCatalogState}
          createOperation={createOperation}
          editOperation={
            selected !== null && selected.id === editOperation?.base.id ? editOperation : null
          }
          rotateOperation={
            selected !== null && selected.id === rotateOperation?.base.id ? rotateOperation : null
          }
          onCreateOperation={setCreateOperation}
          onEditOperation={setEditOperation}
          onRotateOperation={setRotateOperation}
          onOpenChange={setDrawerOpen}
          onSaved={handleSaved}
          onRotated={handleRotated}
          onDeleted={handleDeleted}
          renderUnsavedDialog={false}
        />
      )}

      {createOperation && (
        <Banner
          status="warning"
          title={t(
            createOperation.state === 'reconciling'
              ? 'accessKeys.operation.reconciling'
              : 'accessKeys.operation.indeterminate',
          )}
          endContent={
            <Button
              variant="secondary"
              size="sm"
              label={t('accessKeys.operation.checkResult')}
              onClick={() => checkCreateOperation()}
            />
          }
        />
      )}
      {editOperation && (
        <Banner
          status="warning"
          title={t(
            editOperation.state === 'reconciling'
              ? 'accessKeys.operation.editReconciling'
              : 'accessKeys.operation.editIndeterminate',
            { name: editOperation.patch.name ?? editOperation.base.name },
          )}
          endContent={
            <Button
              variant="secondary"
              size="sm"
              label={t('accessKeys.operation.checkResult')}
              onClick={() => checkEditOperation()}
            />
          }
        />
      )}
      {rotateOperation && (
        <Banner
          status="warning"
          title={t(
            rotateOperation.state === 'reconciling'
              ? 'accessKeys.operation.rotateReconciling'
              : 'accessKeys.operation.rotateIndeterminate',
            { name: rotateOperation.base.name },
          )}
          endContent={
            <Button
              variant="secondary"
              size="sm"
              label={t('accessKeys.operation.checkResult')}
              onClick={() => checkRotateOperation()}
            />
          }
        />
      )}

      <p {...stylex.props(styles.srOnly)} aria-live="polite" aria-atomic="true">
        {deletionAnnouncement}
      </p>

      {accessKeysQuery.isPending || initialLoading ? (
        <div
          {...stylex.props(styles.skeletonGrid)}
          role="status"
          aria-label={t('accessKeys.collection.loading')}
        >
          {Array.from({ length: filters.page_size }, (_, index) => (
            <Skeleton key={index} height={96} radius={2} />
          ))}
        </div>
      ) : accessKeysQuery.isError && !data ? (
        <div role="alert">
          <EmptyState
            title={t('accessKeys.collection.errorTitle')}
            description={t('accessKeys.collection.errorDescription')}
            icon={<TriangleAlert size={20} />}
            actions={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void accessKeysQuery.refetch()}
              />
            }
          />
        </div>
      ) : data ? (
        <>
          {data.summary.total > 0 && (
            <div
              {...stylex.props(styles.summaryStrip)}
              role="group"
              aria-label={t('accessKeys.collection.summary.region')}
            >
              <span {...stylex.props(styles.summaryLabel)}>
                {t('accessKeys.collection.summary.current')}
              </span>
              <Selector
                size="sm"
                variant="input"
                label={t('accessKeys.collection.summary.region')}
                options={statusSummaryItems.map((item) => ({
                  value: item.value ?? 'all',
                  label: `${t(item.id)} (${intl.formatNumber(item.count)})`,
                }))}
                value={filters.status ?? 'all'}
                onChange={(value) =>
                  setStatusFilter(
                    value === 'all' ? undefined : (value as AccessKeyCollectionStatus),
                  )
                }
              />
            </div>
          )}

          {accessKeysQuery.isError && (
            <div role="status">
              <Banner
                status="warning"
                title={t('accessKeys.stale')}
                endContent={
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={() => void accessKeysQuery.refetch()}
                  />
                }
              />
            </div>
          )}
          {(groupsQuery.isError || channelsQuery.isError) && (
            <div role="status">
              <Banner
                status="warning"
                title={t('accessKeys.groupsStale')}
                endContent={
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={retryGroupCatalog}
                  />
                }
              />
            </div>
          )}

          {data.summary.total > 0 && (
            <div
              {...stylex.props(styles.filterBar)}
              role="group"
              aria-label={t('accessKeys.collection.filters.region')}
            >
              <label {...stylex.props(styles.filterField)}>
                <span {...stylex.props(styles.filterLabel)}>
                  {t('accessKeys.collection.filters.searchLabel')}
                </span>
                <TextInput
                  size="sm"
                  label={t('accessKeys.collection.filters.searchLabel')}
                  placeholder={t('accessKeys.collection.filters.searchPlaceholder')}
                  startIcon={<Search size={14} />}
                  value={searchDraft}
                  hasClear
                  onChange={onSearchChange}
                />
              </label>
              {hasFilterCriteria && (
                <span {...stylex.props(styles.filterResult)}>
                  <span aria-live="polite">
                    {t('accessKeys.collection.result', {
                      shown: intl.formatNumber(data.items.length),
                      total: intl.formatNumber(data.pagination.total_items),
                    })}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    label={t('accessKeys.collection.filters.reset')}
                    onClick={resetConditions}
                  />
                </span>
              )}
            </div>
          )}

          {collectionTransition ? (
            <div
              {...stylex.props(styles.skeletonGrid)}
              role="status"
              aria-label={t('accessKeys.collection.loading')}
            >
              {Array.from({ length: skeletonRows }, (_, index) => (
                <Skeleton key={index} height={96} radius={2} />
              ))}
            </div>
          ) : data.summary.total === 0 ? (
            <EmptyState
              title={t('accessKeys.emptyTitle')}
              description={t('accessKeys.emptyDescription')}
              icon={<KeyRound size={20} />}
              actions={
                <Button
                  className="access-key-create"
                  size="sm"
                  icon={<Plus size={15} aria-hidden />}
                  label={t('accessKeys.create')}
                  onClick={createKey}
                />
              }
            />
          ) : data.pagination.total_items === 0 && hasFilterCriteria ? (
            <EmptyState
              title={t('accessKeys.collection.noResultsTitle')}
              description={t('accessKeys.collection.noResultsDescription')}
              icon={<Search size={20} />}
              actions={
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('accessKeys.collection.filters.reset')}
                  onClick={resetConditions}
                />
              }
            />
          ) : data.items.length > 0 ? (
            <>
              <AccessKeyCollection
                ref={collectionRef}
                accessKeys={data.items}
                groups={groupsQuery.data ?? []}
                total={data.summary.total}
                filteredTotal={data.pagination.total_items}
                page={data.pagination.page}
                pageSize={data.pagination.page_size}
                busyIds={pendingStatusIDs}
                lockedIds={lockedAccessKeyIDs}
                onOpen={openKey}
                onToggle={(accessKey) => void toggleStatus(accessKey)}
                onDeleted={(name) => void handleDeleted(name)}
                onReset={(name) => void handleCostLimitsReset(name)}
              />
              <Pagination
                page={data.pagination.page}
                totalItems={data.pagination.total_items}
                totalPages={data.pagination.total_pages}
                pageSize={data.pagination.page_size}
                onChange={setPage}
                isDisabled={collectionBusy}
                size="sm"
              />
            </>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
