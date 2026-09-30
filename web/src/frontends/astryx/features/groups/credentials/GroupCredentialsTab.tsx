import * as stylex from '@stylexjs/stylex'
import {
  AlertDialog,
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
import { KeyRound, Plus, RefreshCw, Search } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useIntl } from 'react-intl'

import { ApiError } from '@shared/http/errors'
import type {
  CredentialCollectionDto,
  CredentialCollectionFilters,
  CredentialItemDto,
  CredentialObservationDto,
  CredentialStatus,
  CredentialTestResultDto,
} from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import {
  channelsQueryOptions,
  type ChannelCapabilitiesDto,
} from '@shared/control/resources/channels'
import {
  batchCredentials,
  cacheCredentialBatch,
  cacheCredentialItem,
  consumeCredentialResetCredit,
  credentialCollectionQueryOptions,
  downloadCredential,
  getCredentialDetail,
  refreshCredential as refreshCredentialRequest,
  refreshCredentialObservation,
  restoreCredential,
  testCredentialConnection,
} from '@shared/control/resources/credentials'
import {
  connectGroupCredentials,
  inspectGroupCredentialConnection,
  type CredentialStage,
} from '@shared/control/resources/credential-stages'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { controlQueryKeys } from '@shared/control/query-keys'
import { presentSubscriptionErrorKey } from '@shared/domain/import/subscription-error-presenter'
import { createUUID } from '@shared/lib/uuid'
import { pagePath } from '@shared/routing/page-routes'
import {
  constrainCredentialSearch,
  isCanonicalCredentialRouteQuery,
  parseCredentialRouteQuery,
  parseCredentialRouteState,
  serializeCredentialRouteQuery,
  type CredentialRouteState,
} from '@shared/routing/group-detail-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'

import { useCollectionLoading } from '../../../app/collection-loading'
import { useT } from '../../../app/i18n'
import { useAppServices } from '../../../app/services'
import { useDebouncedAction } from '../../../app/use-debounced-action'
import { DetailPanel } from '../../../components/DetailPanel'
import { InlineNotice } from '../../../components/InlineNotice'
import { LedgerRecordList, ledgerRecordStyles } from '../../../components/LedgerRecordList'
import { SubscriptionCredentialStager } from '../../import/SubscriptionCredentialStager'
import { CredentialTestDialog } from './CredentialTestDialog'
import { GroupCredentialBatchBar } from './GroupCredentialBatchBar'
import { GroupCredentialRecord } from './GroupCredentialRecord'
import { SubscriptionAccountCard } from './SubscriptionAccountCard'

const batchCredentialConcurrency = 4

// Module scope keeps the `Date.now()` reads out of render scope
// (react-hooks/purity).
function readyConnectionSignature(stages: CredentialStage[]): string | undefined {
  const now = Date.now()
  if (
    stages.length === 0 ||
    !stages.every(({ status, expires_at_ms }) => status === 'ready' && expires_at_ms > now)
  ) {
    return undefined
  }
  return stages
    .map(({ stage_id }) => stage_id)
    .sort()
    .join(',')
}

function partitionConnectionStages(stages: CredentialStage[]): {
  now: number
  ready: CredentialStage[]
} {
  const now = Date.now()
  const ready = stages.filter(
    ({ status, expires_at_ms }) => status === 'ready' && expires_at_ms > now,
  )
  return { now, ready }
}

const narrow = '@media (max-width: 860px)'
const small = '@media (max-width: 560px)'

const styles = stylex.create({
  root: {
    display: 'grid',
    minWidth: 0,
    paddingTop: 'var(--detail-panel-padding-top, var(--space-4))',
  },
  header: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-lg)',
    fontWeight: 650,
  },
  feedback: {
    marginBlockEnd: 'var(--space-3)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-feedback-danger-border, var(--color-danger))',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-text)',
    padding: 'var(--space-3)',
    lineHeight: 'var(--line-normal)',
    overflowWrap: 'anywhere',
  },
  summaryStrip: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
    paddingBlock: 'var(--space-2)',
  },
  summaryLabel: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 560,
  },
  tools: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(260px, 1fr) minmax(0, max-content)', [narrow]: '1fr' },
    alignItems: 'start',
    gap: '10px',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    marginBottom: 'var(--space-4)',
    paddingBlock: '13px var(--space-3)',
  },
  searchControls: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: '10px',
    width: { default: '420px', [small]: '100%' },
    maxWidth: '100%',
  },
  resetPlaceholder: {
    visibility: 'hidden',
    pointerEvents: 'none',
  },
  connect: {
    display: 'grid',
    gap: 'var(--space-3)',
    paddingBlock: 'var(--space-4)',
  },
  accounts: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))',
    alignItems: 'start',
    gap: 'var(--space-3)',
  },
  refreshing: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  pagination: {
    display: 'flex',
    justifyContent: 'flex-end',
    paddingBlock: 'var(--space-3)',
  },
})

function groupDetailHref(groupId: number): string {
  return `${pagePath('groups')}/${groupId}`
}

export function GroupCredentialsTab({
  groupId,
  channelId,
  connectionType,
}: {
  groupId: number
  channelId: string
  connectionType: 'api_key' | 'subscription'
}) {
  const t = useT()
  const intl = useIntl()
  const { apiClient, toast } = useAppServices()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const { rawSearch, searchStr, pathname } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
      pathname: state.location.pathname,
    }),
  })

  const filters = parseCredentialRouteQuery(rawSearch)
  const routeState = parseCredentialRouteState(rawSearch)
  const credentialsQuery = useQuery(credentialCollectionQueryOptions(apiClient, groupId, filters))
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const channelDescriptor = channelsQuery.data?.items.find(
    ({ channel_id }) => channel_id === channelId,
  )
  const authorizationMethods = channelDescriptor?.connection.authorization_methods ?? []
  const channelName = channelDescriptor?.name ?? channelId
  const channelNotices = channelDescriptor?.notices ?? []
  const channelCapabilities: ChannelCapabilitiesDto = channelDescriptor?.capabilities ?? {
    model_discovery: false,
    quota_observation: false,
    credential_actions: [],
    outbound_proxy: false,
  }

  const [searchDraft, setSearchDraft] = useState(filters.q ?? '')
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<number>>(new Set())
  const [pendingOperations, setPendingOperations] = useState<ReadonlySet<string>>(new Set())
  const [loadedDetails, setLoadedDetails] = useState<ReadonlyMap<number, CredentialItemDto>>(
    new Map(),
  )
  const [detailErrors, setDetailErrors] = useState<ReadonlyMap<number, string>>(new Map())
  const [observationErrors, setObservationErrors] = useState<ReadonlyMap<number, string>>(new Map())
  const [batchObservationPending, setBatchObservationPending] = useState<ReadonlySet<number>>(
    new Set(),
  )
  const [feedback, setFeedback] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<{ ids: number[]; mask?: string }>()
  const [resetTarget, setResetTarget] = useState<{
    item: CredentialItemDto
    idempotencyKey: string
  }>()
  const [credentialTestTarget, setCredentialTestTarget] = useState<CredentialItemDto>()
  const [credentialTestResult, setCredentialTestResult] = useState<CredentialTestResultDto>()
  const [credentialTestRequestFailed, setCredentialTestRequestFailed] = useState(false)
  const resetOperationKeysRef = useRef(new Map<number, string>())
  const [connectionWorkspaceOpen, setConnectionWorkspaceOpen] = useState(false)
  const [connectionStages, setConnectionStages] = useState<CredentialStage[]>([])
  const connectOperationKeyRef = useRef<string | undefined>(undefined)
  const [connectFeedback, setConnectFeedback] = useState('')
  const [connectionInspectionPending, setConnectionInspectionPending] = useState(false)
  const inspectedConnectionSignatureRef = useRef('')
  const inspectingConnectionSignatureRef = useRef('')
  const connectionInspectionControllerRef = useRef<AbortController | undefined>(undefined)
  const connectionInspectionOwnerRef = useRef(0)
  const credentialTestControllerRef = useRef<AbortController | undefined>(undefined)
  const credentialTestOwnerRef = useRef(0)
  const autoWrittenSignaturesRef = useRef(new Set<string>())
  const searchDebounce = useDebouncedAction(250)

  const collection = credentialsQuery.data
  const {
    initial: initialLoading,
    transition: collectionTransition,
    refreshing: collectionRefreshing,
    rows: skeletonRows,
  } = useCollectionLoading(
    {
      pending: credentialsQuery.isPending,
      placeholder: credentialsQuery.isPlaceholderData,
      fetching: credentialsQuery.isFetching,
      hasData: collection !== undefined,
      itemCount: collection?.items.length ?? 0,
    },
    { fallbackRows: 20 },
  )

  const selectedCount = selectedIds.size
  const selectedSubscriptionCredentialsReady = (collection?.items ?? [])
    .filter(({ credential_id }) => selectedIds.has(credential_id))
    .every(({ auth_state }) => auth_state === 'ready')
  const allVisibleSelected =
    (collection?.items.length ?? 0) > 0 &&
    (collection?.items ?? []).every(({ credential_id }) => selectedIds.has(credential_id))
  const batchBusy = [...pendingOperations].some((key) => key.startsWith('batch:'))
  const singleBusy = [...pendingOperations].some((key) => !key.startsWith('batch:'))
  // 抽屉只关心自己那一次写入——页面级 singleBusy 会把抽屉锁死连关闭都点不动。
  const connectBusy = [...pendingOperations].some((key) => key.endsWith(':connect'))
  const observationBatchBusy = pendingOperations.has('batch:observation')
  const bulkActionsBusy = batchBusy || singleBusy || connectBusy || credentialsQuery.isFetching
  const dialogBusy =
    deleteTarget === undefined
      ? false
      : deleteTarget.ids.length === 1
        ? pending(deleteTarget.ids[0]!)
        : batchBusy
  const resetDialogBusy =
    resetTarget === undefined ? false : pending(resetTarget.item.credential_id)
  const credentialTestPending =
    credentialTestTarget === undefined
      ? false
      : pendingOperations.has(operation(credentialTestTarget.credential_id, 'test'))
  const hasChangedConditions = filters.q !== undefined || filters.status !== undefined

  const summary = collection?.summary
  const statusSummaryItems = summary
    ? [
        {
          value: undefined,
          label: t('group.credentials.status.all'),
          count: summary.total,
          tone: 'neutral' as const,
        },
        {
          value: 'available',
          label: t('group.credentials.effective.available'),
          count: summary.available,
          tone: 'success' as const,
        },
        {
          value: 'cooldown',
          label: t('group.credentials.effective.cooldown'),
          count: summary.cooldown,
          tone: 'warning' as const,
        },
        {
          value: 'blacklisted',
          label: t('group.credentials.effective.blacklisted'),
          count: summary.blacklisted,
          tone: 'danger' as const,
        },
        {
          value: 'disabled',
          label: t('group.credentials.effective.disabled'),
          count: summary.disabled,
          tone: 'neutral' as const,
        },
      ]
    : []

  function updateRoute(
    next: CredentialCollectionFilters,
    replace = false,
    state: CredentialRouteState = routeState,
  ): void {
    void navigate({
      to: groupDetailHref(groupId),
      search: serializeCredentialRouteQuery(next, state),
      replace,
      resetScroll: false,
    })
  }

  // Classic watch(route.query, deep, immediate): mirror the parsed query into
  // the draft (render-time adjustment keyed on the raw URL), and canonicalize
  // non-canonical query params via a history replace (effect).
  const parsedRouteQ = parseCredentialRouteQuery(rawSearch).q ?? ''
  const [lastSearchStr, setLastSearchStr] = useState(searchStr)
  if (searchStr !== lastSearchStr) {
    setLastSearchStr(searchStr)
    setSearchDraft(parsedRouteQ)
  }
  useEffect(() => {
    // Pending transition: the outgoing route still renders while location has
    // moved — a late canonicalization must not resurrect this page.
    if (pathname !== groupDetailHref(groupId)) return
    searchDebounce.cancel()
    const next = parseCredentialRouteQuery(rawSearch)
    const state = parseCredentialRouteState(rawSearch)
    if (!isCanonicalCredentialRouteQuery(rawSearch, next, state)) {
      void navigate({
        to: groupDetailHref(groupId),
        search: serializeCredentialRouteQuery(next, state),
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the URL string
  }, [searchStr])

  // Classic watch([status, q, page, page_size]): selection does not survive a
  // filter/page change.
  const filterSignature = `${filters.status ?? ''}${filters.q ?? ''}${filters.page}${filters.page_size}`
  const [lastFilterSignature, setLastFilterSignature] = useState(filterSignature)
  if (lastFilterSignature !== filterSignature) {
    setLastFilterSignature(filterSignature)
    setSelectedIds(new Set())
  }

  // Classic watch(groupId): reset all per-group transient state.
  const [lastGroupId, setLastGroupId] = useState(groupId)
  if (lastGroupId !== groupId) {
    setLastGroupId(groupId)
    resetCredentialTestDrafts()
    setConnectionWorkspaceOpen(false)
    setConnectionStages([])
    setLoadedDetails(new Map())
    setDetailErrors(new Map())
    setObservationErrors(new Map())
    setBatchObservationPending(new Set())
  }

  // Ref-side half of the group switch: abort in-flight credential tests and
  // connection inspection so their callbacks cannot write into the new group.
  const groupIdSyncRef = useRef(groupId)
  useEffect(() => {
    if (groupIdSyncRef.current === groupId) return
    groupIdSyncRef.current = groupId
    credentialTestOwnerRef.current += 1
    credentialTestControllerRef.current?.abort()
    credentialTestControllerRef.current = undefined
    connectionInspectionOwnerRef.current += 1
    connectionInspectionControllerRef.current?.abort()
    connectionInspectionControllerRef.current = undefined
    setConnectionInspectionPending(false)
    inspectedConnectionSignatureRef.current = ''
    inspectingConnectionSignatureRef.current = ''
  }, [groupId])

  // Page correction: when the server clamps a deep page, mirror the
  // correction into the URL (classic parity).
  const totalPages = collection?.pagination.total_pages
  const isPlaceholder = credentialsQuery.isPlaceholderData
  useEffect(() => {
    if (!isPlaceholder && totalPages !== undefined && totalPages > 0 && filters.page > totalPages) {
      updateRoute({ ...filters, page: totalPages }, true)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on server pagination
  }, [totalPages, filters.page, isPlaceholder])

  function setFilter(
    patch: Partial<Pick<CredentialCollectionFilters, 'q' | 'status' | 'page_size'>>,
  ): void {
    updateRoute({ ...filters, ...patch, page: 1 })
  }

  function scheduleSearch(): void {
    searchDebounce.schedule(() => {
      setFilter({ q: constrainCredentialSearch(searchDraftRef.current) })
    })
  }
  // Debounced callback reads the freshest draft — a render-closed capture
  // would drop keystrokes typed inside the debounce window.
  const searchDraftRef = useRef(searchDraft)
  useEffect(() => {
    searchDraftRef.current = searchDraft
  })

  function resetFilters(): void {
    searchDebounce.cancel()
    setSearchDraft('')
    updateRoute({ page: 1, page_size: filters.page_size })
  }

  function setStatus(value: string | undefined): void {
    setFilter({ status: value as CredentialStatus | undefined })
  }

  function setPage(page: number): void {
    updateRoute({ ...filters, page })
  }

  function setPageSize(pageSize: number): void {
    if (pageSize === 20 || pageSize === 50 || pageSize === 100) {
      setFilter({ page_size: pageSize })
    }
  }

  function setExpanded(id: number, expanded: boolean): void {
    const next = new Set(routeState.expandedCredentialIDs)
    if (expanded) next.add(id)
    else next.delete(id)
    updateRoute(filters, false, {
      ...routeState,
      expandedCredentialIDs: [...next],
    })
  }

  function credentialExpanded(id: number): boolean {
    return routeState.expandedCredentialIDs.includes(id)
  }

  function setSelected(id: number, checked: boolean): void {
    const next = new Set(selectedIds)
    if (checked) next.add(id)
    else next.delete(id)
    setSelectedIds(next)
  }

  function setAllVisible(checked: boolean): void {
    const next = new Set(selectedIds)
    for (const { credential_id } of collection?.items ?? []) {
      if (checked) next.add(credential_id)
      else next.delete(credential_id)
    }
    setSelectedIds(next)
  }

  function currentSelectionContext(): string {
    return JSON.stringify({
      groupId,
      status: filters.status ?? null,
      query: filters.q ?? null,
      page: filters.page,
      pageSize: filters.page_size,
    })
  }

  function restoreVisibleFailedSelection(context: string, failedIDs: ReadonlySet<number>): void {
    if (context !== currentSelectionContext()) return
    const visibleIDs = new Set((collection?.items ?? []).map(({ credential_id }) => credential_id))
    setSelectedIds(new Set([...failedIDs].filter((id) => visibleIDs.has(id))))
  }

  function operation(id: number, action: string): string {
    return `${id}:${action}`
  }

  function pending(id: number): boolean {
    return [...pendingOperations].some((value) => value.startsWith(`${id}:`))
  }

  function observationRefreshing(id: number): boolean {
    return batchObservationPending.has(id) || pendingOperations.has(operation(id, 'observation'))
  }

  function observationError(id: number): string {
    return observationErrors.get(id) ?? ''
  }

  function clearObservationError(id: number): void {
    if (!observationErrors.has(id)) return
    const next = new Map(observationErrors)
    next.delete(id)
    setObservationErrors(next)
  }

  function setObservationError(id: number, message: string): void {
    const next = new Map(observationErrors)
    next.set(id, message)
    setObservationErrors(next)
  }

  function finishBatchObservation(id: number): void {
    if (!batchObservationPending.has(id)) return
    const next = new Set(batchObservationPending)
    next.delete(id)
    setBatchObservationPending(next)
  }

  function rowBusy(id: number): boolean {
    return (
      batchBusy ||
      [...pendingOperations].some(
        (value) => value.startsWith(`${id}:`) && value !== operation(id, 'usage'),
      )
    )
  }

  function detailBusy(id: number): boolean {
    return pendingOperations.has(operation(id, 'usage'))
  }

  function detailLoaded(id: number): boolean {
    return loadedDetails.has(id)
  }

  function detailError(id: number): string {
    return detailErrors.get(id) ?? ''
  }

  function clearDetailState(id: number): void {
    const loaded = new Map(loadedDetails)
    loaded.delete(id)
    setLoadedDetails(loaded)
    const errors = new Map(detailErrors)
    errors.delete(id)
    setDetailErrors(errors)
  }

  function setPending(id: number | 'batch', action: string, value: boolean): void {
    const next = new Set(pendingOperations)
    const key = id === 'batch' ? `batch:${action}` : operation(id, action)
    if (value) next.add(key)
    else next.delete(key)
    setPendingOperations(next)
  }

  function cachedCurrentCredential(id: number): CredentialItemDto | undefined {
    return queryClient
      .getQueryData<CredentialCollectionDto>(controlQueryKeys.groups.credentials(groupId, filters))
      ?.items.find(({ credential_id }) => credential_id === id)
  }

  function withPreservedDetail(
    item: CredentialItemDto,
    source: CredentialItemDto,
  ): CredentialItemDto {
    if (item.secret_version !== source.secret_version) return item

    const preservedItem: CredentialItemDto = {
      ...item,
      ...(source.last_used_at_ms === undefined ? {} : { last_used_at_ms: source.last_used_at_ms }),
      ...(source.daily_usage === undefined ? {} : { daily_usage: source.daily_usage }),
    }
    const sourceObservation = source.observation
    const targetObservation = item.observation
    if (
      !targetObservation?.snapshot ||
      !sourceObservation?.snapshot ||
      targetObservation.observation_version !== sourceObservation.observation_version ||
      targetObservation.observed_at_ms !== sourceObservation.observed_at_ms
    ) {
      return preservedItem
    }

    const observedUsageByWindow = new Map(
      sourceObservation.snapshot.quota_windows.flatMap((window) =>
        window.observed_usage === undefined ? [] : [[window.id, window.observed_usage] as const],
      ),
    )
    if (observedUsageByWindow.size === 0) return preservedItem

    return {
      ...preservedItem,
      observation: {
        ...targetObservation,
        snapshot: {
          ...targetObservation.snapshot,
          quota_windows: targetObservation.snapshot.quota_windows.map((window) => {
            const observedUsage = observedUsageByWindow.get(window.id)
            return window.observed_usage !== undefined || observedUsage === undefined
              ? window
              : { ...window, observed_usage: observedUsage }
          }),
        },
      },
    }
  }

  function credentialWithDetail(item: CredentialItemDto): CredentialItemDto {
    const detail = loadedDetails.get(item.credential_id)
    return detail === undefined ? item : withPreservedDetail(item, detail)
  }

  async function refetchGroupSummary(): Promise<void> {
    await queryClient.refetchQueries(
      { queryKey: controlQueryKeys.groups.summary(groupId), exact: true },
      { throwOnError: true },
    )
  }

  async function invalidateReconciliationQueries(): Promise<void> {
    try {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: controlQueryKeys.groups.credentials(groupId, filters),
          exact: true,
          refetchType: 'none',
        }),
        queryClient.invalidateQueries({
          queryKey: controlQueryKeys.groups.summary(groupId),
          exact: true,
          refetchType: 'none',
        }),
      ])
    } catch {
      // The reconciliation warning remains actionable even if invalidation fails.
    }
  }

  async function refetchActiveCredentialPage(): Promise<void> {
    await queryClient.refetchQueries(
      {
        queryKey: controlQueryKeys.groups.credentials(groupId, filters),
        exact: true,
        type: 'active',
      },
      { throwOnError: true },
    )
  }

  async function invalidateScheduleQueries(): Promise<void> {
    await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.update)
  }

  async function reconcileItem(result: CredentialItemDto, refetchActive: boolean): Promise<void> {
    try {
      const current = cachedCurrentCredential(result.credential_id)
      if (current !== undefined && current.secret_version !== result.secret_version) {
        clearDetailState(result.credential_id)
      }
      const reconciled = current === undefined ? result : withPreservedDetail(result, current)
      await cacheCredentialItem(queryClient, groupId, reconciled)
      if (refetchActive) {
        await refetchActiveCredentialPage()
        const refreshed = cachedCurrentCredential(result.credential_id)
        if (refreshed !== undefined) {
          await cacheCredentialItem(
            queryClient,
            groupId,
            withPreservedDetail(refreshed, reconciled),
          )
        }
      }
      await refetchGroupSummary()
    } catch {
      setFeedback(t('group.credentials.reconcileFailed'))
      await invalidateReconciliationQueries()
    }
    await invalidateScheduleQueries()
  }

  async function cacheObservation(
    item: CredentialItemDto,
    observation: CredentialObservationDto,
  ): Promise<void> {
    const current = cachedCurrentCredential(item.credential_id)
    const reconciled = { ...(current ?? item), observation }
    await cacheCredentialItem(queryClient, groupId, reconciled)
    await invalidateScheduleQueries()
  }

  async function refreshObservation(item: CredentialItemDto): Promise<void> {
    if (pending(item.credential_id) || observationBatchBusy) return
    setFeedback('')
    clearObservationError(item.credential_id)
    setPending(item.credential_id, 'observation', true)
    try {
      const observation = await refreshCredentialObservation(apiClient, groupId, item.credential_id)
      clearDetailState(item.credential_id)
      await reconcileItem({ ...item, observation }, true)
    } catch (cause) {
      setObservationError(
        item.credential_id,
        t(
          presentSubscriptionErrorKey(
            cause,
            'group.credentials.subscription.syncFailed',
          ) as MessageId,
        ),
      )
    } finally {
      setPending(item.credential_id, 'observation', false)
    }
  }

  async function refreshCredentialToken(item: CredentialItemDto): Promise<void> {
    if (pending(item.credential_id)) return
    setFeedback('')
    setPending(item.credential_id, 'refresh-credential', true)
    try {
      const result = await refreshCredentialRequest(apiClient, groupId, item.credential_id)
      clearDetailState(item.credential_id)
      await reconcileItem(result, true)
      toast.show({
        message: t('group.credentials.subscription.refreshCredentialSucceeded'),
        tone: 'success',
      })
    } catch (cause) {
      setFeedback(
        t(
          presentSubscriptionErrorKey(
            cause,
            'group.credentials.subscription.refreshCredentialFailed',
          ) as MessageId,
        ),
      )
      await Promise.allSettled([refetchActiveCredentialPage(), refetchGroupSummary()])
    } finally {
      setPending(item.credential_id, 'refresh-credential', false)
    }
  }

  function downloadJSONFile(filename: string, value: unknown): void {
    const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], {
      type: 'application/json;charset=utf-8',
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    document.body.append(anchor)
    anchor.click()
    anchor.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  async function downloadCredentialFile(item: CredentialItemDto): Promise<void> {
    if (pending(item.credential_id)) return
    setFeedback('')
    setPending(item.credential_id, 'download', true)
    try {
      const result = await downloadCredential(apiClient, groupId, item.credential_id)
      downloadJSONFile(result.filename, result.credential)
      toast.show({
        message: t('group.credentials.subscription.downloadSucceeded'),
        tone: 'success',
      })
    } catch (cause) {
      setFeedback(
        t(
          presentSubscriptionErrorKey(
            cause,
            'group.credentials.subscription.downloadFailed',
          ) as MessageId,
        ),
      )
    } finally {
      setPending(item.credential_id, 'download', false)
    }
  }

  async function syncSelectedObservations(): Promise<void> {
    const selected = selectedIds
    const items = (collection?.items ?? []).filter(({ credential_id }) =>
      selected.has(credential_id),
    )
    if (
      items.length === 0 ||
      items.some(({ auth_state }) => auth_state !== 'ready') ||
      bulkActionsBusy ||
      !channelCapabilities.quota_observation
    ) {
      return
    }
    const selectionContext = currentSelectionContext()
    setFeedback('')
    setBatchObservationPending(new Set(items.map(({ credential_id }) => credential_id)))
    setPending('batch', 'observation', true)
    let cursor = 0
    let succeeded = 0
    let failed = 0
    const failedIDs = new Set<number>()
    const worker = async () => {
      while (cursor < items.length) {
        const item = items[cursor]
        cursor += 1
        if (item === undefined) return
        clearObservationError(item.credential_id)
        setPending(item.credential_id, 'observation', true)
        try {
          const observation = await refreshCredentialObservation(
            apiClient,
            groupId,
            item.credential_id,
          )
          clearDetailState(item.credential_id)
          await cacheObservation(item, observation)
          succeeded += 1
        } catch (cause) {
          failed += 1
          failedIDs.add(item.credential_id)
          setObservationError(
            item.credential_id,
            t(
              presentSubscriptionErrorKey(
                cause,
                'group.credentials.subscription.syncFailed',
              ) as MessageId,
            ),
          )
        } finally {
          finishBatchObservation(item.credential_id)
          setPending(item.credential_id, 'observation', false)
        }
      }
    }
    try {
      await Promise.all(
        Array.from({ length: Math.min(batchCredentialConcurrency, items.length) }, () => worker()),
      )
      restoreVisibleFailedSelection(selectionContext, failedIDs)
      toast.show({
        message: t('group.credentials.batch.syncResult', {
          succeeded: intl.formatNumber(succeeded),
          failed: intl.formatNumber(failed),
        }),
        tone: failed === 0 ? 'success' : succeeded === 0 ? 'danger' : 'warning',
        duration: 4_000,
      })
    } finally {
      setBatchObservationPending(new Set())
      setPending('batch', 'observation', false)
    }
  }

  async function downloadSelectedCredentials(): Promise<void> {
    const selected = selectedIds
    const items = (collection?.items ?? []).filter(({ credential_id }) =>
      selected.has(credential_id),
    )
    if (items.length === 0 || bulkActionsBusy) return
    const selectionContext = currentSelectionContext()
    setFeedback('')
    setPending('batch', 'export', true)
    let cursor = 0
    let succeeded = 0
    let failed = 0
    const failedIDs = new Set<number>()
    const worker = async () => {
      while (cursor < items.length) {
        const item = items[cursor]
        cursor += 1
        if (item === undefined) return
        try {
          const result = await downloadCredential(apiClient, groupId, item.credential_id)
          downloadJSONFile(result.filename, result.credential)
          succeeded += 1
        } catch {
          failed += 1
          failedIDs.add(item.credential_id)
        }
      }
    }
    try {
      await Promise.all(
        Array.from({ length: Math.min(batchCredentialConcurrency, items.length) }, () => worker()),
      )
      restoreVisibleFailedSelection(selectionContext, failedIDs)
      toast.show({
        message: t('group.credentials.batch.downloadResult', {
          succeeded: intl.formatNumber(succeeded),
          failed: intl.formatNumber(failed),
        }),
        tone: failed === 0 ? 'success' : succeeded === 0 ? 'danger' : 'warning',
        duration: 4_000,
      })
    } finally {
      setPending('batch', 'export', false)
    }
  }

  async function loadCredentialUsage(item: CredentialItemDto): Promise<void> {
    if (batchBusy || pending(item.credential_id) || detailLoaded(item.credential_id)) return
    const errors = new Map(detailErrors)
    errors.delete(item.credential_id)
    setDetailErrors(errors)
    setPending(item.credential_id, 'usage', true)
    try {
      const detail = await getCredentialDetail(apiClient, groupId, item.credential_id)
      await cacheCredentialItem(queryClient, groupId, detail.credential)
      const loaded = new Map(loadedDetails)
      loaded.set(item.credential_id, detail.credential)
      setLoadedDetails(loaded)
    } catch {
      const nextErrors = new Map(detailErrors)
      nextErrors.set(item.credential_id, t('group.credentials.loadFailed'))
      setDetailErrors(nextErrors)
    } finally {
      setPending(item.credential_id, 'usage', false)
    }
  }

  function openResetCreditDialog(item: CredentialItemDto): void {
    if (pending(item.credential_id)) return
    const idempotencyKey = resetOperationKeysRef.current.get(item.credential_id) ?? createUUID()
    resetOperationKeysRef.current.set(item.credential_id, idempotencyKey)
    setResetTarget({ item, idempotencyKey })
  }

  async function confirmResetCredit(): Promise<void> {
    const target = resetTarget
    if (target === undefined || pending(target.item.credential_id)) return
    setFeedback('')
    setPending(target.item.credential_id, 'reset-credit', true)
    try {
      const result = await consumeCredentialResetCredit(
        apiClient,
        groupId,
        target.item.credential_id,
        target.idempotencyKey,
      )
      const observationPending = result.observation_pending || result.observation?.state !== 'fresh'
      if (result.observation) {
        await reconcileItem({ ...target.item, observation: result.observation }, false)
      } else {
        try {
          await refetchActiveCredentialPage()
        } catch {
          await invalidateReconciliationQueries()
        }
        await invalidateScheduleQueries()
      }
      if (observationPending) {
        setFeedback(t('group.credentials.subscription.consumeResetCreditPending'))
      } else {
        toast.show({
          message: t('group.credentials.subscription.consumeResetCreditSucceeded'),
          tone: 'success',
        })
      }
      resetOperationKeysRef.current.delete(target.item.credential_id)
      setResetTarget(undefined)
    } catch (cause) {
      if (cause instanceof ApiError && cause.code !== 'RESET_CREDIT_OUTCOME_UNKNOWN') {
        resetOperationKeysRef.current.delete(target.item.credential_id)
      }
      setFeedback(
        t(
          presentSubscriptionErrorKey(
            cause,
            'group.credentials.subscription.consumeResetCreditFailed',
          ) as MessageId,
        ),
      )
    } finally {
      setPending(target.item.credential_id, 'reset-credit', false)
    }
  }

  const readyConnectionStages = connectionStages.filter(({ status }) => status === 'ready')

  function resetConnectionInspection(): void {
    connectionInspectionOwnerRef.current += 1
    connectionInspectionControllerRef.current?.abort()
    connectionInspectionControllerRef.current = undefined
    setConnectionInspectionPending(false)
    inspectedConnectionSignatureRef.current = ''
    inspectingConnectionSignatureRef.current = ''
  }

  async function inspectConnectionStages(signature: string, stageIDs: string[]): Promise<void> {
    connectionInspectionControllerRef.current?.abort()
    const controller = new AbortController()
    const owner = ++connectionInspectionOwnerRef.current
    connectionInspectionControllerRef.current = controller
    setConnectionInspectionPending(true)
    inspectingConnectionSignatureRef.current = signature
    setConnectFeedback('')
    try {
      const result = await inspectGroupCredentialConnection(
        apiClient,
        groupId,
        stageIDs,
        controller.signal,
      )
      const inspectedStageIDs = new Set(stageIDs)
      if (
        controller.signal.aborted ||
        owner !== connectionInspectionOwnerRef.current ||
        readyConnectionSignature(
          connectionStages.filter(({ stage_id }) => inspectedStageIDs.has(stage_id)),
        ) !== signature
      ) {
        return
      }
      const duplicated = new Set(result.duplicated_stage_ids)
      inspectedConnectionSignatureRef.current = signature
      setConnectionStages(
        connectionStages.map((stage) =>
          inspectedStageIDs.has(stage.stage_id)
            ? { ...stage, duplicate: duplicated.has(stage.stage_id) }
            : stage,
        ),
      )
    } catch (cause) {
      if (controller.signal.aborted || owner !== connectionInspectionOwnerRef.current) return
      setConnectFeedback(
        t(
          presentSubscriptionErrorKey(
            cause,
            'group.credentials.subscription.connectFailed',
          ) as MessageId,
        ),
      )
    } finally {
      if (owner === connectionInspectionOwnerRef.current) {
        connectionInspectionControllerRef.current = undefined
        setConnectionInspectionPending(false)
        inspectingConnectionSignatureRef.current = ''
      }
    }
  }

  function setConnectionWorkspace(open: boolean): void {
    if (!open && connectBusy) return
    setConnectionWorkspaceOpen(open)
    if (!open) {
      resetConnectionInspection()
      setConnectionStages([])
      connectOperationKeyRef.current = undefined
      setConnectFeedback('')
      autoWrittenSignaturesRef.current.clear()
    }
  }

  async function saveConnectedAccounts(): Promise<void> {
    const { now, ready } = partitionConnectionStages(connectionStages)
    const signature = readyConnectionSignature(ready)
    if (ready.length === 0 || signature === undefined) {
      if (connectionStages.some(({ status }) => status === 'ready')) {
        setConnectionStages(
          connectionStages.map((stage) =>
            stage.status === 'ready' && stage.expires_at_ms <= now
              ? { ...stage, status: 'expired' }
              : stage,
          ),
        )
        setConnectFeedback(t('common.subscriptionErrors.stageExpired'))
      }
      return
    }
    if (connectBusy || connectionInspectionPending) return
    if (inspectedConnectionSignatureRef.current !== signature) {
      void inspectConnectionStages(
        signature,
        ready.map(({ stage_id }) => stage_id),
      )
      return
    }
    const duplicatedAccounts = ready
      .filter(({ duplicate }) => duplicate)
      .map(({ account }) => account.email_mask || t('import.subscription.pendingAccount'))
    setConnectFeedback('')
    let succeeded = false
    setPending(0, 'connect', true)
    try {
      connectOperationKeyRef.current ??= createUUID()
      const result = await connectGroupCredentials(
        apiClient,
        groupId,
        ready.map(({ stage_id }) => stage_id),
        connectOperationKeyRef.current,
      )
      await refetchActiveCredentialPage()
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.update)
      void queryClient.invalidateQueries({
        queryKey: controlQueryKeys.groups.summary(groupId),
        exact: true,
        refetchType: 'active',
      })
      toast.show({
        message: t(
          result.credentials_duplicated > 0
            ? duplicatedAccounts.length === result.credentials_duplicated
              ? 'group.credentials.subscription.connectDuplicatedAccounts'
              : 'group.credentials.subscription.connectDuplicated'
            : 'group.credentials.subscription.connectSucceeded',
          {
            added: intl.formatNumber(result.credentials_added),
            duplicated: intl.formatNumber(result.credentials_duplicated),
            accounts: duplicatedAccounts.join(', '),
          },
        ),
        tone: result.credentials_added === 0 ? 'warning' : 'success',
        duration: 4_000,
      })
      succeeded = true
    } catch (cause) {
      setConnectFeedback(
        t(
          presentSubscriptionErrorKey(
            cause,
            'group.credentials.subscription.connectFailed',
          ) as MessageId,
        ),
      )
    } finally {
      setPending(0, 'connect', false)
      if (succeeded) setConnectionWorkspace(false)
    }
  }

  // 授权就绪即写入，省掉一次多余的确认点击。同时发起多个授权时等全部落定
  // 再一次性写入；写入前先标出将跳过的重复账号；有失败的暂存时不自动写入。
  useEffect(() => {
    const stages = connectionStages
    if (!connectionWorkspaceOpen || connectBusy || stages.length === 0) return
    const signature = readyConnectionSignature(stages)
    if (!signature) return
    if (inspectedConnectionSignatureRef.current !== signature) {
      if (inspectingConnectionSignatureRef.current !== signature) {
        void inspectConnectionStages(
          signature,
          stages.map(({ stage_id }) => stage_id),
        )
      }
      return
    }
    if (autoWrittenSignaturesRef.current.has(signature)) return
    autoWrittenSignaturesRef.current.add(signature)
    void saveConnectedAccounts()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mirrors the deep stage watch
  }, [connectionStages, connectionWorkspaceOpen, connectBusy])

  function openConnectionWorkspace(): void {
    resetConnectionInspection()
    setConnectionStages([])
    connectOperationKeyRef.current = undefined
    setConnectFeedback('')
    autoWrittenSignaturesRef.current.clear()
    setConnectionWorkspaceOpen(true)
  }

  useEffect(
    () => () => {
      resetConnectionInspection()
      credentialTestOwnerRef.current += 1
      credentialTestControllerRef.current?.abort()
    },
    [],
  )

  async function reconcileBatch(
    result: Awaited<ReturnType<typeof batchCredentials>>,
  ): Promise<void> {
    try {
      await cacheCredentialBatch(queryClient, groupId, result)
      await refetchActiveCredentialPage()
      await refetchGroupSummary()
    } catch {
      setFeedback(t('group.credentials.reconcileFailed'))
      await invalidateReconciliationQueries()
    }
    await invalidateScheduleQueries()
  }

  function clearDeletedRouteState(ids: readonly number[]): void {
    const deleted = new Set(ids)
    const next: CredentialRouteState = {
      expandedCredentialIDs: routeState.expandedCredentialIDs.filter((id) => !deleted.has(id)),
    }
    updateRoute(filters, true, next)
  }

  async function mutateItem(item: CredentialItemDto, action: 'restore'): Promise<void> {
    if (batchBusy || pending(item.credential_id)) return
    setFeedback('')
    setPending(item.credential_id, action, true)
    try {
      const result = await restoreCredential(apiClient, groupId, item.credential_id)
      await reconcileItem(result, true)
    } catch {
      setFeedback(t('group.credentials.restoreFailed'))
    } finally {
      setPending(item.credential_id, action, false)
    }
  }

  // Render-safe half of resetCredentialTestState: no ref access, so it can run
  // during the group-switch render adjustment. The in-flight test aborts run in
  // the groupId-keyed effect below.
  function resetCredentialTestDrafts(): void {
    const credentialID = credentialTestTarget?.credential_id
    if (credentialID !== undefined) {
      setPending(credentialID, 'test', false)
    }
    setCredentialTestTarget(undefined)
    setCredentialTestResult(undefined)
    setCredentialTestRequestFailed(false)
  }

  function resetCredentialTestState(): void {
    credentialTestOwnerRef.current += 1
    credentialTestControllerRef.current?.abort()
    credentialTestControllerRef.current = undefined
    resetCredentialTestDrafts()
  }

  function setCredentialTestOpen(open: boolean): void {
    if (open || credentialTestPending) return
    resetCredentialTestState()
  }

  function viewCredentialTestLog(logID: string): void {
    void navigate({
      to: pagePath('logs'),
      search: { selected_request_id: logID },
    })
  }

  async function openCredentialTest(item: CredentialItemDto): Promise<void> {
    if (connectionType !== 'api_key' || batchBusy || pending(item.credential_id)) return

    credentialTestControllerRef.current?.abort()
    const controller = new AbortController()
    const owner = ++credentialTestOwnerRef.current
    const activeGroupId = groupId
    credentialTestControllerRef.current = controller
    setCredentialTestTarget(item)
    setCredentialTestResult(undefined)
    setCredentialTestRequestFailed(false)
    setPending(item.credential_id, 'test', true)
    try {
      const result = await testCredentialConnection(
        apiClient,
        activeGroupId,
        item.credential_id,
        controller.signal,
      )
      if (owner !== credentialTestOwnerRef.current || activeGroupId !== groupId) return
      setCredentialTestResult(result)
      if (result.recovered) {
        await Promise.allSettled([refetchActiveCredentialPage(), refetchGroupSummary()])
        await invalidateScheduleQueries()
      }
    } catch {
      if (owner !== credentialTestOwnerRef.current || activeGroupId !== groupId) return
      setCredentialTestRequestFailed(true)
    } finally {
      if (owner === credentialTestOwnerRef.current && activeGroupId === groupId) {
        credentialTestControllerRef.current = undefined
        setPending(item.credential_id, 'test', false)
      }
    }
  }

  async function confirmDelete(): Promise<void> {
    const target = deleteTarget
    if (!target || dialogBusy || batchBusy) return
    setFeedback('')
    if (target.ids.length === 1) {
      const id = target.ids[0]!
      setPending(id, 'delete', true)
      let result: Awaited<ReturnType<typeof batchCredentials>>
      try {
        result = await batchCredentials(apiClient, groupId, {
          action: 'delete',
          credential_ids: [id],
        })
      } catch {
        setFeedback(t('group.credentials.deleteFailed'))
        setPending(id, 'delete', false)
        return
      }
      try {
        await reconcileBatch(result)
        setDeleteTarget(undefined)
        const next = new Set(selectedIds)
        next.delete(id)
        setSelectedIds(next)
        clearDeletedRouteState([id])
      } finally {
        setPending(id, 'delete', false)
      }
      return
    }
    if (await runBatch('delete', target.ids)) setDeleteTarget(undefined)
  }

  async function runBatch(action: 'delete', ids = [...selectedIds]): Promise<boolean> {
    if (ids.length === 0 || batchBusy || singleBusy) return false
    setFeedback('')
    setPending('batch', action, true)
    let result: Awaited<ReturnType<typeof batchCredentials>>
    try {
      result = await batchCredentials(apiClient, groupId, { action, credential_ids: ids })
    } catch {
      setFeedback(t('group.credentials.batch.failed'))
      setPending('batch', action, false)
      return false
    }
    try {
      await reconcileBatch(result)
      setSelectedIds(new Set())
      if (action === 'delete') clearDeletedRouteState(ids)
      return true
    } finally {
      setPending('batch', action, false)
    }
  }

  const importHref = `${pagePath('import')}?mode=existing&group_id=${groupId}`

  return (
    <section
      {...stylex.props(styles.root)}
      aria-labelledby="group-credentials-heading"
      aria-busy={credentialsQuery.isFetching || observationBatchBusy ? true : undefined}
    >
      <div {...stylex.props(styles.header)}>
        <h2 id="group-credentials-heading" {...stylex.props(styles.title)}>
          {connectionType === 'subscription'
            ? t('group.credentials.subscription.title')
            : t('group.credentials.title')}
        </h2>
        {connectionType === 'subscription' && authorizationMethods.length > 0 ? (
          <Button
            size="sm"
            isDisabled={bulkActionsBusy}
            icon={<Plus size={16} aria-hidden="true" />}
            label={t('group.credentials.subscription.connect')}
            onClick={openConnectionWorkspace}
          />
        ) : (
          <Button
            size="sm"
            icon={<Plus size={16} aria-hidden="true" />}
            label={t('group.credentials.add')}
            onClick={() => void navigate({ to: importHref })}
          />
        )}
      </div>

      {connectionType === 'subscription' && (
        <DetailPanel
          isOpen={connectionWorkspaceOpen}
          onOpenChange={setConnectionWorkspace}
          title={t('group.credentials.subscription.connect')}
          subtitle={t('group.credentials.subscription.connectDescription')}
          dismissible={!connectBusy}
          footer={
            <>
              <Button
                variant="secondary"
                size="sm"
                isDisabled={connectBusy}
                label={t('group.credentials.cancel')}
                onClick={() => setConnectionWorkspace(false)}
              />
              <Button
                size="sm"
                isLoading={connectBusy || connectionInspectionPending}
                isDisabled={readyConnectionStages.length === 0 || connectionInspectionPending}
                label={
                  readyConnectionStages.length > 1
                    ? t('group.credentials.subscription.confirmConnectCount', {
                        count: intl.formatNumber(readyConnectionStages.length),
                      })
                    : t('group.credentials.subscription.confirmConnect')
                }
                onClick={() => void saveConnectedAccounts()}
              />
            </>
          }
        >
          <div {...stylex.props(styles.connect)}>
            {connectFeedback && (
              <InlineNotice tone="danger" appearance="ledger">
                {connectFeedback}
              </InlineNotice>
            )}
            <SubscriptionCredentialStager
              stages={connectionStages}
              channelId={channelId}
              channelName={channelName}
              groupId={groupId}
              authorizationMethods={authorizationMethods}
              notices={channelNotices}
              compact
              hideHeader
              context="connect"
              disabled={connectBusy}
              onStagesChange={setConnectionStages}
            />
          </div>
        </DetailPanel>
      )}

      {collectionRefreshing && (
        <span
          {...stylex.props(styles.refreshing)}
          role="status"
          aria-label={t('group.credentials.loading')}
        >
          <RefreshCw size={13} aria-hidden="true" />
        </span>
      )}

      {credentialsQuery.isPending || initialLoading ? (
        <div role="status" aria-label={t('group.credentials.loading')}>
          {Array.from({ length: Math.min(filters.page_size, 8) }, (_, index) => (
            <Skeleton key={index} height={52} radius={2} />
          ))}
        </div>
      ) : credentialsQuery.isError && !collection ? (
        <EmptyState
          title={t('group.credentials.loadFailed')}
          icon={<Search size={20} />}
          actions={
            <Button
              variant="secondary"
              size="sm"
              label={t('common.retry')}
              onClick={() => void credentialsQuery.refetch()}
            />
          }
        />
      ) : collection ? (
        <>
          {credentialsQuery.isError && (
            <Banner
              status="warning"
              title={t('group.credentials.stale')}
              endContent={
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('common.retry')}
                  onClick={() => void credentialsQuery.refetch()}
                />
              }
            />
          )}
          {feedback && (
            <p {...stylex.props(styles.feedback)} role="alert">
              {feedback}
            </p>
          )}
          {collection.summary.total > 0 && (
            <div
              {...stylex.props(styles.summaryStrip)}
              role="group"
              aria-label={t('group.credentials.summary.region')}
            >
              <span {...stylex.props(styles.summaryLabel)}>
                {t('group.credentials.summary.current')}
              </span>
              <Selector
                size="sm"
                variant="input"
                label={t('group.credentials.summary.region')}
                options={statusSummaryItems.map((item) => ({
                  value: item.value ?? 'all',
                  label: `${item.label} (${intl.formatNumber(item.count)})`,
                }))}
                value={filters.status ?? 'all'}
                onChange={(value) =>
                  setStatus(value === 'all' ? undefined : (value as CredentialStatus))
                }
              />
            </div>
          )}
          {collection.summary.total > 0 && (
            <div {...stylex.props(styles.tools)}>
              <div {...stylex.props(styles.searchControls)}>
                <TextInput
                  label={t('group.credentials.filters.search')}
                  isLabelHidden
                  placeholder={
                    connectionType === 'subscription'
                      ? t('group.credentials.subscription.searchPlaceholder')
                      : t('group.credentials.filters.placeholder')
                  }
                  aria-label={t('group.credentials.filters.clear')}
                  value={searchDraft}
                  hasClear
                  size="sm"
                  startIcon={<Search size={14} aria-hidden="true" />}
                  onChange={(value) => {
                    setSearchDraft(value)
                    scheduleSearch()
                  }}
                />
                <Button
                  variant="ghost"
                  size="sm"
                  xstyle={!hasChangedConditions ? styles.resetPlaceholder : undefined}
                  aria-hidden={!hasChangedConditions ? true : undefined}
                  tabIndex={hasChangedConditions ? undefined : -1}
                  isDisabled={!hasChangedConditions}
                  label={t('group.credentials.filters.reset')}
                  onClick={resetFilters}
                />
              </div>
              <GroupCredentialBatchBar
                selectedCount={selectedCount}
                allVisibleSelected={allVisibleSelected}
                pending={batchBusy || singleBusy}
                canSelectAll={collection.items.length > 0}
                canSync={
                  connectionType === 'subscription' &&
                  channelCapabilities.quota_observation &&
                  selectedSubscriptionCredentialsReady
                }
                canDownload={connectionType === 'subscription'}
                onToggleSelect={() => setAllVisible(!allVisibleSelected)}
                onSync={() => void syncSelectedObservations()}
                onDownload={() => void downloadSelectedCredentials()}
                onRemove={() => setDeleteTarget({ ids: [...selectedIds] })}
              />
            </div>
          )}
          {collectionTransition ? (
            <div role="status" aria-label={t('group.credentials.loading')}>
              {Array.from({ length: Math.min(skeletonRows, 8) }, (_, index) => (
                <Skeleton key={index} height={52} radius={2} />
              ))}
            </div>
          ) : collection.summary.total === 0 ? (
            <EmptyState
              title={
                connectionType === 'subscription'
                  ? t('group.credentials.subscription.emptyTitle')
                  : t('group.credentials.emptyTitle')
              }
              description={
                connectionType === 'subscription'
                  ? t('group.credentials.subscription.emptyDescription')
                  : t('group.credentials.emptyDescription')
              }
              icon={<KeyRound size={20} />}
              actions={
                connectionType === 'subscription' && authorizationMethods.length > 0 ? (
                  <Button
                    size="sm"
                    isDisabled={bulkActionsBusy}
                    icon={<Plus size={15} aria-hidden="true" />}
                    label={t('group.credentials.subscription.connect')}
                    onClick={openConnectionWorkspace}
                  />
                ) : (
                  <Button
                    size="sm"
                    icon={<Plus size={15} aria-hidden="true" />}
                    label={t('group.credentials.add')}
                    onClick={() => void navigate({ to: importHref })}
                  />
                )
              }
            />
          ) : collection.pagination.total_items === 0 ? (
            <EmptyState
              title={t('group.credentials.emptyFilterTitle')}
              description={t('group.credentials.emptyFilterDescription')}
              icon={<Search size={20} />}
              actions={
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('group.credentials.filters.reset')}
                  onClick={resetFilters}
                />
              }
            />
          ) : connectionType === 'subscription' ? (
            <div {...stylex.props(styles.accounts)}>
              {collection.items.map((item) => (
                <SubscriptionAccountCard
                  key={item.credential_id}
                  item={credentialWithDetail(item)}
                  selected={selectedIds.has(item.credential_id)}
                  busy={rowBusy(item.credential_id)}
                  refreshingObservation={observationRefreshing(item.credential_id)}
                  detailBusy={detailBusy(item.credential_id)}
                  detailLoaded={detailLoaded(item.credential_id)}
                  detailError={detailError(item.credential_id)}
                  observationError={observationError(item.credential_id)}
                  channelIcon={channelDescriptor?.icon}
                  channelMark={channelDescriptor?.mark}
                  capabilities={channelCapabilities}
                  onSelectedChange={(selected) => setSelected(item.credential_id, selected)}
                  onRestore={(target) => void mutateItem(target, 'restore')}
                  onRefresh={(target) => void refreshObservation(target)}
                  onLoadDetails={(target) => void loadCredentialUsage(target)}
                  onReset={openResetCreditDialog}
                  onDownload={(target) => void downloadCredentialFile(target)}
                  onRefreshCredential={(target) => void refreshCredentialToken(target)}
                  onRemove={(target) =>
                    setDeleteTarget({
                      ids: [target.credential_id],
                      mask: target.account?.email ?? target.mask,
                    })
                  }
                />
              ))}
            </div>
          ) : (
            <LedgerRecordList
              label={t('group.credentials.caption')}
              rowCount={collection.pagination.total_items + 1}
              grid="48px minmax(200px, 1.5fr) 116px minmax(170px, 1.1fr) 80px"
              cardGrid="minmax(0, 0.8fr) minmax(0, 1.2fr)"
              recordMinHeight="52px"
              recordPadding="8px 0"
              columnGap="12px"
              header={
                <>
                  <span
                    role="columnheader"
                    aria-hidden="true"
                    {...stylex.props(ledgerRecordStyles.cell)}
                  />
                  <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
                    {t('group.credentials.columns.credential')}
                  </span>
                  <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
                    {t('group.credentials.columns.status')}
                  </span>
                  <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
                    {t('group.credentials.columns.recent')}
                  </span>
                  <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
                    {t('group.credentials.columns.actions')}
                  </span>
                </>
              }
            >
              {collection.items.map((item, index) => (
                <GroupCredentialRecord
                  key={item.credential_id}
                  item={item}
                  groupId={groupId}
                  rowIndex={
                    (collection.pagination.page - 1) * collection.pagination.page_size + index + 2
                  }
                  selected={selectedIds.has(item.credential_id)}
                  busy={rowBusy(item.credential_id)}
                  expanded={credentialExpanded(item.credential_id)}
                  onSelectedChange={(selected) => setSelected(item.credential_id, selected)}
                  onExpandedChange={(expanded) => setExpanded(item.credential_id, expanded)}
                  onTest={(target) => void openCredentialTest(target)}
                  onRestore={(target) => void mutateItem(target, 'restore')}
                  onRemove={(target) =>
                    setDeleteTarget({ ids: [target.credential_id], mask: target.mask })
                  }
                />
              ))}
            </LedgerRecordList>
          )}
          {collection.summary.total > 0 && (
            <div {...stylex.props(styles.pagination)}>
              <Pagination
                page={collection.pagination.page}
                totalItems={collection.pagination.total_items}
                totalPages={collection.pagination.total_pages}
                pageSize={collection.pagination.page_size}
                pageSizeOptions={[20, 50, 100]}
                onPageSizeChange={setPageSize}
                onChange={setPage}
                isDisabled={credentialsQuery.isFetching || observationBatchBusy}
                size="sm"
              />
            </div>
          )}
        </>
      ) : null}

      <CredentialTestDialog
        open={credentialTestTarget !== undefined}
        mask={credentialTestTarget?.mask ?? ''}
        pending={credentialTestPending}
        requestFailed={credentialTestRequestFailed}
        result={credentialTestResult}
        onOpenChange={setCredentialTestOpen}
        onViewLog={viewCredentialTestLog}
      />

      {/* Reset-credit confirm — classic AppConfirmDialog. */}
      <AlertDialog
        isOpen={resetTarget !== undefined}
        onOpenChange={(value) => {
          if (!value && !resetDialogBusy) setResetTarget(undefined)
        }}
        title={t('group.credentials.subscription.consumeResetCreditTitle')}
        description={t('group.credentials.subscription.consumeResetCreditDescription')}
        cancelLabel={t('group.credentials.cancel')}
        actionLabel={t('group.credentials.subscription.consumeResetCredit')}
        isActionLoading={resetDialogBusy}
        onAction={() => void confirmResetCredit()}
      />

      {/* Delete confirm (single + batch) — classic AppConfirmDialog danger. */}
      <AlertDialog
        isOpen={deleteTarget !== undefined}
        onOpenChange={(value) => {
          if (!value && !dialogBusy) setDeleteTarget(undefined)
        }}
        title={
          deleteTarget?.ids.length === 1
            ? t('group.credentials.deleteTitle')
            : t('group.credentials.batch.deleteTitle')
        }
        description={
          deleteTarget?.ids.length === 1
            ? t('group.credentials.deleteDescription', { mask: deleteTarget.mask })
            : t('group.credentials.batch.deleteDescription', {
                count: intl.formatNumber(deleteTarget?.ids.length ?? 0),
              })
        }
        cancelLabel={t('group.credentials.cancel')}
        actionLabel={t('group.credentials.confirmDelete')}
        actionVariant="destructive"
        isActionLoading={dialogBusy}
        onAction={() => void confirmDelete()}
      />
    </section>
  )
}
