import { AlertDialog, Button, EmptyState, Skeleton, Tooltip } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { Plus, RefreshCw, TriangleAlert } from 'lucide-react'
import { useEffect, useImperativeHandle, useRef, useState, type Ref } from 'react'

import { ApiError, RequestCancelledError } from '@shared/http/errors'
import { channelsQueryOptions } from '@shared/control/resources/channels'
import {
  cacheGroupModels,
  discoverGroupModels,
  groupModelsQueryOptions,
  invalidateGroupModelDependents,
  invalidateGroupSettingsDependents,
  replaceGroupModelsResource,
  updateGroupSettings,
  type GroupModelsDto,
} from '@shared/control/resources/groups'
import type { ModelCandidate } from '@shared/control/resources/providers'
import type { ModelProbeTargetDto } from '@shared/control/resources/model-probe'
import type {
  ModelAliasEditorLabels,
  ModelDiscoveryDrawerLabels,
} from '@shared/domain/models/model-draft'
import {
  appendSelectedCandidates,
  clientModel,
  indexesWithInvalidPriorities,
  indexesWithInvalidWeights,
  indexesWithZeroShare,
  mergeCandidateMetadata,
  readModelNameConflicts,
} from '@shared/domain/models/model-draft'
import {
  createModelSyncDiff,
  createModelDraft,
  findModelNameConflicts,
  normalizedModels,
  sameModels,
  syncedModels,
  type ModelDraftItem,
  type ModelNameConflict,
  type ModelSyncMode,
} from '@shared/domain/groups/models/model-diff'
import {
  parseGroupModelsRouteQuery,
  serializeGroupModelsRouteQuery,
  type GroupModelsRouteState,
} from '@shared/routing/group-detail-route'
import {
  constrainCollectionSearch,
  type SharedRouteQuery,
} from '@shared/routing/route-query'
import { pagePath } from '@shared/routing/page-routes'

import { useStableLoading } from '../../../app/collection-loading'
import { useT } from '../../../app/i18n'
import { RouteLink } from '../../../app/route-link'
import { stringifySharedRouteSearch } from '../../../app/search-codec'
import { useAppServices } from '../../../app/services'
import { useTransientFlag } from '../../../app/use-transient-flag'
import { useUnsavedChanges } from '../../../app/use-unsaved-changes'
import { SectionHeader } from '../../../components/SectionHeader'
import { StickySaveBar } from '../../../components/StickySaveBar'
import { ModelDiscoveryDrawer } from '../../models/ModelDiscoveryDrawer'
import { ModelPricingStatus } from '../../models/ModelPricingStatus'
import { ModelProbeDialog } from '../../models/ModelProbeDialog'
import { useModelProbe } from '../../models/use-model-probe'
import type { GroupEditorState, GroupModelsEditorHandle } from '../editor-handles'
import {
  ModelAliasEditor,
  type ModelAliasEditorHandle,
} from '../ModelAliasEditor'
import { GroupModelSyncDialog } from './GroupModelSyncDialog'

function groupDetailHref(groupId: number): string {
  return `${pagePath('groups')}/${groupId}`
}

/**
 * Classic features/groups/models/GroupModelsTab.vue — model list editor with
 * discovery drawer, probe dialog, sync dialog and unified-mode save reporting.
 */
export function GroupModelsTab({
  groupId,
  channelId,
  enabled = true,
  unified = false,
  blocked = false,
  readonlyRouteFields = false,
  onStateChange,
  ref,
}: {
  groupId: number
  channelId: string
  enabled?: boolean
  unified?: boolean
  blocked?: boolean
  readonlyRouteFields?: boolean
  onStateChange(state: GroupEditorState): void
  ref?: Ref<GroupModelsEditorHandle>
}) {
  const t = useT()
  const { apiClient, toast } = useAppServices()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const rawSearch = useRouterState({
    select: (state) => state.location.search as SharedRouteQuery,
  })
  const routeState = parseGroupModelsRouteQuery(rawSearch)
  const query = useQuery(groupModelsQueryOptions(apiClient, groupId))
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const supportsModelDiscovery =
    channelsQuery.data?.items.find(({ channel_id }) => channel_id === channelId)?.capabilities
      .model_discovery === true
  const initialLoading = useStableLoading(query.isPending && query.data === undefined)
  const queryRefreshing = query.data !== undefined && query.isFetching

  const [saved, setSaved] = useState<ModelDraftItem[]>([])
  const [draft, setDraft] = useState<ModelDraftItem[]>([])
  const [pending, setPending] = useState<'discover' | 'save' | 'sync' | null>(null)
  const [discoveryError, setDiscoveryError] = useState('')
  const [discoveryReady, setDiscoveryReady] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [serverConflicts, setServerConflicts] = useState<ModelNameConflict[]>([])
  const drawerOpen = routeState.discoveryOpen
  const savedModelIDByKey = new Map<number, string>()
  for (const item of saved) savedModelIDByKey.set(item.key, item.id.trim())

  const probe = useModelProbe()

  // In-flight guard for the probe's "apply enabled" action. Drives the dialog's
  // controlled `applying` prop and must be reset before returning on every path.
  const [probeApplying, setProbeApplying] = useState(false)
  // The enabled state for the only group this tab controls. `enabled` is the
  // single source of truth for the enabled initial value (default true).
  const groupEnabledById = new Map<number, boolean>([[groupId, enabled]])

  // Only a saved, unrenamed row identifies a model that is compiled into the route
  // targets; probing a draft would return a meaningless target_unavailable.
  function probeRowTarget(item: ModelDraftItem): ModelProbeTargetDto | null {
    const id = item.id.trim()
    if (id === '' || savedModelIDByKey.get(item.key) !== id) return null
    return { group_id: groupId, model: id }
  }

  const [pendingProbe, setPendingProbe] = useState<ModelProbeTargetDto | null>(null)

  // A disabled group still serves a probe on explicit request, but it asks first so
  // the upstream call is never a surprise.
  function probeRow(item: ModelDraftItem): void {
    const target = probeRowTarget(item)
    if (target === null) return
    if (enabled) {
      void probe.start([target])
      return
    }
    setPendingProbe(target)
  }

  function confirmProbe(): void {
    const target = pendingProbe
    setPendingProbe(null)
    if (target !== null)
      void probe.start([target], { disabledGroupIds: [target.group_id] })
  }

  function viewProbeLog(logId: string): void {
    void navigate({
      to: pagePath('logs'),
      search: { selected_request_id: logId },
    })
  }

  // 调度跳转：带上对外模型/别名、来源分组与 entry 定位，让调度页精确高亮对应行；
  // entry 标识缺失时仅带来源分组，由调度页在详情加载后按模型/别名解析首行。
  function scheduleHrefFor(item: ModelDraftItem): string {
    const search: Record<string, string> = {
      schedule_model: clientModel(item) || item.name || item.id,
      schedule_group: String(groupId),
    }
    const entryID = item.entry_id?.trim()
    if (entryID) search.schedule_row = `${groupId}:${entryID}`
    return `${pagePath('schedule')}${stringifySharedRouteSearch(search)}`
  }

  // Persist the probe's proposed enabled changes per group and invalidate the
  // affected group representations. Keeps the dialog open on failure so the user
  // can retry; never reports success unless every write settled.
  async function onApplyProbeEnabled(changes: Map<number, boolean>): Promise<void> {
    setProbeApplying(true)
    const results = await Promise.allSettled(
      [...changes.entries()].map(async ([targetGroupId, next]) => {
        await updateGroupSettings(apiClient, targetGroupId, { enabled: next })
        await invalidateGroupSettingsDependents(queryClient, targetGroupId)
      }),
    )
    const ok = results.every((result) => result.status === 'fulfilled')
    setProbeApplying(false)
    if (ok) {
      toast.show({ message: t('monitor.modelProbe.toggle.applied', { count: changes.size }), tone: 'success' })
      probe.close()
    } else {
      toast.show({ message: t('monitor.modelProbe.toggle.applyFailed'), tone: 'danger' })
    }
  }

  const modelEditorRef = useRef<ModelAliasEditorHandle>(null)
  const [emptyConfirmOpen, setEmptyConfirmOpen] = useState(false)
  const [candidates, setCandidates] = useState<ModelCandidate[]>([])
  const [syncDialogOpen, setSyncDialogOpen] = useState(false)
  const [syncMode, setSyncMode] = useState<ModelSyncMode>('full')
  const [syncError, setSyncError] = useState('')
  const { value: savedFeedback, clear: clearSavedFeedback, show: showSavedFeedback } =
    useTransientFlag(1_600)
  const nextKeyRef = useRef(1)
  const controllerRef = useRef<AbortController | undefined>(undefined)

  const conflicts = serverConflicts.length
    ? serverConflicts
    : findModelNameConflicts(normalizedModels(draft))
  const emptyAliasIndexes = new Set(
    draft.flatMap((item, index) => (item.alias_enabled && !item.alias.trim() ? [index] : [])),
  )
  const emptyIDIndexes = new Set(
    draft.flatMap((item, index) => (!item.id.trim() ? [index] : [])),
  )
  const invalidWeightIndexes = indexesWithInvalidWeights(draft)
  const invalidPriorityIndexes = indexesWithInvalidPriorities(draft)
  const zeroShareIndexes = indexesWithZeroShare(draft)
  const invalidRowCount = new Set([
    ...conflicts.flatMap((item) => item.indexes),
    ...emptyAliasIndexes,
    ...emptyIDIndexes,
    ...invalidWeightIndexes,
    ...invalidPriorityIndexes,
    ...zeroShareIndexes,
  ]).size
  const validationSummary = [
    conflicts.length ? t('group.modelEditor.conflictSummary') : '',
    emptyIDIndexes.size ? t('group.modelEditor.manualIdRequired') : '',
    emptyAliasIndexes.size ? t('group.modelEditor.emptyAliasSummary') : '',
    zeroShareIndexes.size ? t('group.modelEditor.zeroShareSummary') : '',
    invalidWeightIndexes.size ? t('group.modelEditor.invalidWeight') : '',
    invalidPriorityIndexes.size ? t('group.modelEditor.invalidPriority') : '',
  ]
    .filter(Boolean)
    .join(' · ')
  const saveBarError = validationSummary || saveError
  const dirty =
    !sameModels(saved, draft) || draft.some((item) => item.editable_id && !item.id.trim())
  const savePending = pending === 'save' || pending === 'sync'
  const operationBlocked = pending !== null || blocked

  const empty = normalizedModels(draft).length === 0
  const canSave =
    dirty &&
    !operationBlocked &&
    conflicts.length === 0 &&
    emptyIDIndexes.size === 0 &&
    emptyAliasIndexes.size === 0 &&
    invalidWeightIndexes.size === 0 &&
    invalidPriorityIndexes.size === 0 &&
    zeroShareIndexes.size === 0
  const pendingPricingCount = draft.filter(
    (item) => item.pricing_status === 'pending',
  ).length
  const currentModelIDs = draft.map((item) => item.id.trim()).filter(Boolean)
  const knownPricingStatusByID = new Map(
    saved.map((item) => [item.id, item.pricing_status] as const),
  )
  const syncDiff = createModelSyncDiff(saved, candidates)
  const syncRequestModels = syncedModels(saved, syncDiff, syncMode)
  const syncConflicts = findModelNameConflicts(syncRequestModels)
  const syncChangeCount =
    (syncMode === 'cleanup' ? 0 : syncDiff.additions.length) +
    (syncMode === 'add' ? 0 : syncDiff.removals.length)
  const syncDisabled = !discoveryReady || dirty || operationBlocked
  const syncDisabledReason = dirty ? t('group.modelEditor.sync.dirty') : undefined

  const aliasEditorLabels: ModelAliasEditorLabels = {
    tableLabel: t('group.modelEditor.tableLabel'),
    id: t('group.modelEditor.id'),
    alias: t('group.modelEditor.alias'),
    thirdColumn: t('group.modelEditor.testAliasAndPricing'),
    actions: t('group.modelEditor.actions'),
    search: t('group.modelEditor.searchPlaceholder'),
    searchLabel: t('group.modelEditor.searchLabel'),
    clearSearch: t('group.modelEditor.searchClear'),
    aliasEnabledFor: (id) => t('group.modelEditor.aliasEnabledFor', { id }),
    aliasFor: (id) => t('group.modelEditor.aliasFor', { id }),
    aliasPlaceholder: t('group.modelEditor.aliasPlaceholder'),
    aliasRequired: t('group.modelEditor.aliasRequired'),
    removeFor: (id) => t('group.modelEditor.removeFor', { id }),
    manualId: t('group.modelEditor.manualId'),
    manualIdRequired: t('group.modelEditor.manualIdRequired'),
    add: t('group.modelEditor.add'),
    addInline: t('group.modelEditor.addInline'),
    count: (count) =>
      pendingPricingCount
        ? t('group.modelEditor.summary', { count, pending: pendingPricingCount })
        : t('group.modelEditor.total', { count }),
    empty: t('group.modelEditor.empty'),
    noMatches: t('group.modelEditor.noMatches'),
    nameConflict: (name) => t('group.modelEditor.nameConflict', { name }),
    weight: t('group.modelEditor.weight'),
    priority: t('group.modelEditor.priority'),
    weightDisabled: t('group.modelEditor.weightDisabled'),
    invalidWeight: t('group.modelEditor.invalidWeight'),
    invalidPriority: t('group.modelEditor.invalidPriority'),
    zeroShare: t('group.modelEditor.zeroShareSummary'),
  }
  const discoveryDrawerLabels: ModelDiscoveryDrawerLabels = {
    title: t('group.modelEditor.drawer.title'),
    description: t('group.modelEditor.drawer.description'),
    close: t('group.modelEditor.drawer.close'),
    loading: t('group.modelEditor.drawer.loading'),
    search: t('group.modelEditor.drawer.search'),
    clearSearch: t('group.modelEditor.clearSearch'),
    filterLabel: t('group.modelEditor.drawer.filterLabel'),
    filterUnadded: t('group.modelEditor.drawer.filterAvailable'),
    filterAll: t('group.modelEditor.drawer.filterAll'),
    alreadyAdded: t('group.modelEditor.drawer.alreadyAdded'),
    unadded: t('group.modelEditor.drawer.filterAvailable'),
    noMatches: t('group.modelEditor.drawer.noMatches'),
    empty: t('group.modelEditor.drawer.empty'),
    selected: (count) => t('group.modelEditor.drawer.selected', { count }),
    selectAll: t('group.modelEditor.drawer.selectAll'),
    deselectAll: t('group.modelEditor.drawer.deselectAll'),
    retry: t('common.retry'),
    cancel: t('common.cancel'),
    confirm: t('group.modelEditor.drawer.confirm'),
    pricingStatus: {
      pending: t('group.modelEditor.pricingStatus.pending'),
      configured: t('group.modelEditor.pricingStatus.configured'),
    },
    pricingDiscovered: (source) =>
      t('group.modelEditor.pricingStatus.discovered', { source }),
    sources: {
      catalog: t('group.modelEditor.sources.catalog'),
      live: t('group.modelEditor.sources.live'),
    },
  }

  const { dialog: unsavedChangesDialog } = useUnsavedChanges({
    dirty,
    blocked: operationBlocked,
    allowRouteUpdate: (current, next) => {
      const sameGroup =
        current.routeId === next.routeId &&
        String(current.params.id) === String(next.params.id)
      if (!sameGroup) return false
      const nextTab = typeof next.search.tab === 'string' ? next.search.tab : undefined
      const currentTab =
        typeof current.search.tab === 'string' ? current.search.tab : undefined
      if (unified) return nextTab !== 'credentials'
      return nextTab === 'models' && currentTab === 'models'
    },
  })

  // Latest-state refs so the effects below read current values without
  // re-subscribing on every draft keystroke (classic watches fire only on the
  // watched source, while guards read live state). Written in the first
  // passive effect of every commit — react-hooks/refs forbids render writes.
  const stateRef = useRef({
    dirty,
    operationBlocked,
    blocked,
    saved,
    draft,
    pending,
  })

  function hydrateModels(models: GroupModelsDto | undefined): void {
    const { dirty: isDirty, operationBlocked: isBlocked } = stateRef.current
    if (!models || isDirty || isBlocked) return
    const next = createModelDraft(models.items).map((item) => ({
      ...item,
      key: nextKeyRef.current++,
    }))
    setSaved(next)
    setDraft(next.map((item) => ({ ...item, sources: [...item.sources] })))
    setServerConflicts([])
    setSaveError('')
  }
  const hydrateModelsRef = useRef(hydrateModels)
  const runDiscoveryRef = useRef<() => Promise<void>>(async () => {})
  const onStateChangeRef = useRef(onStateChange)

  // Sync pass — must run before every effect that reads the refs above.
  useEffect(() => {
    stateRef.current = { dirty, operationBlocked, blocked, saved, draft, pending }
    hydrateModelsRef.current = hydrateModels
    runDiscoveryRef.current = runDiscovery
    onStateChangeRef.current = onStateChange
  })

  useEffect(() => {
    hydrateModelsRef.current(query.data)
  }, [query.data])

  // Classic watch(props.blocked): unblocking re-runs hydration so edits that
  // arrived while blocked are picked up.
  const wasBlockedRef = useRef(blocked)
  useEffect(() => {
    if (wasBlockedRef.current && !blocked) hydrateModelsRef.current(query.data)
    wasBlockedRef.current = blocked
  }, [blocked, query.data])

  // Classic watch(dirty): a fresh edit clears the saved flash.
  useEffect(() => {
    if (dirty) clearSavedFeedback()
  }, [dirty, clearSavedFeedback])

  // Classic watch(props.groupId): switching groups drops discovery state.
  const lastGroupIdRef = useRef(groupId)
  useEffect(() => {
    if (lastGroupIdRef.current === groupId) return
    lastGroupIdRef.current = groupId
    setCandidates([])
    setDiscoveryReady(false)
    setSyncDialogOpen(false)
    setSyncError('')
  }, [groupId])

  async function runDiscovery(): Promise<void> {
    if (!supportsModelDiscovery || stateRef.current.operationBlocked) return
    controllerRef.current?.abort()
    setDiscoveryReady(false)
    const active = new AbortController()
    controllerRef.current = active
    setPending('discover')
    try {
      const result = await discoverGroupModels(apiClient, groupId, active.signal)
      if (controllerRef.current !== active || stateRef.current.blocked) return
      setCandidates(result.models)
      setDraft((current) => mergeCandidateMetadata(current, result.models))
      setDiscoveryReady(true)
    } catch (cause: unknown) {
      if (cause instanceof RequestCancelledError || controllerRef.current !== active) return
      setDiscoveryReady(false)
      setDiscoveryError(
        cause instanceof ApiError && cause.code === 'NO_ACTIVE_CREDENTIAL'
          ? t('group.modelEditor.noActiveCredential.title')
          : t('common.modelDiscoveryFailed'),
      )
    } finally {
      if (controllerRef.current === active) {
        controllerRef.current = undefined
        setPending(null)
      }
    }
  }
  // Classic watch([discoveryOpen, models, supported]): opening the drawer
  // triggers discovery once; closing aborts an in-flight discover.
  useEffect(() => {
    if (!routeState.discoveryOpen) {
      if (stateRef.current.pending === 'discover') controllerRef.current?.abort()
      return
    }
    if (
      supportsModelDiscovery &&
      query.data &&
      !stateRef.current.operationBlocked &&
      !discoveryReady
    ) {
      void runDiscoveryRef.current()
    }
  }, [routeState.discoveryOpen, query.data, supportsModelDiscovery, discoveryReady])

  function navigateRoute(state: GroupModelsRouteState, replace = false): void {
    void navigate({
      to: groupDetailHref(groupId),
      search: serializeGroupModelsRouteQuery(state),
      replace,
    })
  }

  function updateRoute(patch: Partial<GroupModelsRouteState>, replace = false): void {
    navigateRoute({ ...routeState, ...patch }, replace)
  }

  function setDiscoverySearch(value: string): void {
    if (stateRef.current.operationBlocked) return
    updateRoute({ discoverySearch: constrainCollectionSearch(value) }, true)
  }

  function setDiscoveryFilter(value: 'unadded' | 'all'): void {
    if (stateRef.current.operationBlocked) return
    updateRoute({ discoveryFilter: value })
  }

  function setDiscoveryOpen(open: boolean): void {
    if (stateRef.current.operationBlocked) return
    updateRoute(
      open
        ? { discoveryOpen: true }
        : {
            discoveryOpen: false,
            discoverySearch: undefined,
            discoveryFilter: 'unadded',
          },
    )
  }

  function updateModels(models: ModelDraftItem[]): void {
    if (stateRef.current.operationBlocked) return
    setServerConflicts([])
    setSaveError('')
    setDraft((current) => {
      const previousByKey = new Map(current.map((item) => [item.key, item] as const))
      return models.map((item) => {
        const previous = previousByKey.get(item.key)
        return {
          ...item,
          pricing_status:
            previous && previous.id === item.id
              ? item.pricing_status
              : pricingStatusForID(item.id),
          name: previous && previous.id === item.id ? item.name : item.id,
          sources: previous && previous.id === item.id ? [...item.sources] : [],
        }
      })
    })
  }

  function pricingStatusForID(id: string): ModelDraftItem['pricing_status'] {
    return knownPricingStatusByID.get(id.trim()) ?? 'pending'
  }

  function createManualRow(): ModelDraftItem {
    return {
      id: '',
      name: '',
      sources: [],
      alias: '',
      alias_enabled: false,
      weight: null,
      priority: null,
      pricing_status: 'pending',
      editable_id: true,
      key: nextKeyRef.current++,
    }
  }

  function addManual(): void {
    void modelEditorRef.current?.addManual()
  }

  function requestDiscovery(): void {
    if (!supportsModelDiscovery || stateRef.current.operationBlocked) return
    setCandidates([])
    setDiscoveryReady(false)
    setDiscoveryError('')
    if (drawerOpen) void runDiscovery()
    else setDiscoveryOpen(true)
  }

  function confirmCandidates(selectedCandidates: ModelCandidate[]): void {
    if (stateRef.current.operationBlocked) return
    setDraft((current) =>
      appendSelectedCandidates(current, selectedCandidates, (candidate) => ({
        id: candidate.id,
        name: candidate.name,
        sources: [...candidate.sources],
        alias: '',
        alias_enabled: false,
        weight: null,
        priority: null,
        pricing_status: candidate.pricing_status,
        key: nextKeyRef.current++,
      })),
    )
    setServerConflicts([])
    setSaveError('')
    setDiscoveryOpen(false)
  }

  function setSyncModeValue(mode: ModelSyncMode): void {
    setSyncMode(mode)
    setSyncError('')
  }

  function setSyncDialogOpenValue(open: boolean): void {
    if (!open && pending === 'sync') return
    setSyncDialogOpen(open)
    if (!open) setSyncError('')
  }

  function requestSync(): void {
    if (syncDisabled) return
    setSyncMode('full')
    setSyncError('')
    setSyncDialogOpen(true)
  }

  function acceptSavedModels(result: GroupModelsDto): void {
    const next = createModelDraft(result.items).map((item) => ({
      ...item,
      key: nextKeyRef.current++,
    }))
    setSaved(next)
    setDraft(next.map((item) => ({ ...item, sources: [...item.sources] })))
    setServerConflicts([])
    cacheGroupModels(queryClient, groupId, result)
  }

  async function confirmSync(): Promise<void> {
    if (stateRef.current.operationBlocked || syncChangeCount === 0 || syncConflicts.length)
      return
    const active = new AbortController()
    const models = syncRequestModels
    controllerRef.current = active
    setPending('sync')
    clearSavedFeedback()
    setSyncError('')
    try {
      const result = await replaceGroupModelsResource(
        apiClient,
        groupId,
        { models },
        active.signal,
      )
      if (controllerRef.current !== active || stateRef.current.blocked) return
      acceptSavedModels(result)
      setDiscoveryReady(false)
      setSyncDialogOpen(false)
      setDiscoveryOpen(false)
      await invalidateGroupModelDependents(queryClient, groupId)
      showSavedFeedback()
    } catch (cause: unknown) {
      if (cause instanceof RequestCancelledError || controllerRef.current !== active) return
      setSyncError(t('group.modelEditor.sync.saveFailed'))
    } finally {
      if (controllerRef.current === active) {
        controllerRef.current = undefined
        setPending(null)
      }
    }
  }

  function requestSave(): void {
    if (stateRef.current.operationBlocked || !canSave) return
    if (empty) {
      setEmptyConfirmOpen(true)
      return
    }
    void save()
  }

  async function save(): Promise<void> {
    if (stateRef.current.operationBlocked || !canSave) return
    const active = new AbortController()
    controllerRef.current = active
    setPending('save')
    clearSavedFeedback()
    setSaveError('')
    let shouldFocusInvalid = false
    try {
      const result = await replaceGroupModelsResource(
        apiClient,
        groupId,
        { models: normalizedModels(draft) },
        active.signal,
      )
      if (controllerRef.current !== active || stateRef.current.blocked) return
      acceptSavedModels(result)
      setEmptyConfirmOpen(false)
      await invalidateGroupModelDependents(queryClient, groupId)
      showSavedFeedback()
    } catch (cause: unknown) {
      if (cause instanceof RequestCancelledError || controllerRef.current !== active) return
      const nextConflicts =
        cause instanceof ApiError && cause.code === 'MODEL_NAME_CONFLICT'
          ? readModelNameConflicts(cause.data)
          : []
      if (nextConflicts.length) {
        setServerConflicts(nextConflicts)
        shouldFocusInvalid = true
      } else {
        setSaveError(t('group.modelEditor.saveFailed'))
      }
    } finally {
      if (controllerRef.current === active) {
        controllerRef.current = undefined
        setPending(null)
      }
    }
    if (shouldFocusInvalid) {
      await modelEditorRef.current?.focusFirstInvalid()
    }
  }

  function discard(): void {
    if (stateRef.current.operationBlocked) return
    clearSavedFeedback()
    setServerConflicts([])
    setSaveError('')
    setDraft(stateRef.current.saved.map((item) => ({ ...item, sources: [...item.sources] })))
  }

  async function focusFirstInvalid(): Promise<void> {
    await modelEditorRef.current?.focusFirstInvalid()
  }

  useImperativeHandle(ref, () => ({ requestSave, discard, focusFirstInvalid }))

  // Classic watch([dirty, operationBlocked, saveBarError, savedFeedback]):
  // report aggregate state to the host's unified save bar.
  useEffect(() => {
    onStateChangeRef.current({
      dirty,
      pending: operationBlocked,
      error: saveBarError,
      saved: savedFeedback,
      invalidRowCount,
    })
  }, [dirty, operationBlocked, saveBarError, savedFeedback, invalidRowCount])

  useEffect(
    () => () => {
      controllerRef.current?.abort()
    },
    [],
  )

  return (
    <section {...stylex.props(styles.root)} aria-labelledby="group-models-heading">
      <SectionHeader
        headingId="group-models-heading"
        title={t('group.modelEditor.title')}
        actions={
          <>
            {supportsModelDiscovery && (
              <Button
                variant="secondary"
                size="sm"
                isLoading={pending === 'discover'}
                isDisabled={!query.data || operationBlocked}
                onClick={requestDiscovery}
                icon={<RefreshCw size={16} aria-hidden />}
                label={t('group.modelEditor.discover')}
              />
            )}
            <Button
              size="sm"
              isDisabled={!query.data || operationBlocked}
              onClick={addManual}
              icon={<Plus size={16} aria-hidden />}
              label={t('group.modelEditor.add')}
            />
          </>
        }
      />

      {queryRefreshing && (
        <div
          role="status"
          aria-label={t('group.modelEditor.loading')}
          {...stylex.props(styles.refreshing)}
        >
          <RefreshCw size={13} aria-hidden />
        </div>
      )}

      {(query.isPending && !query.data) || initialLoading ? (
        <div
          role="status"
          aria-label={t('group.modelEditor.loading')}
          {...stylex.props(styles.skeleton)}
        >
          <Skeleton height={58} radius={2} />
          <Skeleton height={58} radius={2} />
          <Skeleton height={58} radius={2} />
          <Skeleton height={58} radius={2} />
          <Skeleton height={58} radius={2} />
          <Skeleton height={58} radius={2} />
        </div>
      ) : query.isError && !query.data ? (
        <div role="alert">
          <EmptyState
            title={t('group.modelEditor.loadFailed')}
            icon={<TriangleAlert size={20} />}
            actions={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void query.refetch()}
              />
            }
          />
        </div>
      ) : query.data ? (
        <>
          <ModelAliasEditor
            ref={modelEditorRef}
            value={draft}
            conflicts={conflicts}
            labels={aliasEditorLabels}
            createRow={createManualRow}
            disabled={operationBlocked}
            searchable={false}
            readonlyRouteFields={readonlyRouteFields}
            onChange={updateModels}
            renderThirdColumn={(item) => (
              <div {...stylex.props(styles.pricingCell)}>
                <div
                  {...stylex.props(styles.testAlias)}
                  data-testid="group-models__test-alias"
                >
                  <span {...stylex.props(styles.testAliasLabel)}>
                    {t('group.modelEditor.testAlias')}
                  </span>
                  {item.test_alias ? (
                    <code {...stylex.props(styles.testAliasCode)}>{item.test_alias}</code>
                  ) : (
                    <span {...stylex.props(styles.testAliasPending)}>
                      {t('group.modelEditor.testAliasPending')}
                    </span>
                  )}
                </div>
                {item.pricing_status === 'pending' && item.price_id !== undefined ? (
                  <RouteLink
                    to={`${pagePath('models')}${stringifySharedRouteSearch({ selected_price_id: item.price_id })}`}
                    {...stylex.props(styles.pricingLink)}
                  >
                    <ModelPricingStatus
                      status={item.pricing_status}
                      labels={{
                        pending: t('group.modelEditor.pricingStatus.pending'),
                        configured: t('group.modelEditor.pricingStatus.configured'),
                      }}
                    />
                  </RouteLink>
                ) : (
                  <ModelPricingStatus
                    status={item.pricing_status}
                    labels={{
                      pending: t('group.modelEditor.pricingStatus.pending'),
                      configured: t('group.modelEditor.pricingStatus.configured'),
                    }}
                  />
                )}
                {item.circuit_breaker && (
                  <span {...stylex.props(styles.breakerSummary)}>
                    {t('group.modelEditor.breaker')}:{' '}
                    {item.circuit_breaker.blacklist_threshold ?? '—'}
                  </span>
                )}
                <RouteLink
                  to={scheduleHrefFor(item)}
                  {...stylex.props(styles.scheduleLink)}
                  data-testid="group-models__schedule-link"
                >
                  {t('group.modelEditor.schedule')}
                </RouteLink>
                <Tooltip
                  content={
                    probeRowTarget(item) === null ? t('monitor.modelProbe.draftHint') : ''
                  }
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    isDisabled={probeRowTarget(item) === null || probe.pending}
                    onClick={() => probeRow(item)}
                    label={t('monitor.modelProbe.button')}
                    xstyle={styles.probeButton}
                  />
                </Tooltip>
              </div>
            )}
          />
          <ModelProbeDialog
            open={probe.open}
            pending={probe.pending}
            failed={probe.failed}
            stopped={probe.stopped}
            results={probe.results}
            groupEnabledById={groupEnabledById}
            applying={probeApplying}
            disabledGroupIds={enabled ? [] : [groupId]}
            total={probe.total}
            completed={probe.completed}
            onOpenChange={(open) => {
              if (!open) probe.close()
            }}
            onStop={probe.stop}
            onViewLog={viewProbeLog}
            onApplyEnabled={onApplyProbeEnabled}
          />
          <AlertDialog
            isOpen={pendingProbe !== null}
            onOpenChange={(open) => {
              if (!open) setPendingProbe(null)
            }}
            title={t('monitor.modelProbe.disabledConfirm.title')}
            description={t('monitor.modelProbe.disabledConfirm.description')}
            cancelLabel={t('monitor.modelProbe.disabledConfirm.cancel')}
            actionLabel={t('monitor.modelProbe.disabledConfirm.confirm')}
            onAction={confirmProbe}
          />
          {supportsModelDiscovery && (
            <ModelDiscoveryDrawer
              open={drawerOpen}
              candidates={candidates}
              currentIds={currentModelIDs}
              loading={pending === 'discover'}
              error={discoveryError}
              labels={discoveryDrawerLabels}
              dismissible={!operationBlocked}
              blocked={operationBlocked}
              search={routeState.discoverySearch ?? ''}
              filter={routeState.discoveryFilter}
              onOpenChange={setDiscoveryOpen}
              onSearchChange={setDiscoverySearch}
              onFilterChange={setDiscoveryFilter}
              onRetry={requestDiscovery}
              onConfirm={confirmCandidates}
              footerActions={
                <Tooltip content={syncDisabledReason ?? ''}>
                  <Button
                    variant="secondary"
                    size="sm"
                    isDisabled={syncDisabled}
                    onClick={requestSync}
                    label={t('group.modelEditor.sync.action')}
                  />
                </Tooltip>
              }
            />
          )}

          <GroupModelSyncDialog
            open={syncDialogOpen}
            mode={syncMode}
            additions={syncDiff.additions}
            removals={syncDiff.removals}
            conflicts={syncConflicts}
            pending={pending === 'sync'}
            error={syncError}
            onOpenChange={setSyncDialogOpenValue}
            onModeChange={setSyncModeValue}
            onConfirm={confirmSync}
          />

          <AlertDialog
            isOpen={emptyConfirmOpen}
            onOpenChange={setEmptyConfirmOpen}
            title={t('group.modelEditor.emptyConfirm.title')}
            description={t('group.modelEditor.emptyConfirm.description')}
            cancelLabel={t('group.modelEditor.emptyConfirm.cancel')}
            actionLabel={t('group.modelEditor.emptyConfirm.confirm')}
            actionVariant="destructive"
            isActionLoading={pending === 'save'}
            onAction={() => void save()}
          />

          {!unified && (
            <StickySaveBar
              appearance="ledger"
              errorPlacement="floating"
              alwaysVisible
              dirty={dirty}
              pending={savePending}
              status={saveBarError ? 'error' : savedFeedback ? 'saved' : 'idle'}
              error={saveBarError}
              errorActionLabel={
                invalidRowCount ? t('group.modelEditor.locateFirstInvalid') : ''
              }
              onErrorAction={() => void modelEditorRef.current?.focusFirstInvalid()}
              statusContent={
                <div>
                  <strong>
                    {savePending
                      ? t('group.modelEditor.saving')
                      : savedFeedback
                        ? t('group.modelEditor.savedFeedback')
                        : dirty
                          ? t('group.modelEditor.unsaved')
                          : t('group.modelEditor.saved')}
                  </strong>
                  <span>
                    {savePending
                      ? t('group.modelEditor.savingNote')
                      : savedFeedback
                        ? t('group.modelEditor.savedFeedbackNote')
                        : dirty
                          ? invalidRowCount > 0
                            ? t('group.modelEditor.invalidNote', { count: invalidRowCount })
                            : t('group.modelEditor.dirtyNote')
                          : t('group.modelEditor.saveNote')}
                  </span>
                </div>
              }
              actions={
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    isDisabled={savePending || !dirty}
                    onClick={discard}
                    label={t('common.discard')}
                  />
                  <Button
                    size="sm"
                    isDisabled={savePending || !canSave}
                    onClick={requestSave}
                    label={t('group.modelEditor.save')}
                  />
                </>
              }
            />
          )}
        </>
      ) : null}
      {unsavedChangesDialog}
    </section>
  )
}

const styles = stylex.create({
  root: {
    display: 'grid',
    gap: 0,
    minWidth: 0,
    paddingTop: {
      default: 'var(--detail-panel-padding-top)',
      '@media (max-width: 800px)': 'var(--detail-panel-padding-top-compact)',
    },
  },
  refreshing: {
    display: 'flex',
    justifyContent: 'flex-end',
    color: 'var(--color-text-faint)',
    paddingBottom: 'var(--space-2)',
  },
  skeleton: {
    display: 'grid',
    gap: '2px',
  },
  pricingCell: {
    display: 'grid',
    minWidth: 0,
    gap: '5px',
    justifyItems: 'start',
  },
  testAlias: {
    display: 'grid',
    minWidth: 0,
    gap: '2px',
  },
  testAliasLabel: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  testAliasPending: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  testAliasCode: {
    minWidth: 0,
    overflowWrap: 'anywhere',
    fontSize: 'var(--text-meta)',
  },
  pricingLink: {
    textDecoration: 'none',
    color: 'inherit',
  },
  scheduleLink: {
    color: 'var(--color-action)',
    fontSize: 'var(--text-meta)',
    fontWeight: 600,
  },
  breakerSummary: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
  },
  probeButton: {
    paddingLeft: 0,
    paddingRight: 0,
    color: 'var(--color-action)',
    fontSize: 'var(--text-meta)',
    fontWeight: 600,
  },
})
