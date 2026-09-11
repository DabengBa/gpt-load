<script setup lang="ts">
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { Plus, RefreshCw } from '@lucide/vue'
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import { ApiError, RequestCancelledError } from '@/api/errors'
import { useApiClient } from '@/api/client-context'
import { useStableLoading } from '@/app/loading-state'
import { channelsQueryOptions } from '@/app/resources/channels'
import {
  cacheGroupModels,
  discoverGroupModels,
  groupModelsQueryOptions,
  invalidateGroupModelDependents,
  replaceGroupModelsResource,
  type GroupModelsDto,
} from '@/app/resources/groups'
import type { ModelCandidate } from '@/app/resources/providers'
import type { ModelProbeTargetDto } from '@/app/resources/model-probe'
import { useUnsavedChanges } from '@/app/unsaved-changes'
import { useTransientFlag } from '@/app/use-transient-flag'
import { constrainCollectionSearch } from '@/app/route-query'
import { groupDetailLocation, monitorLocation } from '@/app/route-locations'
import AppButton from '@/components/ui/AppButton.vue'
import AppConfirmDialog from '@/components/ui/AppConfirmDialog.vue'
import AsyncRefreshIndicator from '@/components/ui/AsyncRefreshIndicator.vue'
import PanelHeader from '@/components/ui/PanelHeader.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import SkeletonSurface from '@/components/ui/SkeletonSurface.vue'
import StickySaveBar from '@/components/ui/StickySaveBar.vue'
import ModelAliasEditor from '@/features/models/ModelAliasEditor.vue'
import ModelDiscoveryDrawer from '@/features/models/ModelDiscoveryDrawer.vue'
import {
  appendSelectedCandidates,
  indexesWithInvalidPriorities,
  indexesWithInvalidWeights,
  indexesWithZeroShare,
  mergeCandidateMetadata,
  readModelNameConflicts,
  type ModelAliasEditorLabels,
  type ModelDiscoveryDrawerLabels,
} from '@/features/models/model-draft'
import ModelPricingStatus from '@/features/models/ModelPricingStatus.vue'
import ModelProbeDialog from '@/features/models/ModelProbeDialog.vue'
import { useModelProbe } from '@/features/models/use-model-probe'

import GroupModelSyncDialog from './GroupModelSyncDialog.vue'
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
} from './model-diff'
import {
  parseGroupModelsRouteQuery,
  serializeGroupModelsRouteQuery,
  type GroupModelsRouteState,
  normalizeGroupTab,
} from '../group-route'

const props = withDefaults(
  defineProps<{
    groupId: number
    channelId: string
    readonlyRouteFields?: boolean
    unified?: boolean
    blocked?: boolean
  }>(),
  { readonlyRouteFields: false, unified: false, blocked: false },
)
const emit = defineEmits<{
  state: [
    state: {
      dirty: boolean
      pending: boolean
      error: string
      saved: boolean
      invalidRowCount: number
    },
  ]
}>()
const client = useApiClient()
const queryClient = useQueryClient()
const route = useRoute()
const router = useRouter()
const { t } = useI18n()
const routeState = computed(() => parseGroupModelsRouteQuery(route.query))
const query = useQuery(groupModelsQueryOptions(client, () => props.groupId))
const channelsQuery = useQuery(channelsQueryOptions(client, ''))
const supportsModelDiscovery = computed(
  () =>
    channelsQuery.data.value?.items.find(({ channel_id }) => channel_id === props.channelId)
      ?.capabilities.model_discovery === true,
)
const initialLoading = useStableLoading(
  () => query.isPending.value && query.data.value === undefined,
)
const queryRefreshing = computed(() => query.data.value !== undefined && query.isFetching.value)
const saved = ref<ModelDraftItem[]>([])
const draft = ref<ModelDraftItem[]>([])
const pending = ref<'discover' | 'save' | 'sync' | null>(null)
const discoveryError = ref('')
const discoveryReady = ref(false)
const saveError = ref('')
const serverConflicts = ref<ModelNameConflict[]>([])
const drawerOpen = computed(() => routeState.value.discoveryOpen)
const savedModelIDByKey = computed(() => {
  const ids = new Map<number, string>()
  for (const item of saved.value) ids.set(item.key, item.id.trim())
  return ids
})
const {
  open: probeOpen,
  pending: probePending,
  failed: probeFailed,
  stopped: probeStopped,
  results: probeResults,
  total: probeTotal,
  completed: probeCompleted,
  start: startProbe,
  stop: stopProbe,
  close: closeProbe,
} = useModelProbe()

// Only a saved, unrenamed row identifies a model that is compiled into the route
// targets; probing a draft would return a meaningless target_unavailable.
function probeRowTarget(item: ModelDraftItem): ModelProbeTargetDto | null {
  const id = item.id.trim()
  if (id === '' || savedModelIDByKey.value.get(item.key) !== id) return null
  return { group_id: props.groupId, model: id }
}

function probeRow(item: ModelDraftItem): void {
  const target = probeRowTarget(item)
  if (target === null) return
  void startProbe([target])
}

function handleProbeOpen(value: boolean): void {
  if (!value) closeProbe()
}

function viewProbeLog(logId: string): void {
  void router.push(monitorLocation({ tab: 'logs', selected_request_id: logId }))
}
const modelEditor = ref<{
  addManual: () => Promise<void>
  focusFirstInvalid: () => Promise<void>
}>()
const emptyConfirmOpen = ref(false)
const candidates = ref<ModelCandidate[]>([])
const syncDialogOpen = ref(false)
const syncMode = ref<ModelSyncMode>('full')
const syncError = ref('')
const {
  value: savedFeedback,
  clear: clearSavedFeedback,
  show: showSavedFeedback,
} = useTransientFlag(1_600)
let nextKey = 1
let controller: AbortController | undefined

const conflicts = computed(() =>
  serverConflicts.value.length
    ? serverConflicts.value
    : findModelNameConflicts(normalizedModels(draft.value)),
)
const emptyAliasIndexes = computed(
  () =>
    new Set(
      draft.value.flatMap((item, index) =>
        item.alias_enabled && !item.alias.trim() ? [index] : [],
      ),
    ),
)
const emptyIDIndexes = computed(
  () => new Set(draft.value.flatMap((item, index) => (!item.id.trim() ? [index] : []))),
)
const invalidWeightIndexes = computed(() => indexesWithInvalidWeights(draft.value))
const invalidPriorityIndexes = computed(() => indexesWithInvalidPriorities(draft.value))
const zeroShareIndexes = computed(() => indexesWithZeroShare(draft.value))
const invalidRowCount = computed(
  () =>
    new Set([
      ...conflicts.value.flatMap((item) => item.indexes),
      ...emptyAliasIndexes.value,
      ...emptyIDIndexes.value,
      ...invalidWeightIndexes.value,
      ...invalidPriorityIndexes.value,
      ...zeroShareIndexes.value,
    ]).size,
)
const validationSummary = computed(() =>
  [
    conflicts.value.length ? t('group.modelEditor.conflictSummary') : '',
    emptyIDIndexes.value.size ? t('group.modelEditor.manualIdRequired') : '',
    emptyAliasIndexes.value.size ? t('group.modelEditor.emptyAliasSummary') : '',
    zeroShareIndexes.value.size ? t('group.modelEditor.zeroShareSummary') : '',
    invalidWeightIndexes.value.size ? t('group.modelEditor.invalidWeight') : '',
    invalidPriorityIndexes.value.size ? t('group.modelEditor.invalidPriority') : '',
  ]
    .filter(Boolean)
    .join(' · '),
)
const saveBarError = computed(() => validationSummary.value || saveError.value)
const dirty = computed(
  () =>
    !sameModels(saved.value, draft.value) ||
    draft.value.some((item) => item.editable_id && !item.id.trim()),
)
const savePending = computed(() => pending.value === 'save' || pending.value === 'sync')
const operationBlocked = computed(() => pending.value !== null || props.blocked)

watch(
  [dirty, operationBlocked, saveBarError, savedFeedback],
  ([isDirty, isPending, currentError, saved]) => {
    emit('state', {
      dirty: isDirty,
      pending: isPending,
      error: currentError,
      saved,
      invalidRowCount: invalidRowCount.value,
    })
  },
  { immediate: true },
)
const empty = computed(() => normalizedModels(draft.value).length === 0)
const canSave = computed(
  () =>
    dirty.value &&
    !operationBlocked.value &&
    conflicts.value.length === 0 &&
    emptyIDIndexes.value.size === 0 &&
    emptyAliasIndexes.value.size === 0 &&
    invalidWeightIndexes.value.size === 0 &&
    invalidPriorityIndexes.value.size === 0 &&
    zeroShareIndexes.value.size === 0,
)
const pendingPricingCount = computed(
  () => draft.value.filter((item) => item.pricing_status === 'pending').length,
)
const currentModelIDs = computed(() => draft.value.map((item) => item.id.trim()).filter(Boolean))
const knownPricingStatusByID = computed(
  () => new Map(saved.value.map((item) => [item.id, item.pricing_status] as const)),
)
const syncDiff = computed(() => createModelSyncDiff(saved.value, candidates.value))
const syncRequestModels = computed(() => syncedModels(saved.value, syncDiff.value, syncMode.value))
const syncConflicts = computed(() => findModelNameConflicts(syncRequestModels.value))
const syncChangeCount = computed(() => {
  const additions = syncMode.value === 'cleanup' ? 0 : syncDiff.value.additions.length
  const removals = syncMode.value === 'add' ? 0 : syncDiff.value.removals.length
  return additions + removals
})
const syncDisabled = computed(() => !discoveryReady.value || dirty.value || operationBlocked.value)
const syncDisabledReason = computed(() =>
  dirty.value ? t('group.modelEditor.sync.dirty') : undefined,
)
const aliasEditorLabels = computed<ModelAliasEditorLabels>(() => ({
  tableLabel: t('group.modelEditor.tableLabel'),
  id: t('group.modelEditor.id'),
  alias: t('group.modelEditor.alias'),
  thirdColumn: t('group.modelEditor.pricing'),
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
    pendingPricingCount.value
      ? t('group.modelEditor.summary', { count, pending: pendingPricingCount.value })
      : t('group.modelEditor.total', { count }),
  empty: t('group.modelEditor.empty'),
  noMatches: t('group.modelEditor.noMatches'),
  nameConflict: (name) => t('group.modelEditor.nameConflict', { name }),
  weight: t('group.modelEditor.weight'),
  priority: t('group.modelEditor.priority'),
  priorityFallback: t('group.modelEditor.priorityFallback'),
  weightDisabled: t('group.modelEditor.weightDisabled'),
  invalidWeight: t('group.modelEditor.invalidWeight'),
  invalidPriority: t('group.modelEditor.invalidPriority'),
  zeroShare: t('group.modelEditor.zeroShareSummary'),
}))
const discoveryDrawerLabels = computed<ModelDiscoveryDrawerLabels>(() => ({
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
  pricingDiscovered: (source) => t('group.modelEditor.pricingStatus.discovered', { source }),
  sources: {
    catalog: t('group.modelEditor.sources.catalog'),
    live: t('group.modelEditor.sources.live'),
  },
}))
useUnsavedChanges(dirty, {
  blocked: operationBlocked,
  allowRouteUpdate: (to, from) => {
    const sameGroup = to.name === from.name && String(to.params.id) === String(from.params.id)
    if (!sameGroup) return false
    if (props.unified) return to.query.tab !== 'credentials'
    return (
      normalizeGroupTab(to.query.tab) === 'models' && normalizeGroupTab(from.query.tab) === 'models'
    )
  },
})

watch(dirty, (isDirty) => {
  if (isDirty) clearSavedFeedback()
})

watch(
  () => props.groupId,
  () => {
    candidates.value = []
    discoveryReady.value = false
    syncDialogOpen.value = false
    syncError.value = ''
  },
)

function hydrateModels(models: GroupModelsDto | undefined): void {
  if (!models || dirty.value || operationBlocked.value) return
  const next = createModelDraft(models.items).map((item) => ({ ...item, key: nextKey++ }))
  saved.value = next
  draft.value = next.map((item) => ({ ...item, sources: [...item.sources] }))
  serverConflicts.value = []
  saveError.value = ''
}

watch(() => query.data.value, hydrateModels, { immediate: true })

watch(
  () => props.blocked,
  (blocked, wasBlocked) => {
    if (wasBlocked && !blocked) hydrateModels(query.data.value)
  },
)

watch(
  [() => routeState.value.discoveryOpen, () => query.data.value, supportsModelDiscovery],
  ([open, models, supported]) => {
    if (!open) {
      if (pending.value === 'discover') controller?.abort()
      return
    }
    if (supported && models && !operationBlocked.value && !discoveryReady.value) void runDiscovery()
  },
  { immediate: true },
)

function navigateRoute(state: GroupModelsRouteState, replace = false): void {
  const location = groupDetailLocation(props.groupId, serializeGroupModelsRouteQuery(state))
  void (replace ? router.replace(location) : router.push(location))
}

function updateRoute(patch: Partial<GroupModelsRouteState>, replace = false): void {
  navigateRoute({ ...routeState.value, ...patch }, replace)
}

function setDiscoverySearch(value: string): void {
  if (operationBlocked.value) return
  updateRoute({ discoverySearch: constrainCollectionSearch(value) }, true)
}

function setDiscoveryFilter(value: 'unadded' | 'all'): void {
  if (operationBlocked.value) return
  updateRoute({ discoveryFilter: value })
}

function setDiscoveryOpen(open: boolean): void {
  if (operationBlocked.value) return
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
  if (operationBlocked.value) return
  serverConflicts.value = []
  saveError.value = ''
  const previousByKey = new Map(draft.value.map((item) => [item.key, item] as const))
  draft.value = models.map((item) => {
    const previous = previousByKey.get(item.key)
    return {
      ...item,
      pricing_status:
        previous && previous.id === item.id ? item.pricing_status : pricingStatusForID(item.id),
      name: previous && previous.id === item.id ? item.name : item.id,
      sources: previous && previous.id === item.id ? [...item.sources] : [],
    }
  })
}

function pricingStatusForID(id: string): ModelDraftItem['pricing_status'] {
  return knownPricingStatusByID.value.get(id.trim()) ?? 'pending'
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
    key: nextKey++,
  }
}

function addManual(): void {
  void modelEditor.value?.addManual()
}

function requestDiscovery(): void {
  if (!supportsModelDiscovery.value || operationBlocked.value) return
  candidates.value = []
  discoveryReady.value = false
  discoveryError.value = ''
  if (drawerOpen.value) void runDiscovery()
  else setDiscoveryOpen(true)
}

async function runDiscovery(): Promise<void> {
  if (!supportsModelDiscovery.value || operationBlocked.value) return
  controller?.abort()
  discoveryReady.value = false
  const active = new AbortController()
  controller = active
  pending.value = 'discover'
  try {
    const result = await discoverGroupModels(client, props.groupId, active.signal)
    if (controller !== active || props.blocked) return
    candidates.value = result.models
    draft.value = mergeCandidateMetadata(draft.value, result.models)
    discoveryReady.value = true
  } catch (cause: unknown) {
    if (cause instanceof RequestCancelledError || controller !== active) return
    discoveryReady.value = false
    discoveryError.value =
      cause instanceof ApiError && cause.code === 'NO_ACTIVE_CREDENTIAL'
        ? t('group.modelEditor.noActiveCredential.title')
        : t('common.modelDiscoveryFailed')
  } finally {
    if (controller === active) {
      controller = undefined
      pending.value = null
    }
  }
}

function confirmCandidates(selectedCandidates: ModelCandidate[]): void {
  if (operationBlocked.value) return
  draft.value = appendSelectedCandidates(draft.value, selectedCandidates, (candidate) => ({
    id: candidate.id,
    name: candidate.name,
    sources: [...candidate.sources],
    alias: '',
    alias_enabled: false,
    weight: null,
    priority: null,
    pricing_status: candidate.pricing_status,
    key: nextKey++,
  }))
  serverConflicts.value = []
  saveError.value = ''
  setDiscoveryOpen(false)
}

function setSyncMode(mode: ModelSyncMode): void {
  syncMode.value = mode
  syncError.value = ''
}

function setSyncDialogOpen(open: boolean): void {
  if (!open && pending.value === 'sync') return
  syncDialogOpen.value = open
  if (!open) syncError.value = ''
}

function requestSync(): void {
  if (syncDisabled.value) return
  syncMode.value = 'full'
  syncError.value = ''
  syncDialogOpen.value = true
}

function acceptSavedModels(result: GroupModelsDto): void {
  const next = createModelDraft(result.items).map((item) => ({ ...item, key: nextKey++ }))
  saved.value = next
  draft.value = next.map((item) => ({ ...item, sources: [...item.sources] }))
  serverConflicts.value = []
  cacheGroupModels(queryClient, props.groupId, result)
}

async function confirmSync(): Promise<void> {
  if (operationBlocked.value || syncChangeCount.value === 0 || syncConflicts.value.length) return
  const active = new AbortController()
  const models = syncRequestModels.value
  controller = active
  pending.value = 'sync'
  clearSavedFeedback()
  syncError.value = ''
  try {
    const result = await replaceGroupModelsResource(
      client,
      props.groupId,
      { models },
      active.signal,
    )
    if (controller !== active || props.blocked) return
    acceptSavedModels(result)
    discoveryReady.value = false
    syncDialogOpen.value = false
    setDiscoveryOpen(false)
    await invalidateGroupModelDependents(queryClient, props.groupId)
    showSavedFeedback()
  } catch (cause: unknown) {
    if (cause instanceof RequestCancelledError || controller !== active) return
    syncError.value = t('group.modelEditor.sync.saveFailed')
  } finally {
    if (controller === active) {
      controller = undefined
      pending.value = null
    }
  }
}

function requestSave(): void {
  if (operationBlocked.value || !canSave.value) return
  if (empty.value) {
    emptyConfirmOpen.value = true
    return
  }
  void save()
}

async function save(): Promise<void> {
  if (operationBlocked.value || !canSave.value) return
  const active = new AbortController()
  controller = active
  pending.value = 'save'
  clearSavedFeedback()
  saveError.value = ''
  let shouldFocusInvalid = false
  try {
    const result = await replaceGroupModelsResource(
      client,
      props.groupId,
      {
        models: normalizedModels(draft.value),
      },
      active.signal,
    )
    if (controller !== active || props.blocked) return
    acceptSavedModels(result)
    emptyConfirmOpen.value = false
    await invalidateGroupModelDependents(queryClient, props.groupId)
    showSavedFeedback()
  } catch (cause: unknown) {
    if (cause instanceof RequestCancelledError || controller !== active) return
    const nextConflicts =
      cause instanceof ApiError && cause.code === 'MODEL_NAME_CONFLICT'
        ? readModelNameConflicts(cause.data)
        : []
    if (nextConflicts.length) {
      serverConflicts.value = nextConflicts
      shouldFocusInvalid = true
    } else {
      saveError.value = t('group.modelEditor.saveFailed')
    }
  } finally {
    if (controller === active) {
      controller = undefined
      pending.value = null
    }
  }
  if (shouldFocusInvalid) {
    await nextTick()
    await modelEditor.value?.focusFirstInvalid()
  }
}

function discard(): void {
  if (operationBlocked.value) return
  clearSavedFeedback()
  serverConflicts.value = []
  saveError.value = ''
  draft.value = saved.value.map((item) => ({ ...item, sources: [...item.sources] }))
}

async function focusFirstInvalid(): Promise<void> {
  await modelEditor.value?.focusFirstInvalid()
}

defineExpose({ requestSave, discard, focusFirstInvalid })

onBeforeUnmount(() => {
  controller?.abort()
})
</script>

<template>
  <section class="group-models" aria-labelledby="group-models-heading">
    <PanelHeader heading-id="group-models-heading" :title="t('group.modelEditor.title')">
      <template #actions>
        <AppButton
          v-if="supportsModelDiscovery"
          variant="secondary"
          :busy="pending === 'discover'"
          :disabled="!query.data.value || operationBlocked"
          @click="requestDiscovery"
        >
          <RefreshCw :size="16" aria-hidden="true" />{{ t('group.modelEditor.discover') }}
        </AppButton>
        <AppButton :disabled="!query.data.value || operationBlocked" @click="addManual">
          <Plus :size="16" aria-hidden="true" />{{ t('group.modelEditor.add') }}
        </AppButton>
      </template>
    </PanelHeader>

    <AsyncRefreshIndicator :active="queryRefreshing" :label="t('group.modelEditor.loading')" />

    <SkeletonSurface
      v-if="(query.isPending.value && !query.data.value) || initialLoading"
      variant="collection"
      :rows="6"
      :columns="3"
      row-height="58px"
      mobile-row-height="112px"
      show-controls
      :show-pagination="false"
      :concealed="!initialLoading"
      :label="t('group.modelEditor.loading')"
    />
    <QueryFeedback
      v-else-if="query.isError.value && !query.data.value"
      state="error"
      :message="t('group.modelEditor.loadFailed')"
      :retry-label="t('common.retry')"
      @retry="query.refetch()"
    />
    <template v-else-if="query.data.value">
      <ModelAliasEditor
        ref="modelEditor"
        :model-value="draft"
        :conflicts="conflicts"
        :labels="aliasEditorLabels"
        :create-row="createManualRow"
        :disabled="operationBlocked"
        :searchable="false"
        :readonly-route-fields="readonlyRouteFields"
        @update:model-value="updateModels"
      >
        <template #third-column="{ item }">
          <div class="group-models__pricing-cell">
            <ModelPricingStatus
              :status="item.pricing_status"
              :labels="{
                pending: t('group.modelEditor.pricingStatus.pending'),
                configured: t('group.modelEditor.pricingStatus.configured'),
              }"
            />
            <span v-if="item.circuit_breaker" class="group-models__breaker-summary">
              {{ t('group.modelEditor.breaker') }}:
              {{ item.circuit_breaker.blacklist_threshold ?? '—' }}
            </span>
            <RouterLink
              class="group-models__schedule-link"
              :to="monitorLocation({ tab: 'schedule', schedule_model: item.name || item.id })"
            >
              {{ t('group.modelEditor.schedule') }}
            </RouterLink>
            <AppButton
              variant="link"
              size="inline"
              :disabled="probeRowTarget(item) === null || probePending"
              :title="probeRowTarget(item) === null ? t('monitor.modelProbe.draftHint') : undefined"
              @click="probeRow(item)"
            >
              {{ t('monitor.modelProbe.button') }}
            </AppButton>
          </div>
        </template>
      </ModelAliasEditor>
      <ModelProbeDialog
        :open="probeOpen"
        :pending="probePending"
        :failed="probeFailed"
        :stopped="probeStopped"
        :results="probeResults"
        :total="probeTotal"
        :completed="probeCompleted"
        @update:open="handleProbeOpen"
        @stop="stopProbe"
        @view-log="viewProbeLog"
      />
      <ModelDiscoveryDrawer
        v-if="supportsModelDiscovery"
        :open="drawerOpen"
        :candidates="candidates"
        :current-ids="currentModelIDs"
        :loading="pending === 'discover'"
        :error="discoveryError"
        :labels="discoveryDrawerLabels"
        :dismissible="!operationBlocked"
        :blocked="operationBlocked"
        :search="routeState.discoverySearch ?? ''"
        :filter="routeState.discoveryFilter"
        @update:open="setDiscoveryOpen"
        @update:search="setDiscoverySearch"
        @update:filter="setDiscoveryFilter"
        @retry="requestDiscovery"
        @confirm="confirmCandidates"
      >
        <template #footer-actions>
          <AppButton
            variant="secondary"
            size="compact"
            :disabled="syncDisabled"
            :title="syncDisabledReason"
            @click="requestSync"
          >
            {{ t('group.modelEditor.sync.action') }}
          </AppButton>
        </template>
      </ModelDiscoveryDrawer>

      <GroupModelSyncDialog
        :open="syncDialogOpen"
        :mode="syncMode"
        :additions="syncDiff.additions"
        :removals="syncDiff.removals"
        :conflicts="syncConflicts"
        :pending="pending === 'sync'"
        :error="syncError"
        @update:open="setSyncDialogOpen"
        @update:mode="setSyncMode"
        @confirm="confirmSync"
      />

      <AppConfirmDialog
        appearance="ledger"
        :open="emptyConfirmOpen"
        :title="t('group.modelEditor.emptyConfirm.title')"
        :description="t('group.modelEditor.emptyConfirm.description')"
        :close-label="t('group.modelEditor.emptyConfirm.close')"
        :cancel-label="t('group.modelEditor.emptyConfirm.cancel')"
        :confirm-label="t('group.modelEditor.emptyConfirm.confirm')"
        tone="danger"
        :pending="pending === 'save'"
        @update:open="emptyConfirmOpen = $event"
        @confirm="save"
      />

      <StickySaveBar
        v-if="!unified"
        appearance="ledger"
        error-placement="floating"
        always-visible
        :dirty="dirty"
        :pending="savePending"
        :status="saveBarError ? 'error' : savedFeedback ? 'saved' : 'idle'"
        :error="saveBarError"
        :error-action-label="invalidRowCount ? t('group.modelEditor.locateFirstInvalid') : ''"
        @error-action="modelEditor?.focusFirstInvalid()"
        ><template #status
          ><div>
            <strong>
              {{
                savePending
                  ? t('group.modelEditor.saving')
                  : savedFeedback
                    ? t('group.modelEditor.savedFeedback')
                    : dirty
                      ? t('group.modelEditor.unsaved')
                      : t('group.modelEditor.saved')
              }}
            </strong>
            <span>
              {{
                savePending
                  ? t('group.modelEditor.savingNote')
                  : savedFeedback
                    ? t('group.modelEditor.savedFeedbackNote')
                    : dirty
                      ? invalidRowCount > 0
                        ? t('group.modelEditor.invalidNote', { count: invalidRowCount })
                        : t('group.modelEditor.dirtyNote')
                      : t('group.modelEditor.saveNote')
              }}
            </span>
          </div></template
        ><template #discard="{ disabled }"
          ><AppButton variant="ghost" size="sm" :disabled="disabled || !dirty" @click="discard">{{
            t('common.discard')
          }}</AppButton></template
        ><template #save="{ disabled }"
          ><AppButton size="sm" :disabled="disabled || !canSave" @click="requestSave">{{
            t('group.modelEditor.save')
          }}</AppButton></template
        ></StickySaveBar
      >
    </template>
  </section>
</template>

<style scoped>
.group-models__pricing-cell {
  display: grid;
  gap: 5px;
  justify-items: start;
}
.group-models__schedule-link {
  color: var(--color-action);
  font-size: var(--text-meta);
  font-weight: 600;
}
.group-models__breaker-summary {
  color: var(--color-text-muted);
  font-family: var(--font-mono);
  font-size: var(--text-meta);
}
.group-models {
  display: grid;
  gap: 0;
  min-width: 0;
  padding-top: var(--detail-panel-padding-top);
}

@media (max-width: 800px) {
  .group-models {
    padding-top: var(--detail-panel-padding-top-compact);
  }
}
</style>
