<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query'
import { computed, nextTick, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { useApiClient } from '@shared/http/client-context'
import { useToast } from '@/app/toast'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@/app/resources/invalidation'
import { groupDetailLocation } from '@/app/route-locations'
import {
  isReasoningEffort,
  reasoningEffortValues,
  isModelRouteScheduleRevisionConflict,
  recoverModelRouteScheduleEntry,
  updateModelRouteSchedule,
  type ModelRouteScheduleDetailDto,
  type ModelRouteScheduleEntryDto,
  type ModelRouteScheduleGroupDto,
  type ModelRouteSchedulePatchUpdate,
} from '@/app/resources/model-route-schedule'
import type { ReasoningEffortDto } from '@/api/control/types'
import type { ModelProbeTargetDto } from '@/app/resources/model-probe'
import AppButton from '@/components/ui/AppButton.vue'
import AppConfirmDialog from '@/components/ui/AppConfirmDialog.vue'
import AppSwitch from '@/components/ui/AppSwitch.vue'
import AppSelect from '@/components/ui/AppSelect.vue'
import AppTextInput from '@/components/ui/AppTextInput.vue'
import InlineFeedback from '@/components/ui/InlineFeedback.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import StickySaveBar from '@/components/ui/StickySaveBar.vue'
import ModelProbeScopeDialog from '@/features/models/ModelProbeScopeDialog.vue'
import { formatLocalInstant } from '@/lib/format'

import type { ScheduleDrafts, ScheduleMode } from './monitor-route'

export interface SchedulePanelDetailLabels {
  title: string
  loading: string
  refresh: string
  stale: string
  observedAt?: string
  routeUnavailable: string
  group: string
  upstreamModel: string
  weight: string
  priority: string
  share: string
  status: string
  available: string
  cooldown: string
  blacklisted: string
  failures: string
  recover: string
  breakerRecovery: string
  breakerThreshold?: string
  breakerCooldown?: string
  cooldownUntil?: string
  scheduledReleaseAt?: string
  invalidValue: string
  derivedReadOnly: string
  save: string
  discard: string
  unsaved: string
  saved: string
  saveFailed: string
  conflict: string
  refreshToResolve: string
  recoverFailed: string
  noEntries: string
  unknownReason?: string
  reasonLabels?: Partial<Record<string, string>>
  draftPreview?: string
  toggleEnabled?: string
  toggleFailed?: string
  disabled?: string
  calls24h?: string
  successRate24h?: string
  reasoning: string
  inherit: string
  groupDisabled: string
}

type EditableField = 'weight' | 'priority'
type Draft = Partial<Record<EditableField, number | null>>
type RecoverKey = string

const props = withDefaults(
  defineProps<{
    detail?: ModelRouteScheduleDetailDto
    loading?: boolean
    error?: string
    stale?: boolean
    labels?: Partial<SchedulePanelDetailLabels>
    locale?: string
    mode?: ScheduleMode
    selectedRow?: string
    sourceGroupId?: number
    drafts?: ScheduleDrafts
  }>(),
  {
    detail: undefined,
    loading: false,
    error: '',
    stale: false,
    labels: () => ({}),
    locale: 'en-US',
    mode: 'all',
    selectedRow: undefined,
    sourceGroupId: undefined,
    drafts: () => ({}),
  },
)
const emit = defineEmits<{
  refresh: []
  saved: [snapshotRevision: number]
  recovered: [groupId: number, entryId: string]
  probe: [groupId: number, modelId: string, disabled: boolean]
  'probe-all': [targets: ModelProbeTargetDto[], disabledGroupIds: number[]]
  'draft-change': [drafts: ScheduleDrafts]
  'row-change': [row: string | undefined]
}>()

const queryClient = useQueryClient()
const client = useApiClient()
const { t } = useI18n()
const toast = useToast()
const draftMap = reactive<Record<string, Draft>>({})
const entryReasoningDrafts = reactive<Record<string, ReasoningEffortDto | null>>({})
const rawInputs = reactive<Record<string, string>>({})
const invalidInputs = reactive<Record<string, boolean>>({})
const pending = ref(false)
const saveStatus = ref<'idle' | 'saved' | 'error'>('idle')
const saveError = ref('')
const recovering = ref<RecoverKey>('')
const togglingEntries = ref(new Set<string>())
const optimisticEnabled = ref(new Map<string, boolean>())
const preserveDraftRevisions = ref(new Set<number>())
const preserveDraftSnapshots = ref(new Map<number, ScheduleDrafts>())
// Ignore the one URL echo caused by a local edit; later history changes hydrate normally.
const pendingLocalDraftFingerprint = ref<string>()
const text = (key: keyof SchedulePanelDetailLabels): string => {
  const value = props.labels[key]
  return typeof value === 'string' && value ? value : t(`monitor.schedule.detail.${key}`)
}

const rows = computed(() =>
  (props.detail?.groups ?? [])
    .flatMap((group) =>
      group.entries
        .filter((entry) => {
          if (props.mode === 'primary') return !entry.fallback
          if (props.mode === 'fallback') return entry.fallback
          return true
        })
        .map((entry) => ({ group, entry })),
    )
    .sort(
      (a, b) =>
        a.entry.priority - b.entry.priority ||
        a.group.group_id - b.group.group_id ||
        (a.entry.entry_id < b.entry.entry_id ? -1 : a.entry.entry_id > b.entry.entry_id ? 1 : 0),
    ),
)

// URL 行定位：分组模型页带 sourceGroupId 进入时，在详情加载后把行选择解析到
// 来源分组的第一个匹配 entry（alias 或 model_id 等于当前外部模型）；目标行不
// 存在时静默清除行选择，保留模型上下文，不报错。
const scheduleTableRef = ref<HTMLElement>()

function revealScheduleRow(key: string): void {
  void nextTick(() => {
    const container = scheduleTableRef.value
    if (!container) return
    container
      .querySelector(`[data-row-key="${CSS.escape(key)}"]`)
      ?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  })
}

watch(
  () => [props.detail, props.sourceGroupId, props.selectedRow, props.mode] as const,
  async ([detail, sourceGroupId, selectedRow]) => {
    if (!detail || sourceGroupId === undefined) return
    const keyOf = ({ group, entry }: (typeof rows.value)[number]): string =>
      rowKey(group.group_id, entry.entry_id)
    const rowsByKey = new Map(rows.value.map((row) => [keyOf(row), row]))
    if (selectedRow !== undefined) {
      if (rowsByKey.has(selectedRow)) {
        revealScheduleRow(selectedRow)
        return
      }
      emit('row-change', undefined)
      return
    }
    const externalModel = detail.external_model ?? ''
    const target = rows.value.find(
      ({ group, entry }) =>
        group.group_id === sourceGroupId &&
        (entry.alias === externalModel || entry.model_id === externalModel),
    )
    if (target) emit('row-change', keyOf(target))
  },
)
// Batch scope is exactly what is on screen: the same mode-filtered rows the
// table renders, deduplicated to (group, model) targets. Disabled groups are
// split out so the operator decides whether to spend an upstream call on a group
// that is not serving traffic.
const probeScopes = computed(() => {
  const all: ModelProbeTargetDto[] = []
  const enabled: ModelProbeTargetDto[] = []
  const disabledGroupIds = new Set<number>()
  const seen = new Set<string>()
  for (const { group, entry } of rows.value) {
    const key = `${group.group_id}:${entry.model_id}`
    if (seen.has(key)) continue
    seen.add(key)
    const target = { group_id: group.group_id, model: entry.model_id }
    all.push(target)
    if (groupEnabled(group)) {
      enabled.push(target)
    } else {
      disabledGroupIds.add(group.group_id)
    }
  }
  return { all, enabled, disabledGroupIds: [...disabledGroupIds] }
})
const probeScopeOpen = ref(false)
const pendingProbe = ref<{ groupId: number; modelId: string } | null>(null)

// A disabled row still probes on explicit request, so it asks first.
function requestProbe(groupId: number, modelId: string, disabled: boolean): void {
  if (!disabled) {
    emit('probe', groupId, modelId, false)
    return
  }
  pendingProbe.value = { groupId, modelId }
}

function confirmSingleProbe(): void {
  const pending = pendingProbe.value
  pendingProbe.value = null
  if (pending) emit('probe', pending.groupId, pending.modelId, true)
}

function handleSingleProbeOpen(value: boolean): void {
  if (!value) pendingProbe.value = null
}

function probeVisibleRows(): void {
  const { all, enabled } = probeScopes.value
  if (all.length === 0) return
  if (enabled.length === all.length) {
    emit('probe-all', all, [])
    return
  }
  probeScopeOpen.value = true
}

function confirmProbeAll(): void {
  const { all, disabledGroupIds } = probeScopes.value
  probeScopeOpen.value = false
  emit('probe-all', all, disabledGroupIds)
}

function confirmProbeEnabled(): void {
  const { enabled } = probeScopes.value
  probeScopeOpen.value = false
  emit('probe-all', enabled, [])
}

const dirty = computed(
  () => Object.keys(draftMap).length > 0 || Object.keys(entryReasoningDrafts).length > 0,
)
const invalid = computed(() => Object.values(invalidInputs).some(Boolean))
const hasDetail = computed(() => props.detail !== undefined)
const hasScheduleDraft = computed(() => Object.keys(draftMap).length > 0)
const previewShares = computed(() => {
  const result = new Map<string, number>()
  if (!props.detail || !hasScheduleDraft.value) return result
  const entries = props.detail.groups.flatMap((group) =>
    group.entries.map((entry) => {
      const draft = draftMap[draftKey(group.group_id, entry.entry_id)]
      return {
        group,
        entry,
        weight: draft?.weight ?? entry.weight,
        priority: draft?.priority ?? entry.priority,
      }
    }),
  )
  const totals = new Map<number, number>()
  for (const candidate of entries) {
    if (entryEnabled(candidate.group.group_id, candidate.entry))
      totals.set(candidate.priority, (totals.get(candidate.priority) ?? 0) + candidate.weight)
  }
  for (const candidate of entries) {
    const total = totals.get(candidate.priority) ?? 0
    result.set(
      rowKey(candidate.group.group_id, candidate.entry.entry_id),
      entryEnabled(candidate.group.group_id, candidate.entry) && total > 0
        ? Math.max(0, candidate.weight) / total
        : 0,
    )
  }
  return result
})
const observedLabel = computed(() => {
  const observed = props.detail?.observed_at_ms
  return observed === undefined
    ? ''
    : `${text('observedAt')} ${formatLocalInstant(observed, props.locale)}`
})

function groupEnabled(group: ModelRouteScheduleGroupDto): boolean {
  return group.enabled
}

function entryEnabled(groupID: number, entry: ModelRouteScheduleEntryDto): boolean {
  return optimisticEnabled.value.get(rowKey(groupID, entry.entry_id)) ?? entry.enabled
}

async function toggleEntryEnabled(
  groupID: number,
  entry: ModelRouteScheduleEntryDto,
  next: boolean,
): Promise<void> {
  const detail = props.detail
  const key = rowKey(groupID, entry.entry_id)
  if (!detail || togglingEntries.value.has(key) || entry.entry_id.startsWith('derived:')) return
  togglingEntries.value = new Set(togglingEntries.value).add(key)
  optimisticEnabled.value = new Map(optimisticEnabled.value).set(key, next)
  preserveDraftRevisions.value = new Set(preserveDraftRevisions.value).add(detail.snapshot_revision)
  preserveDraftSnapshots.value = new Map(preserveDraftSnapshots.value).set(
    detail.snapshot_revision,
    cloneDrafts(draftMap),
  )
  try {
    const response = await updateModelRouteSchedule(client, {
      snapshot_revision: detail.snapshot_revision,
      protocol: detail.protocol,
      external_model: detail.external_model ?? '',
      operation: detail.operation,
      updates: [{ group_id: groupID, entry_id: entry.entry_id, enabled: next }],
    })
    await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.update)
    emit('saved', response.snapshot_revision_new)
  } catch (error: unknown) {
    toast.show({
      message: isModelRouteScheduleRevisionConflict(error)
        ? text('conflict')
        : text('toggleFailed'),
      tone: 'danger',
    })
  } finally {
    await nextTick()
    const revisions = new Set(preserveDraftRevisions.value)
    revisions.delete(detail.snapshot_revision)
    preserveDraftRevisions.value = revisions
    const snapshots = new Map(preserveDraftSnapshots.value)
    snapshots.delete(detail.snapshot_revision)
    preserveDraftSnapshots.value = snapshots
    const optimistic = new Map(optimisticEnabled.value)
    optimistic.delete(key)
    optimisticEnabled.value = optimistic
    const pending = new Set(togglingEntries.value)
    pending.delete(key)
    togglingEntries.value = pending
  }
}

function isPriorityStart(index: number): boolean {
  return index === 0 || rows.value[index - 1]?.entry.priority !== rows.value[index]?.entry.priority
}

function formatCount(value: number): string {
  return new Intl.NumberFormat(props.locale).format(value)
}

function formatRate(value: number): string {
  return new Intl.NumberFormat(props.locale, {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value)
}

function rowKey(groupID: number, entryID: string): string {
  return `${groupID}:${entryID}`
}

function draftKey(groupID: number, entryID: string): string {
  return rowKey(groupID, entryID)
}

function fieldKey(groupID: number, entryID: string, field: EditableField): string {
  return `${draftKey(groupID, entryID)}\u0000${field}`
}

function hasOwn(source: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(source, key)
}

function entryReasoningValue(
  groupID: number,
  entry: ModelRouteScheduleEntryDto,
): ReasoningEffortDto | '' {
  const key = draftKey(groupID, entry.entry_id)
  return hasOwn(entryReasoningDrafts, key)
    ? (entryReasoningDrafts[key] ?? '')
    : (entry.reasoning.configured ?? '')
}

// 未配置项把有效值并入选项文案：单元格只有一行，操作者不用展开
// 即可知道该条目当前实际生效的推理强度来自哪里。
function reasoningOptionsFor(entry: ModelRouteScheduleEntryDto) {
  const effective = entry.reasoning.effective
  const inheritLabel = effective === null ? text('inherit') : `${text('inherit')} · ${effective}`
  return [
    { value: '', label: inheritLabel },
    ...reasoningEffortValues.map((value) => ({ value, label: value })),
  ]
}

function setEntryReasoning(
  groupID: number,
  entry: ModelRouteScheduleEntryDto,
  value: string,
): void {
  if (value !== '' && !isReasoningEffort(value)) return
  const next = value === '' ? null : value
  const key = draftKey(groupID, entry.entry_id)
  if (next === entry.reasoning.configured) delete entryReasoningDrafts[key]
  else entryReasoningDrafts[key] = next
}

function configuredValue(entry: ModelRouteScheduleEntryDto, field: EditableField): number | null {
  if (field === 'weight') return null
  return entry.priority === 1 ? null : entry.priority
}

function effectiveValue(entry: ModelRouteScheduleEntryDto, field: EditableField): number {
  return field === 'weight' ? entry.weight : entry.priority
}

function inputValue(
  groupID: number,
  entry: ModelRouteScheduleEntryDto,
  field: EditableField,
): string {
  const key = fieldKey(groupID, entry.entry_id, field)
  if (rawInputs[key] !== undefined) return rawInputs[key]
  const value = draftMap[draftKey(groupID, entry.entry_id)]?.[field]
  return value === null || value === undefined
    ? String(effectiveValue(entry, field))
    : String(value)
}

function placeholder(entry: ModelRouteScheduleEntryDto, field: EditableField): string {
  const configured = configuredValue(entry, field)
  return String(configured === null ? effectiveValue(entry, field) : configured)
}

function isValidValue(field: EditableField, value: number): boolean {
  if (!Number.isSafeInteger(value)) return false
  return field === 'weight' ? value >= 0 && value <= 100 : value >= 1
}

function setDraftValue(
  groupID: number,
  entry: ModelRouteScheduleEntryDto,
  field: EditableField,
  value: number | null,
): void {
  const key = draftKey(groupID, entry.entry_id)
  const baseline = configuredValue(entry, field)
  const effective = effectiveValue(entry, field)
  const sameAsServer = value !== null && value === (baseline ?? effective)
  const draft = draftMap[key] ?? {}
  if (value === null) draft[field] = null
  else if (sameAsServer) delete draft[field]
  else draft[field] = value

  if (Object.keys(draft).length === 0) delete draftMap[key]
  else draftMap[key] = draft
  emitDraftChange({ ...draftMap })
  emit('row-change', key)
}

function setInput(
  groupID: number,
  entry: ModelRouteScheduleEntryDto,
  field: EditableField,
  value: string,
): void {
  const key = fieldKey(groupID, entry.entry_id, field)
  rawInputs[key] = value
  delete invalidInputs[key]
  if (value.trim() === '') {
    setDraftValue(groupID, entry, field, null)
    return
  }
  const parsed = Number(value)
  if (!/^\d+$/.test(value.trim()) || !isValidValue(field, parsed)) {
    invalidInputs[key] = true
    return
  }
  setDraftValue(groupID, entry, field, parsed)
}

function resetDrafts(): void {
  for (const key of Object.keys(draftMap)) delete draftMap[key]
  for (const key of Object.keys(entryReasoningDrafts)) delete entryReasoningDrafts[key]
  for (const key of Object.keys(rawInputs)) delete rawInputs[key]
  for (const key of Object.keys(invalidInputs)) delete invalidInputs[key]
  saveStatus.value = 'idle'
  saveError.value = ''
}

function draftFingerprint(source: ScheduleDrafts): string {
  return JSON.stringify(
    Object.keys(source)
      .sort()
      .map((key) => [key, source[key]]),
  )
}

function cloneDrafts(source: ScheduleDrafts): ScheduleDrafts {
  return Object.fromEntries(
    Object.entries(source).map(([key, draft]) => [key, draft === undefined ? {} : { ...draft }]),
  )
}

function emitDraftChange(source: ScheduleDrafts): void {
  for (const revision of preserveDraftRevisions.value) {
    preserveDraftSnapshots.value = new Map(preserveDraftSnapshots.value).set(
      revision,
      cloneDrafts(source),
    )
  }
  pendingLocalDraftFingerprint.value = draftFingerprint(source)
  emit('draft-change', source)
}

function hydrateDraftState(source: ScheduleDrafts): void {
  resetDrafts()
  pendingLocalDraftFingerprint.value = undefined
  const rowsByKey = new Map(
    rows.value.map(({ group, entry }) => [
      draftKey(group.group_id, entry.entry_id),
      { group, entry },
    ]),
  )
  for (const [key, draft] of Object.entries(source)) {
    const row = rowsByKey.get(key)
    if (!row) continue
    const next = { ...draft }
    draftMap[key] = next
    for (const field of ['weight', 'priority'] as const) {
      if (!Object.prototype.hasOwnProperty.call(next, field)) continue
      const value = next[field]
      rawInputs[fieldKey(row.group.group_id, row.entry.entry_id, field)] =
        value === null || value === undefined ? '' : String(value)
    }
  }
}

type DetailWatchKey = [
  revision: number,
  externalModel: string | null,
  protocol: string,
]

watch(
  (): DetailWatchKey => {
    const detail = props.detail
    return detail
      ? [detail.snapshot_revision, detail.external_model, detail.protocol]
      : [0, '', '']
  },
  (value, previous) => {
    if (previous?.[0] && previous[0] !== value[0]) {
      const preserve = preserveDraftRevisions.value.has(previous[0])
      if (preserve) {
        const revisions = new Set(preserveDraftRevisions.value)
        revisions.delete(previous[0])
        preserveDraftRevisions.value = revisions
        const snapshots = new Map(preserveDraftSnapshots.value)
        const draftSnapshot = snapshots.get(previous[0]) ?? props.drafts
        snapshots.delete(previous[0])
        preserveDraftSnapshots.value = snapshots
        hydrateDraftState(draftSnapshot)
        return
      }
      hydrateDraftState({})
      emitDraftChange({})
      return
    }
    hydrateDraftState(props.drafts)
  },
  { immediate: true },
)

watch(
  () => props.drafts,
  (source) => {
    const fingerprint = draftFingerprint(source)
    if (pendingLocalDraftFingerprint.value === fingerprint) {
      pendingLocalDraftFingerprint.value = undefined
      return
    }
    hydrateDraftState(source)
  },
  { deep: true },
)

function updates(): ModelRouteSchedulePatchUpdate[] {
  const result: ModelRouteSchedulePatchUpdate[] = []
  for (const group of props.detail?.groups ?? []) {
    for (const entry of group.entries) {
      const draft = draftMap[draftKey(group.group_id, entry.entry_id)]
      const reasoningKey = draftKey(group.group_id, entry.entry_id)
      if (
        (!draft && !hasOwn(entryReasoningDrafts, reasoningKey)) ||
        entry.entry_id.startsWith('derived:')
      )
        continue
      const update: ModelRouteSchedulePatchUpdate = {
        group_id: group.group_id,
        entry_id: entry.entry_id,
      }
      if (draft && hasOwn(draft, 'weight')) update.weight = draft.weight
      if (draft && hasOwn(draft, 'priority')) update.priority = draft.priority
      if (hasOwn(entryReasoningDrafts, reasoningKey)) {
        update.reasoning_effort = entryReasoningDrafts[reasoningKey] ?? null
      }
      if (Object.keys(update).length > 2) result.push(update)
    }
  }
  return result
}

async function save(): Promise<void> {
  if (!props.detail || !dirty.value || invalid.value) return
  const body = {
    snapshot_revision: props.detail.snapshot_revision,
    protocol: props.detail.protocol,
    external_model: props.detail.external_model ?? '',
    operation: props.detail.operation,
    updates: updates(),
  }
  if (body.updates.length === 0) return
  pending.value = true
  saveStatus.value = 'idle'
  saveError.value = ''
  try {
    const response = await updateModelRouteSchedule(client, body)
    await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.update)
    resetDrafts()
    emitDraftChange({})
    saveStatus.value = 'saved'
    emit('saved', response.snapshot_revision_new)
  } catch (error: unknown) {
    saveStatus.value = 'error'
    saveError.value = isModelRouteScheduleRevisionConflict(error)
      ? text('conflict')
      : text('saveFailed')
  } finally {
    pending.value = false
  }
}

function discard(): void {
  resetDrafts()
  emitDraftChange({})
}

async function recover(groupID: number, entry: ModelRouteScheduleEntryDto): Promise<void> {
  const key = rowKey(groupID, entry.entry_id)
  recovering.value = key
  saveError.value = ''
  try {
    await recoverModelRouteScheduleEntry(client, {
      group_id: groupID,
      entry_id: entry.entry_id,
      failure_version: entry.runtime.failure_version,
    })
    await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.recover)
    emit('recovered', groupID, entry.entry_id)
  } catch {
    saveError.value = text('recoverFailed')
  } finally {
    recovering.value = ''
  }
}

function runtimeLabel(entry: ModelRouteScheduleEntryDto): string {
  if (entry.runtime.state === 'blacklisted') return text('blacklisted')
  if (entry.runtime.state === 'cooldown') return text('cooldown')
  return text('available')
}

function runtimeTone(entry: ModelRouteScheduleEntryDto): string {
  return `schedule-detail__runtime--${entry.runtime.state}`
}

function shareValue(groupID: number, entry: ModelRouteScheduleEntryDto): number {
  if (!entryEnabled(groupID, entry)) return 0
  return previewShares.value.get(rowKey(groupID, entry.entry_id)) ?? entry.configured_share
}

function isDraftShare(groupID: number, entry: ModelRouteScheduleEntryDto): boolean {
  return previewShares.value.has(rowKey(groupID, entry.entry_id))
}

function breakerRecoveryLabel(entry: ModelRouteScheduleEntryDto): string {
  const breaker = entry.circuit_breaker.effective
  const parts: string[] = []
  if (breaker.blacklist_threshold !== null) {
    parts.push(`${text('breakerThreshold')} ${breaker.blacklist_threshold}`)
  }
  if (breaker.cooldown_seconds !== null) {
    parts.push(`${text('breakerCooldown')} ${breaker.cooldown_seconds}s`)
  }
  if (entry.runtime.cooldown_until_ms !== null) {
    parts.push(
      `${text('cooldownUntil')}: ${formatLocalInstant(entry.runtime.cooldown_until_ms, props.locale)}`,
    )
  }
  if (entry.runtime.blacklist_release_at_ms !== null) {
    parts.push(
      `${text('scheduledReleaseAt')}: ${formatLocalInstant(entry.runtime.blacklist_release_at_ms, props.locale)}`,
    )
  }
  return parts.join(' · ')
}
</script>

<template>
  <section class="schedule-detail" aria-labelledby="schedule-detail-title">
    <header class="schedule-detail__header">
      <div>
        <h2 id="schedule-detail-title">{{ text('title') }}</h2>
      </div>
      <div class="schedule-detail__observed">
        <span>{{ observedLabel }}</span>
      </div>
      <AppButton
        v-if="rows.length > 0"
        variant="secondary"
        size="compact"
        :disabled="pending"
        :title="t('monitor.modelProbe.description')"
        @click="probeVisibleRows"
      >
        {{ t('monitor.modelProbe.batch', { count: probeScopes.all.length }) }}
      </AppButton>
    </header>

    <QueryFeedback v-if="loading" state="loading" :message="text('loading')" />
    <QueryFeedback
      v-else-if="error"
      state="error"
      :message="error"
      :retry-label="text('refresh')"
      @retry="emit('refresh')"
    />
    <template v-else-if="hasDetail && detail">
      <QueryFeedback
        v-if="stale"
        state="stale"
        :message="text('stale')"
        :retry-label="text('refresh')"
        @retry="emit('refresh')"
      />
      <QueryFeedback
        v-if="detail.external_model === null || !detail.routable"
        state="stale"
        :message="text('routeUnavailable')"
        :retry-label="text('refresh')"
        @retry="emit('refresh')"
      />
      <InlineFeedback v-if="saveError" tone="danger">
        {{ saveError }}
        <template #action>
          <AppButton variant="link" size="inline" @click="emit('refresh')">
            {{ text('refreshToResolve') }}
          </AppButton>
        </template>
      </InlineFeedback>

      <div v-if="rows.length === 0" class="schedule-detail__empty" role="status">
        {{ text('noEntries') }}
      </div>
      <div v-else ref="scheduleTableRef" class="schedule-table-wrap">
        <div class="schedule-table" role="table" :aria-label="text('title')">
          <div class="schedule-row schedule-row--header" role="row">
            <span role="columnheader">{{ text('priority') }}</span>
            <span role="columnheader">{{ text('upstreamModel') }}</span>
            <span role="columnheader">{{ text('group') }}</span>
            <span role="columnheader">{{ text('reasoning') }}</span>
            <span role="columnheader">{{ text('weight') }}</span>
            <span role="columnheader">{{ text('share') }}</span>
            <span role="columnheader">{{ text('status') }}</span>
            <span role="columnheader">{{ text('breakerRecovery') }}</span>
          </div>
          <div
            v-for="({ group, entry }, index) in rows"
            :key="rowKey(group.group_id, entry.entry_id)"
            class="schedule-row"
            :data-row-key="rowKey(group.group_id, entry.entry_id)"
            :class="{
              'schedule-row--selected': selectedRow === rowKey(group.group_id, entry.entry_id),
              'schedule-row--priority-start': isPriorityStart(index),
            }"
            role="row"
            @click="emit('row-change', rowKey(group.group_id, entry.entry_id))"
          >
            <div class="schedule-cell schedule-cell--priority" role="cell">
              <span class="schedule-cell__label">{{ text('priority') }}</span>
              <div class="schedule-cell__priority-edit">
                <AppTextInput
                  :id="`priority-${index}`"
                  :model-value="inputValue(group.group_id, entry, 'priority')"
                  type="number"
                  :label="`${text('priority')} ${entry.model_id}`"
                  :placeholder="placeholder(entry, 'priority')"
                  :invalid="invalidInputs[fieldKey(group.group_id, entry.entry_id, 'priority')]"
                  :disabled="entry.entry_id.startsWith('derived:')"
                  size="compact"
                  @update:model-value="setInput(group.group_id, entry, 'priority', $event)"
                />
              </div>
            </div>
            <div class="schedule-cell schedule-cell--model" role="cell">
              <span class="schedule-cell__label">{{ text('upstreamModel') }}</span>
              <strong>{{ entry.alias || entry.model_id }}</strong>
              <small v-if="entry.alias">{{ entry.model_id }}</small>
              <AppSwitch
                :model-value="entryEnabled(group.group_id, entry)"
                :disabled="
                  pending ||
                  togglingEntries.has(rowKey(group.group_id, entry.entry_id)) ||
                  entry.entry_id.startsWith('derived:')
                "
                :label="`${text('toggleEnabled')} ${entry.model_id}`"
                @click.stop
                @update:model-value="toggleEntryEnabled(group.group_id, entry, $event)"
              />
            </div>
            <div class="schedule-cell schedule-cell--group" role="cell">
              <span class="schedule-cell__label">{{ text('group') }}</span>
              <RouterLink
                class="schedule-cell__group-link"
                :to="groupDetailLocation(group.group_id)"
                @click.stop
              >
                <strong>{{ group.group_name }}</strong>
              </RouterLink>
              <small class="schedule-cell__stats">
                {{ text('calls24h') }}: {{ formatCount(group.request_count) }} ·
                {{ text('successRate24h') }}: {{ formatRate(group.success_rate) }}
              </small>
            </div>
            <div class="schedule-cell schedule-cell--reasoning" role="cell">
              <span class="schedule-cell__label">{{ text('reasoning') }}</span>
              <AppSelect
                :model-value="entryReasoningValue(group.group_id, entry)"
                :options="reasoningOptionsFor(entry)"
                :label="`${text('reasoning')} ${entry.model_id}`"
                :disabled="pending || entry.entry_id.startsWith('derived:')"
                size="compact"
                @click.stop
                @update:model-value="setEntryReasoning(group.group_id, entry, $event)"
              />
            </div>
            <div class="schedule-cell schedule-cell--input" role="cell">
              <label class="schedule-cell__label" :for="`weight-${index}`">{{ text('weight') }}</label>
              <AppTextInput
                :id="`weight-${index}`"
                :model-value="inputValue(group.group_id, entry, 'weight')"
                type="number"
                :label="`${text('weight')} ${entry.model_id}`"
                :placeholder="placeholder(entry, 'weight')"
                :invalid="invalidInputs[fieldKey(group.group_id, entry.entry_id, 'weight')]"
                :disabled="entry.entry_id.startsWith('derived:')"
                size="compact"
                @update:model-value="setInput(group.group_id, entry, 'weight', $event)"
              />
            </div>
            <div class="schedule-cell schedule-cell--share" role="cell">
              <span class="schedule-cell__label">{{ text('share') }}</span>
              <strong>{{ (shareValue(group.group_id, entry) * 100).toFixed(1) }}%</strong>
              <span
                class="schedule-share-track"
                :aria-label="`${text('share')} ${(shareValue(group.group_id, entry) * 100).toFixed(1)}%`"
              >
                <span :style="{ width: `${shareValue(group.group_id, entry) * 100}%` }" />
              </span>
              <small v-if="isDraftShare(group.group_id, entry)">{{ text('draftPreview') }}</small>
            </div>
            <div class="schedule-cell schedule-cell--status" role="cell">
              <span class="schedule-cell__label">{{ text('status') }}</span>
              <strong
                v-if="!entryEnabled(group.group_id, entry)"
                class="schedule-detail__runtime--blacklisted"
                >{{ text('disabled') }}</strong
              >
              <strong v-else-if="!groupEnabled(group)" class="schedule-detail__runtime--cooldown">{{
                text('groupDisabled')
              }}</strong>
              <strong v-else :class="runtimeTone(entry)">{{ runtimeLabel(entry) }}</strong>
              <small v-if="entry.runtime.failure_count"
                >{{ entry.runtime.failure_count }} {{ text('failures') }}</small
              >
            </div>
            <div class="schedule-cell schedule-cell--breaker" role="cell">
              <span class="schedule-cell__label">{{ text('breakerRecovery') }}</span>
              <span>{{ breakerRecoveryLabel(entry) }}</span>
              <AppButton
                v-if="entry.runtime.state !== 'available'"
                variant="secondary"
                size="compact"
                :busy="recovering === rowKey(group.group_id, entry.entry_id)"
                :disabled="entry.entry_id.startsWith('derived:')"
                @click.stop="recover(group.group_id, entry)"
              >
                {{ text('recover') }}
              </AppButton>
              <AppButton
                variant="secondary"
                size="compact"
                :disabled="pending || entry.entry_id.startsWith('derived:')"
                @click.stop="requestProbe(group.group_id, entry.model_id, !groupEnabled(group))"
              >
                {{ t('monitor.modelProbe.button') }}
              </AppButton>
            </div>
          </div>
        </div>
      </div>

      <StickySaveBar
        appearance="ledger"
        :dirty="dirty"
        :pending="pending"
        :status="saveStatus"
        :error="saveStatus === 'error' ? saveError : ''"
        :error-action-label="saveStatus === 'error' ? text('refreshToResolve') : ''"
        @error-action="emit('refresh')"
      >
        <template #status>
          <span v-if="saveStatus === 'saved'">{{ text('saved') }}</span>
          <span v-else-if="dirty">{{ text('unsaved') }}</span>
        </template>
        <template #discard="{ disabled }">
          <AppButton variant="ghost" size="compact" :disabled="disabled" @click="discard">
            {{ text('discard') }}
          </AppButton>
        </template>
        <template #save="{ disabled }">
          <AppButton variant="primary" size="compact" :disabled="disabled || invalid" @click="save">
            {{ text('save') }}
          </AppButton>
        </template>
      </StickySaveBar>
    </template>

    <ModelProbeScopeDialog
      :open="probeScopeOpen"
      :total="probeScopes.all.length"
      :disabled-count="probeScopes.all.length - probeScopes.enabled.length"
      @update:open="probeScopeOpen = $event"
      @probe-all="confirmProbeAll"
      @probe-enabled="confirmProbeEnabled"
    />
    <AppConfirmDialog
      :open="pendingProbe !== null"
      :title="t('monitor.modelProbe.disabledConfirm.title')"
      :description="t('monitor.modelProbe.disabledConfirm.description')"
      :close-label="t('monitor.modelProbe.disabledConfirm.cancel')"
      :cancel-label="t('monitor.modelProbe.disabledConfirm.cancel')"
      :confirm-label="t('monitor.modelProbe.disabledConfirm.confirm')"
      appearance="ledger"
      @update:open="handleSingleProbeOpen"
      @confirm="confirmSingleProbe"
    />
  </section>
</template>

<style scoped>
.schedule-detail {
  display: grid;
  min-width: 0;
  gap: var(--space-3);
}
.schedule-detail__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-3);
}
.schedule-detail h2 {
  margin: 0;
  color: var(--color-text);
  font-size: var(--text-lg);
}
.schedule-detail__observed {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px 12px;
  color: var(--color-text-faint);
  font-size: var(--text-meta);
  text-align: right;
}
.schedule-detail__empty {
  border: 1px dashed var(--color-border-control);
  color: var(--color-text-muted);
  padding: 28px;
  text-align: center;
}
.schedule-table-wrap {
  /* position 让滚动容器成为内部绝对定位元素（如 .sr-only）的包含块：
     否则这些 1px 隐藏标签会把容器外的位置计入文档可滚动溢出，
     在窄屏上把整个页面撑宽。 */
  position: relative;
  min-width: 0;
  max-width: 100%;
  max-height: 72vh;
  overflow: auto;
  border: 1px solid var(--color-border-subtle);
}
.schedule-table {
  min-width: 1020px;
}
.schedule-row {
  display: grid;
  grid-template-columns:
    80px minmax(160px, 1.3fr) minmax(140px, 1fr) minmax(112px, 0.8fr)
    76px 84px 104px minmax(160px, 1.1fr);
  min-height: 48px;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid var(--color-border-subtle);
  padding: 7px 12px;
}
.schedule-row:last-child {
  border-bottom: 0;
}
.schedule-row--header {
  position: sticky;
  z-index: 1;
  top: 0;
  min-height: 34px;
  background: var(--color-surface-sunken);
  color: var(--color-text-muted);
  font-size: 10px;
  font-weight: 700;
}
.schedule-row--priority-start:not(.schedule-row--header) {
  border-top: 2px solid var(--color-border-strong);
}
.schedule-row--selected,
.schedule-row:not(.schedule-row--header):hover {
  background: var(--color-action-soft);
}
.schedule-cell {
  min-width: 0;
  color: var(--color-text-muted);
  font-size: var(--text-meta);
}
.schedule-cell--priority {
  align-self: stretch;
  display: grid;
  align-content: center;
  justify-items: start;
  border-inline-start: 3px solid var(--color-action);
  padding-inline-start: 9px;
}
.schedule-cell__priority-edit {
  display: flex;
  align-items: center;
  gap: 3px;
  margin-top: 4px;
}
.schedule-cell__priority-edit :deep(.app-text-input) {
  width: 58px;
  flex: none;
}
.schedule-cell :deep(input[type='number']) {
  color: var(--color-text);
  -moz-appearance: textfield;
  appearance: textfield;
}
.schedule-cell :deep(input[type='number']::-webkit-inner-spin-button),
.schedule-cell :deep(input[type='number']::-webkit-outer-spin-button) {
  -webkit-appearance: none;
  margin: 0;
}
.schedule-cell :deep(input[type='number']::placeholder) {
  color: var(--color-text-muted);
  opacity: 1;
}
.schedule-cell__label {
  display: block;
  margin-bottom: 2px;
  color: var(--color-text-faint);
  font-size: 10px;
  font-weight: 650;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.schedule-cell strong,
.schedule-cell small {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.schedule-cell--reasoning {
  display: grid;
  align-content: center;
  gap: 2px;
}
.schedule-cell--reasoning :deep(.app-select__trigger) {
  width: 100%;
}
.schedule-cell strong {
  color: var(--color-text);
  font-weight: 650;
}
.schedule-cell__group-link {
  display: block;
  color: inherit;
  text-decoration: none;
}
.schedule-cell__group-link:hover strong {
  color: var(--color-action);
}
@media (min-width: 621px) {
  .schedule-cell__label {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
  }
}
.schedule-cell__stats {
  color: var(--color-text-faint);
  font-family: var(--font-mono);
}
.schedule-cell small {
  margin-top: 2px;
  color: var(--color-text-faint);
  font-size: 10px;
}
.schedule-cell--input {
  display: grid;
  align-content: center;
  gap: 2px;
}
.schedule-cell--input :deep(.app-text-input__input) {
  width: 62px;
  font-family: var(--font-mono);
}
.schedule-cell--share {
  display: grid;
  align-content: center;
  color: var(--color-action);
  font-family: var(--font-mono);
  font-weight: 700;
}
.schedule-share-track {
  display: block;
  height: 4px;
  margin-top: 5px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--color-border-subtle);
}
.schedule-share-track span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--color-action);
}
.schedule-cell--breaker {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  font-family: var(--font-mono);
}
.schedule-detail__runtime--available {
  color: var(--color-success) !important;
}
.schedule-detail__runtime--cooldown {
  color: var(--color-warning) !important;
}
.schedule-detail__runtime--blacklisted {
  color: var(--color-danger) !important;
}
.schedule-detail :deep(.sticky-save-bar) {
  position: sticky;
  z-index: 2;
  bottom: 12px;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}
@media (max-width: 620px) {
  .schedule-detail__header {
    display: grid;
  }
  .schedule-detail__observed {
    justify-content: flex-start;
    text-align: left;
  }
  .schedule-table-wrap {
    max-height: none;
    overflow-x: visible;
  }
  .schedule-table {
    min-width: 0;
  }
  .schedule-row--header {
    display: none;
  }
  .schedule-row {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 10px 12px;
    padding: 12px;
  }
  .schedule-cell--priority,
  .schedule-cell--model,
  .schedule-cell--group,
  .schedule-cell--share,
  .schedule-cell--status,
  .schedule-cell--breaker,
  .schedule-cell--input {
    grid-column: auto;
  }
  .schedule-cell--reasoning {
    grid-column: 1 / -1;
  }
  .schedule-cell--breaker {
    align-items: flex-start;
    flex-direction: column;
  }
  .schedule-cell--input :deep(.app-text-input__input) {
    width: 100%;
  }
}
</style>
