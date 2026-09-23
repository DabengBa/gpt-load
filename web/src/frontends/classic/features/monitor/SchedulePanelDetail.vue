<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query'
import { computed, nextTick, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { useApiClient } from '@shared/http/client-context'
import { useToast } from '@/app/toast'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@/app/resources/invalidation'
import { groupDetailLocation } from '@/app/route-locations'
import {
  cacheGroupSettings,
  invalidateGroupSettingsDependents,
  updateGroupSettings,
} from '@/app/resources/groups'
import {
  reasoningEffortValues,
  type ModelRouteScheduleReasoningEntryDto,
  isReasoningEffort,
  isModelRouteScheduleRevisionConflict,
  recoverModelRouteScheduleEntry,
  updateModelRouteSchedule,
  type ModelRouteScheduleDetailDto,
  type ModelRouteScheduleEntryDto,
  type ModelRouteScheduleGroupPatchUpdate,
  type ModelRouteScheduleGroupDto,
  type ModelRouteSchedulePatchUpdate,
  type ModelRouteScheduleReasoningSource,
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
  cooldownUntil?: string
  scheduledReleaseAt?: string
  clear: string
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
  enabled?: string
  disabled?: string
  calls24h?: string
  successRate24h?: string
  reasoning: string
  groupDefault: string
  entryOverride: string
  inherit: string
  effective: string
  source: string
  capability: string
  supported: string
  unsupported: string
  capabilityUnknown: string
  sourceEntry: string
  sourceGroup: string
  sourceClient: string
  sourceProviderDefault: string
}

type EditableField = 'weight' | 'priority'
type Draft = Partial<Record<EditableField, number | null>>
type RecoverKey = string

const props = withDefaults(
  defineProps<{
    detail?: ModelRouteScheduleDetailDto
    loading?: boolean
    error?: string
    refreshing?: boolean
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
    refreshing: false,
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
const groupReasoningDrafts = reactive<Record<string, ReasoningEffortDto | null>>({})
const rawInputs = reactive<Record<string, string>>({})
const invalidInputs = reactive<Record<string, boolean>>({})
const pending = ref(false)
const saveStatus = ref<'idle' | 'saved' | 'error'>('idle')
const saveError = ref('')
const recovering = ref<RecoverKey>('')
const togglingGroupIDs = ref(new Set<number>())
const optimisticEnabled = ref(new Map<number, boolean>())
const preserveDraftRevisions = ref(new Set<number>())
const preserveDraftSnapshots = ref(new Map<number, ScheduleDrafts>())
// Ignore the one URL echo caused by a local edit; later history changes hydrate normally.
const pendingLocalDraftFingerprint = ref<string>()

const text = (key: keyof SchedulePanelDetailLabels): string => {
  const value = props.labels[key]
  return typeof value === 'string' && value ? value : t(`monitor.schedule.detail.${key}`)
}

const rows = computed(() =>
  (props.detail?.groups ?? []).flatMap((group) =>
    group.entries
      .filter((entry) => {
        if (props.mode === 'primary') return !entry.fallback
        if (props.mode === 'fallback') return entry.fallback
        return true
      })
      .map((entry) => ({ group, entry })),
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
  () =>
    Object.keys(draftMap).length > 0 ||
    Object.keys(entryReasoningDrafts).length > 0 ||
    Object.keys(groupReasoningDrafts).length > 0,
)
const invalid = computed(
  () =>
    Object.values(invalidInputs).some(Boolean) ||
    reasoningPreviews.value.some(({ entries }) => entries.some(({ unsupported }) => unsupported)),
)
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
    totals.set(candidate.priority, (totals.get(candidate.priority) ?? 0) + candidate.weight)
  }
  for (const candidate of entries) {
    const total = totals.get(candidate.priority) ?? 0
    result.set(
      rowKey(candidate.group.group_id, candidate.entry.entry_id),
      total > 0 ? Math.max(0, candidate.weight) / total : 0,
    )
  }
  return result
})
const observedLabel = computed(() => {
  const observed = props.detail?.observed_at_ms
  return observed === undefined
    ? ''
    : `${text('stale')} ${formatLocalInstant(observed, props.locale)}`
})

function groupEnabled(group: ModelRouteScheduleGroupDto): boolean {
  return optimisticEnabled.value.get(group.group_id) ?? group.enabled
}

function groupTogglePending(groupID: number): boolean {
  return togglingGroupIDs.value.has(groupID)
}

async function toggleGroupEnabled(
  group: ModelRouteScheduleGroupDto,
  next: boolean,
): Promise<boolean> {
  // The switch is disabled while a toggle is pending, so this guard is nearly
  // unreachable in the modal flow; report failure instead of a false success.
  if (groupTogglePending(group.group_id)) return false
  optimisticEnabled.value = new Map(optimisticEnabled.value).set(group.group_id, next)
  togglingGroupIDs.value = new Set(togglingGroupIDs.value).add(group.group_id)
  try {
    if (props.detail) {
      preserveDraftRevisions.value = new Set([
        ...preserveDraftRevisions.value,
        props.detail.snapshot_revision,
      ])
      preserveDraftSnapshots.value = new Map(preserveDraftSnapshots.value).set(
        props.detail.snapshot_revision,
        cloneDrafts(draftMap),
      )
    }
    const settings = await updateGroupSettings(client, group.group_id, { enabled: next })
    cacheGroupSettings(queryClient, group.group_id, settings)
    await invalidateGroupSettingsDependents(queryClient, group.group_id)
    return true
  } catch {
    const optimistic = new Map(optimisticEnabled.value)
    optimistic.delete(group.group_id)
    optimisticEnabled.value = optimistic
    toast.show({ message: text('toggleFailed'), tone: 'danger' })
  } finally {
    await nextTick()
    if (props.detail) {
      const revisions = new Set(preserveDraftRevisions.value)
      revisions.delete(props.detail.snapshot_revision)
      const snapshots = new Map(preserveDraftSnapshots.value)
      snapshots.delete(props.detail.snapshot_revision)
      preserveDraftSnapshots.value = snapshots
      preserveDraftRevisions.value = revisions
    }
    const optimistic = new Map(optimisticEnabled.value)
    optimistic.delete(group.group_id)
    optimisticEnabled.value = optimistic
    const pending = new Set(togglingGroupIDs.value)
    pending.delete(group.group_id)
    togglingGroupIDs.value = pending
  }
  return false
}

async function applyProbeEnabled(changes: Map<number, boolean>): Promise<boolean> {
  let attempted = false
  let allSucceeded = true
  for (const [groupID, next] of changes) {
    const group = props.detail?.groups.find((g) => g.group_id === groupID)
    if (!group) continue
    attempted = true
    const ok = await toggleGroupEnabled(group, next)
    if (!ok) allSucceeded = false
  }
  return attempted && allSucceeded
}

function isFirstGroupRow(index: number): boolean {
  return index === 0 || rows.value[index - 1]?.group.group_id !== rows.value[index]?.group.group_id
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

function groupDraftKey(groupID: number): string {
  return String(groupID)
}

function groupReasoningValue(group: ModelRouteScheduleGroupDto): ReasoningEffortDto | '' {
  const key = groupDraftKey(group.group_id)
  return hasOwn(groupReasoningDrafts, key)
    ? (groupReasoningDrafts[key] ?? '')
    : (group.reasoning_effort_default ?? '')
}

function entryReasoningValue(
  groupID: number,
  entry: ModelRouteScheduleReasoningEntryDto,
): ReasoningEffortDto | '' {
  const key = draftKey(groupID, entry.entry_id)
  return hasOwn(entryReasoningDrafts, key)
    ? (entryReasoningDrafts[key] ?? '')
    : (entry.reasoning.configured ?? '')
}

function reasoningOptions(
  levels: readonly ReasoningEffortDto[],
  current: ReasoningEffortDto | null,
): Array<{ value: string; label: string }> {
  const options = [{ value: '', label: text('inherit') }]
  options.push(...levels.map((value) => ({ value, label: value })))
  if (current !== null && !levels.includes(current)) {
    options.push({
      value: current,
      label: t('monitor.schedule.detail.unsupportedEffortValue', { value: current }),
    })
  }
  return options
}

function groupReasoningLevels(group: ModelRouteScheduleGroupDto): ReasoningEffortDto[] {
  const inherited = group.reasoning_entries.filter(
    (entry) => entryReasoningValue(group.group_id, entry) === '',
  )
  return reasoningEffortValues.filter((level) =>
    inherited.every(
      ({ reasoning }) =>
        reasoning.capability.supported && reasoning.capability.levels.includes(level),
    ),
  )
}

const reasoningPreviews = computed(() =>
  (props.detail?.groups ?? []).flatMap((group) => {
    const groupDrafts = Object.keys(groupReasoningDrafts)
    const entryDrafts = Object.keys(entryReasoningDrafts)
    const fieldDrafts = Object.keys(draftMap)
    const touched =
      groupDrafts.includes(groupDraftKey(group.group_id)) ||
      group.entries.some(
        (entry) =>
          entryDrafts.includes(draftKey(group.group_id, entry.entry_id)) ||
          fieldDrafts.includes(draftKey(group.group_id, entry.entry_id)),
      )
    if (!touched) return []
    return [
      {
        group,
        entries: group.reasoning_entries.map((entry) => {
          const override = entryReasoningValue(group.group_id, entry)
          const effective = override || groupReasoningValue(group)
          const unsupported =
            effective !== '' &&
            (!entry.reasoning.capability.supported ||
              !entry.reasoning.capability.levels.includes(effective))
          return { entry, effective, override, unsupported }
        }),
      },
    ]
  }),
)

function groupReasoningOptions(group: ModelRouteScheduleGroupDto) {
  const value = groupReasoningValue(group)
  return reasoningOptions(groupReasoningLevels(group), value === '' ? null : value)
}

function entryReasoningOptions(groupID: number, entry: ModelRouteScheduleEntryDto) {
  const value = entryReasoningValue(groupID, entry)
  return reasoningOptions(
    entry.reasoning.capability.supported ? entry.reasoning.capability.levels : [],
    value === '' ? null : value,
  )
}

function setGroupReasoning(group: ModelRouteScheduleGroupDto, value: string): void {
  if (value !== '' && !isReasoningEffort(value)) return
  const next = value === '' ? null : value
  const key = groupDraftKey(group.group_id)
  if (next === group.reasoning_effort_default) delete groupReasoningDrafts[key]
  else groupReasoningDrafts[key] = next
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

function reasoningSourceLabel(source: ModelRouteScheduleReasoningSource): string {
  if (source === 'entry') return text('sourceEntry')
  if (source === 'group') return text('sourceGroup')
  if (source === 'client') return text('sourceClient')
  return text('sourceProviderDefault')
}

function reasoningCapabilityLabel(entry: ModelRouteScheduleEntryDto): string {
  if (!entry.reasoning.capability.known) return text('capabilityUnknown')
  return entry.reasoning.capability.supported ? text('supported') : text('unsupported')
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
  return value === null || value === undefined ? '' : String(value)
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

function clearField(
  groupID: number,
  entry: ModelRouteScheduleEntryDto,
  field: EditableField,
): void {
  rawInputs[fieldKey(groupID, entry.entry_id, field)] = ''
  delete invalidInputs[fieldKey(groupID, entry.entry_id, field)]
  setDraftValue(groupID, entry, field, null)
}

function resetDrafts(): void {
  for (const key of Object.keys(draftMap)) delete draftMap[key]
  for (const key of Object.keys(entryReasoningDrafts)) delete entryReasoningDrafts[key]
  for (const key of Object.keys(groupReasoningDrafts)) delete groupReasoningDrafts[key]
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
  accessKeyID: number,
]

watch(
  (): DetailWatchKey => {
    const detail = props.detail
    return detail
      ? [detail.snapshot_revision, detail.external_model, detail.protocol, detail.access_key.id]
      : [0, '', '', 0]
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

function groupUpdates(): ModelRouteScheduleGroupPatchUpdate[] {
  return (props.detail?.groups ?? []).flatMap((group) => {
    const key = groupDraftKey(group.group_id)
    return hasOwn(groupReasoningDrafts, key)
      ? [
          {
            group_id: group.group_id,
            reasoning_effort_default: groupReasoningDrafts[key] ?? null,
          },
        ]
      : []
  })
}

async function save(): Promise<void> {
  if (!props.detail || !dirty.value || invalid.value) return
  const body = {
    snapshot_revision: props.detail.snapshot_revision,
    protocol: props.detail.protocol,
    external_model: props.detail.external_model ?? '',
    access_key_id: props.detail.access_key.id,
    operation: props.detail.operation,
    group_updates: groupUpdates(),
    updates: updates(),
  }
  if (body.updates.length === 0 && body.group_updates.length === 0) return
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
  return previewShares.value.get(rowKey(groupID, entry.entry_id)) ?? entry.configured_share
}

function isDraftShare(groupID: number, entry: ModelRouteScheduleEntryDto): boolean {
  return previewShares.value.has(rowKey(groupID, entry.entry_id))
}

function breakerRecoveryLabel(entry: ModelRouteScheduleEntryDto): string {
  const breaker = entry.circuit_breaker.effective
  const threshold = breaker.blacklist_threshold ?? '-'
  const cooldown = breaker.cooldown_seconds ?? '-'
  const recovery: string[] = []
  if (entry.runtime.cooldown_until_ms !== null) {
    recovery.push(
      `${text('cooldownUntil')}: ${formatLocalInstant(entry.runtime.cooldown_until_ms, props.locale)}`,
    )
  }
  if (entry.runtime.blacklist_release_at_ms !== null) {
    recovery.push(
      `${text('scheduledReleaseAt')}: ${formatLocalInstant(entry.runtime.blacklist_release_at_ms, props.locale)}`,
    )
  }
  return `${threshold}/${cooldown}s${recovery.length > 0 ? ` · ${recovery.join(' · ')}` : ''}`
}

defineExpose({ applyProbeEnabled })
</script>

<template>
  <section class="schedule-detail" aria-labelledby="schedule-detail-title">
    <header class="schedule-detail__header">
      <div>
        <p class="schedule-detail__eyebrow">{{ detail?.external_model ?? text('title') }}</p>
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
        v-if="refreshing"
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
            <span role="columnheader">{{ text('group') }}</span>
            <span role="columnheader">{{ text('upstreamModel') }}</span>
            <span role="columnheader">{{ text('reasoning') }}</span>
            <span role="columnheader">{{ text('weight') }}</span>
            <span role="columnheader">{{ text('priority') }}</span>
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
            <div class="schedule-cell schedule-cell--group" role="cell">
              <RouterLink
                class="schedule-cell__group-link"
                :to="groupDetailLocation(group.group_id)"
                @click.stop
              >
                <strong>{{ group.group_name }}</strong>
              </RouterLink>
              <div v-if="isFirstGroupRow(index)" class="schedule-cell__group-controls">
                <AppSwitch
                  :model-value="groupEnabled(group)"
                  :disabled="groupTogglePending(group.group_id)"
                  :label="`${text('toggleEnabled')} ${group.group_name}`"
                  @click.stop
                  @update:model-value="toggleGroupEnabled(group, $event)"
                />
                <span class="schedule-cell__group-state">
                  {{ groupEnabled(group) ? text('enabled') : text('disabled') }}
                </span>
              </div>
              <small v-if="isFirstGroupRow(index)" class="schedule-cell__stats">
                {{ text('calls24h') }}: {{ formatCount(group.request_count) }} ·
                {{ text('successRate24h') }}: {{ formatRate(group.success_rate) }}
              </small>
            </div>
            <div class="schedule-cell" role="cell">
              <strong>{{ entry.alias || entry.model_id }}</strong>
              <small v-if="entry.alias">{{ entry.model_id }}</small>
            </div>
            <div class="schedule-cell schedule-cell--reasoning" role="cell">
              <label v-if="isFirstGroupRow(index)" class="schedule-reasoning-control">
                <span>{{ text('groupDefault') }}</span>
                <AppSelect
                  :model-value="groupReasoningValue(group)"
                  :options="groupReasoningOptions(group)"
                  :label="`${text('groupDefault')} ${group.group_name}`"
                  :disabled="
                    pending ||
                    (groupReasoningLevels(group).length === 0 &&
                      group.reasoning_effort_default === null)
                  "
                  size="compact"
                  @click.stop
                  @update:model-value="setGroupReasoning(group, $event)"
                />
              </label>
              <small v-if="isFirstGroupRow(index)">{{
                t('monitor.schedule.detail.groupReasoningScope', {
                  count: group.reasoning_entries.length,
                })
              }}</small>
              <label class="schedule-reasoning-control">
                <span>{{ text('entryOverride') }}</span>
                <AppSelect
                  :model-value="entryReasoningValue(group.group_id, entry)"
                  :options="entryReasoningOptions(group.group_id, entry)"
                  :label="`${text('entryOverride')} ${entry.model_id}`"
                  :disabled="
                    pending ||
                    entry.entry_id.startsWith('derived:') ||
                    (!entry.reasoning.capability.supported && entry.reasoning.configured === null)
                  "
                  size="compact"
                  @click.stop
                  @update:model-value="setEntryReasoning(group.group_id, entry, $event)"
                />
              </label>
              <dl class="schedule-reasoning-meta">
                <div>
                  <dt>{{ text('effective') }}</dt>
                  <dd>{{ entry.reasoning.effective ?? text('inherit') }}</dd>
                </div>
                <div>
                  <dt>{{ text('source') }}</dt>
                  <dd>{{ reasoningSourceLabel(entry.reasoning.source) }}</dd>
                </div>
                <div>
                  <dt>{{ text('capability') }}</dt>
                  <dd
                    :class="{
                      'schedule-reasoning-meta__unsupported': !entry.reasoning.capability.supported,
                    }"
                  >
                    {{ reasoningCapabilityLabel(entry) }}
                  </dd>
                </div>
              </dl>
              <small class="schedule-reasoning-reason">{{
                entry.reasoning.capability.reason
              }}</small>
            </div>
            <div class="schedule-cell schedule-cell--input" role="cell">
              <label class="sr-only" :for="`weight-${index}`">{{ text('weight') }}</label>
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
              <button
                type="button"
                class="schedule-cell__clear"
                :disabled="entry.entry_id.startsWith('derived:')"
                @click.stop="clearField(group.group_id, entry, 'weight')"
              >
                {{ text('clear') }}
              </button>
            </div>
            <div class="schedule-cell schedule-cell--input" role="cell">
              <label class="sr-only" :for="`priority-${index}`">{{ text('priority') }}</label>
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
              <button
                type="button"
                class="schedule-cell__clear"
                :disabled="entry.entry_id.startsWith('derived:')"
                @click.stop="clearField(group.group_id, entry, 'priority')"
              >
                {{ text('clear') }}
              </button>
            </div>
            <div class="schedule-cell schedule-cell--share" role="cell">
              {{ (shareValue(group.group_id, entry) * 100).toFixed(1) }}%
              <small v-if="isDraftShare(group.group_id, entry)">{{ text('draftPreview') }}</small>
            </div>
            <div class="schedule-cell" role="cell">
              <strong :class="runtimeTone(entry)">{{ runtimeLabel(entry) }}</strong>
              <small v-if="entry.runtime.failure_count"
                >{{ entry.runtime.failure_count }} {{ text('failures') }}</small
              >
            </div>
            <div class="schedule-cell schedule-cell--breaker" role="cell">
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

      <div
        v-for="preview in reasoningPreviews"
        :key="preview.group.group_id"
        class="schedule-reasoning-preview"
        aria-live="polite"
      >
        <p>
          {{
            t('monitor.schedule.detail.groupReasoningPreview', { group: preview.group.group_name })
          }}
        </p>
        <ul>
          <li v-for="item in preview.entries" :key="item.entry.entry_id">
            {{ item.entry.model_id }}: {{ item.effective || text('inherit') }} ·
            {{ item.override ? text('entryOverride') : text('groupDefault') }}
            <strong v-if="item.unsupported" class="schedule-reasoning-meta__unsupported">
              — {{ t('monitor.schedule.detail.reasoningSaveBlocked') }}</strong
            >
          </li>
        </ul>
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
.schedule-detail__eyebrow {
  margin: 0 0 3px;
  overflow: hidden;
  color: var(--color-action);
  font-family: var(--font-mono);
  font-size: var(--text-meta);
  text-overflow: ellipsis;
  white-space: nowrap;
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
  overflow-x: auto;
  border: 1px solid var(--color-border-subtle);
}
.schedule-table {
  min-width: 1260px;
}
.schedule-row {
  display: grid;
  grid-template-columns:
    minmax(150px, 1.1fr) minmax(145px, 1fr) minmax(250px, 1.7fr) 90px 85px 90px
    minmax(105px, 0.8fr) minmax(180px, 1.25fr);
  min-height: 58px;
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
.schedule-cell strong,
.schedule-cell small {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.schedule-cell--reasoning {
  display: grid;
  gap: 6px;
}
.schedule-reasoning-control {
  display: grid;
  grid-template-columns: minmax(72px, auto) minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  color: var(--color-text-faint);
  font-size: 10px;
}
.schedule-reasoning-control :deep(.app-select__trigger) {
  width: 100%;
}
.schedule-reasoning-preview {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 12px;
}
.schedule-reasoning-preview ul {
  padding-inline-start: 20px;
}
.schedule-reasoning-meta {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  gap: 3px 10px;
  margin: 0;
}
.schedule-reasoning-meta div {
  display: flex;
  min-width: 0;
  gap: 3px;
}
.schedule-reasoning-meta dt {
  color: var(--color-text-faint);
}
.schedule-reasoning-meta dd {
  margin: 0;
  color: var(--color-text);
}
.schedule-reasoning-meta__unsupported {
  color: var(--color-danger) !important;
  font-weight: 650;
}
.schedule-reasoning-reason {
  overflow: visible !important;
  color: var(--color-text-faint);
  text-overflow: clip !important;
  white-space: normal !important;
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
.schedule-cell__group-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
}
.schedule-cell__group-state {
  color: var(--color-text-faint);
  font-size: 10px;
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
  grid-template-columns: 62px auto;
  align-items: center;
  gap: 3px;
}
.schedule-cell--input :deep(.app-text-input__input) {
  width: 62px;
  font-family: var(--font-mono);
}
.schedule-cell__clear {
  border: 0;
  background: transparent;
  color: var(--color-text-faint);
  padding: 2px;
  font-size: 10px;
  cursor: pointer;
}
.schedule-cell__clear:hover {
  color: var(--color-action);
}
.schedule-cell--share {
  color: var(--color-action);
  font-family: var(--font-mono);
  font-weight: 700;
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
}
</style>
