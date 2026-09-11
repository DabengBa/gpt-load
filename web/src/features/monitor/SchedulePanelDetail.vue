<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query'
import { computed, nextTick, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { useApiClient } from '@/api/client-context'
import { useToast } from '@/app/toast'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@/app/resources/invalidation'
import { groupDetailLocation } from '@/app/route-locations'
import {
  cacheGroupSettings,
  invalidateGroupSettingsDependents,
  updateGroupSettings,
} from '@/app/resources/groups'
import {
  isModelRouteScheduleRevisionConflict,
  recoverModelRouteScheduleEntry,
  updateModelRouteSchedule,
  type ModelRouteScheduleDetailDto,
  type ModelRouteScheduleEntryDto,
  type ModelRouteScheduleGroupDto,
  type ModelRouteSchedulePatchUpdate,
} from '@/app/resources/model-route-schedule'
import type { ModelProbeTargetDto } from '@/app/resources/model-probe'
import AppButton from '@/components/ui/AppButton.vue'
import AppSwitch from '@/components/ui/AppSwitch.vue'
import AppTextInput from '@/components/ui/AppTextInput.vue'
import InlineFeedback from '@/components/ui/InlineFeedback.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import StickySaveBar from '@/components/ui/StickySaveBar.vue'
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
    drafts: () => ({}),
  },
)
const emit = defineEmits<{
  refresh: []
  saved: [snapshotRevision: number]
  recovered: [groupId: number, entryId: string]
  probe: [groupId: number, modelId: string]
  'probe-all': [targets: ModelProbeTargetDto[]]
  'draft-change': [drafts: ScheduleDrafts]
  'row-change': [row: string | undefined]
}>()

const queryClient = useQueryClient()
const client = useApiClient()
const { t } = useI18n()
const toast = useToast()
const draftMap = reactive<Record<string, Draft>>({})
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
// Batch scope is exactly what is on screen: the same mode-filtered rows the
// table renders, deduplicated to (group, model) targets.
const probeTargets = computed<ModelProbeTargetDto[]>(() => {
  const targets: ModelProbeTargetDto[] = []
  const seen = new Set<string>()
  for (const { group, entry } of rows.value) {
    const key = `${group.group_id}:${entry.model_id}`
    if (seen.has(key)) continue
    seen.add(key)
    targets.push({ group_id: group.group_id, model: entry.model_id })
  }
  return targets
})

function probeEntry(groupId: number, modelId: string): void {
  emit('probe', groupId, modelId)
}

function probeVisibleRows(): void {
  if (probeTargets.value.length === 0) return
  emit('probe-all', probeTargets.value)
}

const dirty = computed(() => Object.keys(draftMap).length > 0)
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

async function toggleGroupEnabled(group: ModelRouteScheduleGroupDto, next: boolean): Promise<void> {
  if (groupTogglePending(group.group_id)) return
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
  for (const { group, entry } of rows.value) {
    const draft = draftMap[draftKey(group.group_id, entry.entry_id)]
    if (!draft || entry.entry_id.startsWith('derived:')) continue
    const update: ModelRouteSchedulePatchUpdate = {
      group_id: group.group_id,
      entry_id: entry.entry_id,
    }
    if (Object.prototype.hasOwnProperty.call(draft, 'weight')) update.weight = draft.weight
    if (Object.prototype.hasOwnProperty.call(draft, 'priority')) update.priority = draft.priority
    if (Object.keys(update).length > 2) result.push(update)
  }
  return result
}

async function save(): Promise<void> {
  if (!props.detail || !dirty.value || invalid.value) return
  const body = {
    snapshot_revision: props.detail.snapshot_revision,
    protocol: props.detail.protocol,
    external_model: props.detail.external_model ?? '',
    access_key_id: props.detail.access_key.id,
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

async function recover(groupID: number, entryID: string): Promise<void> {
  const key = rowKey(groupID, entryID)
  recovering.value = key
  saveError.value = ''
  try {
    await recoverModelRouteScheduleEntry(client, { group_id: groupID, entry_id: entryID })
    await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.recover)
    emit('recovered', groupID, entryID)
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
  const recovery =
    entry.runtime.cooldown_until_ms === null
      ? ''
      : ` · ${formatLocalInstant(entry.runtime.cooldown_until_ms, props.locale)}`
  return `${threshold}/${cooldown}s${recovery}`
}
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
        {{ t('monitor.modelProbe.batch', { count: probeTargets.length }) }}
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
      <div v-else class="schedule-table-wrap">
        <div class="schedule-table" role="table" :aria-label="text('title')">
          <div class="schedule-row schedule-row--header" role="row">
            <span role="columnheader">{{ text('group') }}</span>
            <span role="columnheader">{{ text('upstreamModel') }}</span>
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
                @click.stop="recover(group.group_id, entry.entry_id)"
              >
                {{ text('recover') }}
              </AppButton>
              <AppButton
                variant="secondary"
                size="compact"
                :disabled="pending || entry.entry_id.startsWith('derived:')"
                @click.stop="probeEntry(group.group_id, entry.model_id)"
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
  min-width: 980px;
}
.schedule-row {
  display: grid;
  grid-template-columns:
    minmax(150px, 1.25fr) minmax(145px, 1.15fr) 90px 85px 90px minmax(105px, 0.9fr)
    minmax(180px, 1.35fr);
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
