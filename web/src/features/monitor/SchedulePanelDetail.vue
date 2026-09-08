<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query'
import { computed, reactive, ref, watch } from 'vue'

import { useApiClient } from '@/api/client-context'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@/app/resources/invalidation'
import {
  isModelRouteScheduleRevisionConflict,
  updateModelRouteSchedule,
  recoverModelRouteScheduleEntry,
  type ModelRouteScheduleDetailDto,
  type ModelRouteScheduleEntryDto,
  type ModelRouteScheduleGroupDto,
  type ModelRouteSchedulePatchUpdate,
} from '@/app/resources/model-route-schedule'
import AppButton from '@/components/ui/AppButton.vue'
import AppTextInput from '@/components/ui/AppTextInput.vue'
import InlineFeedback from '@/components/ui/InlineFeedback.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import StickySaveBar from '@/components/ui/StickySaveBar.vue'
import { formatLocalInstant } from '@/lib/format'

export interface SchedulePanelDetailLabels {
  title: string
  loading: string
  failed: string
  refresh: string
  observed: string
  stale: string
  routeUnavailable: string
  groupWeight: string
  channel: string
  entryId: string
  weight: string
  priority: string
  fallback: string
  share: string
  reason: string
  runtime: string
  available: string
  cooldown: string
  blacklisted: string
  failures: string
  recover: string
  credentials: string
  breaker: string
  threshold: string
  cooldownSeconds: string
  effective: string
  configured: string
  inherited: string
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
  routeStrategyLabels?: Partial<Record<string, string>>
  unknownRouteStrategy?: string
  sourceLabels?: Partial<Record<string, string>>
  unknownBreakerSource?: string
  nativeRoute?: string
  draftPreview?: string
  observedShare?: string
}

type EditableField = 'weight' | 'priority' | 'blacklist_threshold' | 'cooldown_seconds'
type Draft = Partial<Record<EditableField, number | null>> & { breaker_clear?: boolean }

type RecoverKey = string

const props = withDefaults(
  defineProps<{
    detail?: ModelRouteScheduleDetailDto
    loading?: boolean
    error?: string
    refreshing?: boolean
    labels?: Partial<SchedulePanelDetailLabels>
    locale?: string
  }>(),
  {
    detail: undefined,
    loading: false,
    error: '',
    refreshing: false,
    labels: () => ({}),
    locale: 'en-US',
  },
)
const emit = defineEmits<{
  refresh: []
  saved: [snapshotRevision: number]
  recovered: [groupId: number, entryId: string]
}>()

const queryClient = useQueryClient()
const client = useApiClient()
const drafts = reactive<Record<string, Draft>>({})
const rawInputs = reactive<Record<string, string>>({})
const invalidInputs = reactive<Record<string, boolean>>({})
const pending = ref(false)
const saveStatus = ref<'idle' | 'saved' | 'error'>('idle')
const saveError = ref('')
const recovering = ref<RecoverKey>('')

const text = (key: keyof SchedulePanelDetailLabels): string => {
  const value = props.labels[key]
  return typeof value === 'string' ? value : ''
}

const entries = computed(() => props.detail?.groups.flatMap((group) => group.entries) ?? [])
const dirty = computed(() => Object.keys(drafts).length > 0)
const invalid = computed(() => Object.values(invalidInputs).some(Boolean))
const hasDetail = computed(() => props.detail !== undefined)
const hasWeightDraft = computed(() =>
  Object.values(drafts).some((draft) => Object.prototype.hasOwnProperty.call(draft, 'weight')),
)

function effectiveGroupWeight(group: ModelRouteScheduleGroupDto): number {
  return group.group_weight ?? 50
}

const previewShares = computed(() => {
  const result = new Map<string, number>()
  if (!props.detail || !hasWeightDraft.value) return result

  const candidates = props.detail.groups.flatMap((group) =>
    group.entries
      .filter(
        (entry) =>
          entry.included &&
          !entry.fallback &&
          (entry.routable || entry.reason_code === 'entry_weight_zero'),
      )
      .map((entry) => {
        const weight = drafts[draftKey(entry)]?.weight ?? entry.weight
        const credentialWeight = entry.credentials.reduce((total, credential) => {
          if (!credential.available) return total
          const configured = credential.weight_manual
          const automatic = credential.weight_auto === 0 ? 50 : credential.weight_auto
          return total + (configured ?? automatic)
        }, 0)
        const groupWeight = effectiveGroupWeight(group)
        return {
          group,
          entry,
          weight,
          mass: groupWeight * weight * credentialWeight,
        }
      }),
  )
  const activeTier = candidates.reduce(
    (tier, candidate) =>
      candidate.mass > 0 && (tier === 0 || candidate.entry.priority < tier)
        ? candidate.entry.priority
        : tier,
    0,
  )
  if (activeTier === 0) {
    for (const candidate of candidates) {
      result.set(entryKey(candidate.group.group_id, candidate.entry.entry_id), 0)
    }
    return result
  }
  const activeTotal = candidates
    .filter((candidate) => candidate.entry.priority === activeTier)
    .reduce((total, candidate) => total + candidate.mass, 0)
  if (activeTotal <= 0) return result

  for (const candidate of candidates) {
    if (candidate.entry.priority !== activeTier) {
      result.set(entryKey(candidate.group.group_id, candidate.entry.entry_id), 0)
      continue
    }
    result.set(
      entryKey(candidate.group.group_id, candidate.entry.entry_id),
      Math.max(0, candidate.mass) / activeTotal,
    )
  }
  return result
})
const observedLabel = computed(() => {
  const observed = props.detail?.observed_at_ms
  return observed === undefined
    ? ''
    : `${text('observed')} ${formatLocalInstant(observed, props.locale)}`
})

function entryKey(groupID: number, entryID: string): string {
  return `${groupID}\u0000${entryID}`
}

function draftKey(entry: ModelRouteScheduleEntryDto): string {
  const group = props.detail?.groups.find(({ entries: groupEntries }) =>
    groupEntries.includes(entry),
  )
  return entryKey(group?.group_id ?? 0, entry.entry_id)
}

function fieldKey(entry: ModelRouteScheduleEntryDto, field: EditableField): string {
  return `${draftKey(entry)}\u0000${field}`
}

function configuredValue(entry: ModelRouteScheduleEntryDto, field: EditableField): number | null {
  if (field === 'weight') return entry.weight_manual
  if (field === 'priority') return entry.priority === 1 ? null : entry.priority
  return entry.circuit_breaker.configured[field]
}

function effectiveValue(entry: ModelRouteScheduleEntryDto, field: EditableField): number {
  if (field === 'weight') return entry.weight
  if (field === 'priority') return entry.priority
  return entry.circuit_breaker.effective[field] ?? 0
}

function inputValue(entry: ModelRouteScheduleEntryDto, field: EditableField): string {
  const key = fieldKey(entry, field)
  return rawInputs[key] ?? ''
}

function placeholder(entry: ModelRouteScheduleEntryDto, field: EditableField): string {
  const configured = configuredValue(entry, field)
  return configured === null ? String(effectiveValue(entry, field)) : String(configured)
}

function fieldLabel(field: EditableField): string {
  if (field === 'weight') return text('weight')
  if (field === 'priority') return text('priority')
  if (field === 'blacklist_threshold') return text('threshold')
  return text('cooldownSeconds')
}

function isValidValue(field: EditableField, value: number): boolean {
  if (!Number.isSafeInteger(value)) return false
  if (field === 'weight') return value >= 0 && value <= 100
  if (field === 'priority' || field === 'blacklist_threshold') return value >= 1
  return value >= 0
}

function setDraftValue(
  entry: ModelRouteScheduleEntryDto,
  field: EditableField,
  value: number | null,
) {
  const key = draftKey(entry)
  const baseline = configuredValue(entry, field)
  const effective = effectiveValue(entry, field)
  const sameAsServer = value !== null && value === (baseline ?? effective)
  const draft = drafts[key] ?? {}
  if (field === 'blacklist_threshold' || field === 'cooldown_seconds') {
    delete draft.breaker_clear
  }
  if (value === null ? baseline === null : sameAsServer) {
    delete draft[field]
  } else {
    draft[field] = value
  }
  if (Object.keys(draft).length === 0) delete drafts[key]
  else drafts[key] = draft
}

function setInput(entry: ModelRouteScheduleEntryDto, field: EditableField, value: string): void {
  const key = fieldKey(entry, field)
  rawInputs[key] = value
  delete invalidInputs[key]
  if (value.trim() === '') {
    setDraftValue(entry, field, null)
    const draft = drafts[draftKey(entry)]
    if (draft) delete draft[field]
    if (draft && Object.keys(draft).length === 0) delete drafts[draftKey(entry)]
    return
  }
  const parsed = Number(value)
  if (!/^\d+$/.test(value.trim()) || !isValidValue(field, parsed)) {
    invalidInputs[key] = true
    return
  }
  setDraftValue(entry, field, parsed)
}

function clearField(entry: ModelRouteScheduleEntryDto, field: EditableField): void {
  rawInputs[fieldKey(entry, field)] = ''
  delete invalidInputs[fieldKey(entry, field)]
  setDraftValue(entry, field, null)
}

function clearBreaker(entry: ModelRouteScheduleEntryDto): void {
  if (
    entry.circuit_breaker.configured.blacklist_threshold === null &&
    entry.circuit_breaker.configured.cooldown_seconds === null
  ) {
    return
  }
  const key = draftKey(entry)
  drafts[key] = { ...(drafts[key] ?? {}), breaker_clear: true }
  rawInputs[fieldKey(entry, 'blacklist_threshold')] = ''
  rawInputs[fieldKey(entry, 'cooldown_seconds')] = ''
}

function resetDrafts(): void {
  for (const key of Object.keys(drafts)) delete drafts[key]
  for (const key of Object.keys(rawInputs)) delete rawInputs[key]
  for (const key of Object.keys(invalidInputs)) delete invalidInputs[key]
  for (const entry of entries.value) {
    for (const field of [
      'weight',
      'priority',
      'blacklist_threshold',
      'cooldown_seconds',
    ] as const) {
      rawInputs[fieldKey(entry, field)] = ''
    }
  }
  saveStatus.value = 'idle'
  saveError.value = ''
}

watch(
  () => {
    const detail = props.detail
    return detail
      ? [
          detail.snapshot_revision,
          detail.external_model,
          detail.protocol,
          detail.operation,
          detail.access_key.id,
        ]
      : [0, '', '', '', 0]
  },
  resetDrafts,
  { immediate: true },
)

function updates(): ModelRouteSchedulePatchUpdate[] {
  const result: ModelRouteSchedulePatchUpdate[] = []
  for (const group of props.detail?.groups ?? []) {
    for (const entry of group.entries) {
      const key = entryKey(group.group_id, entry.entry_id)
      const draft = drafts[key]
      if (!draft || entry.entry_id.startsWith('derived:')) continue
      const update: ModelRouteSchedulePatchUpdate = {
        group_id: group.group_id,
        entry_id: entry.entry_id,
      }
      for (const field of ['weight', 'priority'] as const) {
        if (field in draft) update[field] = draft[field]
      }
      if (draft.breaker_clear) {
        update.circuit_breaker = null
      } else {
        const breaker: Record<string, number | null> = {}
        for (const field of ['blacklist_threshold', 'cooldown_seconds'] as const) {
          if (field in draft) breaker[field] = draft[field] ?? null
        }
        if (Object.keys(breaker).length > 0) update.circuit_breaker = breaker
      }
      if (Object.keys(update).length > 2 || update.circuit_breaker !== undefined)
        result.push(update)
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
}

async function recover(groupID: number, entryID: string): Promise<void> {
  const key = entryKey(groupID, entryID)
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

function reasonLabel(entry: ModelRouteScheduleEntryDto): string {
  const code = entry.reason_code
  return code ? (props.labels.reasonLabels?.[code] ?? text('unknownReason')) : text('unknownReason')
}

function routeStrategyLabel(strategy: string): string {
  return props.labels.routeStrategyLabels?.[strategy] ?? text('unknownRouteStrategy')
}

function breakerSourceLabel(source: unknown): string {
  if (typeof source !== 'string' || source.trim() === '') return text('unknownBreakerSource')
  return props.labels.sourceLabels?.[source] ?? text('unknownBreakerSource')
}

function shareKey(groupID: number, entryID: string): string {
  return entryKey(groupID, entryID)
}

function shareValue(groupID: number, entry: ModelRouteScheduleEntryDto): number {
  return previewShares.value.get(shareKey(groupID, entry.entry_id)) ?? entry.effective_share
}

function isDraftShare(groupID: number, entry: ModelRouteScheduleEntryDto): boolean {
  return previewShares.value.has(shareKey(groupID, entry.entry_id))
}

function credentialSummary(entry: ModelRouteScheduleEntryDto): string {
  const available = entry.credentials.filter((credential) => credential.available).length
  return `${available}/${entry.credentials.length}`
}

function cooldownLabel(until: number | null): string {
  return until === null ? '' : formatLocalInstant(until, props.locale)
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
        <span v-if="detail?.route_requirement === 'native'">{{ text('nativeRoute') }}</span>
        <span v-if="detail?.route_strategy">{{ routeStrategyLabel(detail.route_strategy) }}</span>
      </div>
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

      <div v-if="detail.groups.length === 0" class="schedule-detail__empty" role="status">
        {{ text('noEntries') }}
      </div>
      <div v-else class="schedule-detail__groups">
        <article v-for="group in detail.groups" :key="group.group_id" class="schedule-group">
          <header class="schedule-group__header">
            <div>
              <h3>{{ group.group_name }}</h3>
              <p>
                {{ text('channel') }}: <code>{{ group.channel_id }}</code>
              </p>
            </div>
            <span class="schedule-group__weight">
              {{ text('groupWeight') }} <strong>{{ effectiveGroupWeight(group) }}</strong>
            </span>
          </header>

          <div class="schedule-entries" role="list">
            <article
              v-for="entry in group.entries"
              :key="entry.entry_id"
              class="schedule-entry"
              :class="{ 'schedule-entry--excluded': !entry.included || !entry.routable }"
              role="listitem"
            >
              <div class="schedule-entry__identity">
                <div class="schedule-entry__name">
                  <strong>{{ entry.alias || entry.model_id }}</strong>
                  <span v-if="entry.alias" class="schedule-entry__model">{{ entry.model_id }}</span>
                </div>
                <code>{{ text('entryId') }} {{ entry.entry_id }}</code>
                <span v-if="entry.entry_id.startsWith('derived:')" class="schedule-entry__readonly">
                  {{ text('derivedReadOnly') }}
                </span>
              </div>

              <div class="schedule-entry__editors">
                <label>
                  <span>{{ text('weight') }}</span>
                  <AppTextInput
                    :model-value="inputValue(entry, 'weight')"
                    type="number"
                    :label="`${fieldLabel('weight')} ${entry.model_id}`"
                    :placeholder="placeholder(entry, 'weight')"
                    :invalid="invalidInputs[fieldKey(entry, 'weight')]"
                    :disabled="entry.entry_id.startsWith('derived:')"
                    size="compact"
                    @update:model-value="setInput(entry, 'weight', $event)"
                  />
                  <AppButton
                    variant="link"
                    size="inline"
                    :disabled="entry.entry_id.startsWith('derived:')"
                    @click="clearField(entry, 'weight')"
                    >{{ text('clear') }}</AppButton
                  >
                  <small
                    v-if="invalidInputs[fieldKey(entry, 'weight')]"
                    class="schedule-entry__validation"
                    role="alert"
                  >
                    {{ text('invalidValue') }}
                  </small>
                </label>
                <label>
                  <span>{{ text('priority') }}</span>
                  <AppTextInput
                    :model-value="inputValue(entry, 'priority')"
                    type="number"
                    :label="`${fieldLabel('priority')} ${entry.model_id}`"
                    :placeholder="placeholder(entry, 'priority')"
                    :invalid="invalidInputs[fieldKey(entry, 'priority')]"
                    :disabled="entry.entry_id.startsWith('derived:')"
                    size="compact"
                    @update:model-value="setInput(entry, 'priority', $event)"
                  />
                  <AppButton
                    variant="link"
                    size="inline"
                    :disabled="entry.entry_id.startsWith('derived:')"
                    @click="clearField(entry, 'priority')"
                    >{{ text('clear') }}</AppButton
                  >
                  <small
                    v-if="invalidInputs[fieldKey(entry, 'priority')]"
                    class="schedule-entry__validation"
                    role="alert"
                  >
                    {{ text('invalidValue') }}
                  </small>
                </label>
                <label>
                  <span>{{ text('threshold') }}</span>
                  <AppTextInput
                    :model-value="inputValue(entry, 'blacklist_threshold')"
                    type="number"
                    :label="`${fieldLabel('blacklist_threshold')} ${entry.model_id}`"
                    :placeholder="placeholder(entry, 'blacklist_threshold')"
                    :invalid="invalidInputs[fieldKey(entry, 'blacklist_threshold')]"
                    :disabled="entry.entry_id.startsWith('derived:')"
                    size="compact"
                    @update:model-value="setInput(entry, 'blacklist_threshold', $event)"
                  />
                  <AppButton
                    variant="link"
                    size="inline"
                    :disabled="entry.entry_id.startsWith('derived:')"
                    @click="clearField(entry, 'blacklist_threshold')"
                    >{{ text('clear') }}</AppButton
                  >
                  <small
                    v-if="invalidInputs[fieldKey(entry, 'blacklist_threshold')]"
                    class="schedule-entry__validation"
                    role="alert"
                  >
                    {{ text('invalidValue') }}
                  </small>
                </label>
                <label>
                  <span>{{ text('cooldownSeconds') }}</span>
                  <AppTextInput
                    :model-value="inputValue(entry, 'cooldown_seconds')"
                    type="number"
                    :label="`${fieldLabel('cooldown_seconds')} ${entry.model_id}`"
                    :placeholder="placeholder(entry, 'cooldown_seconds')"
                    :invalid="invalidInputs[fieldKey(entry, 'cooldown_seconds')]"
                    :disabled="entry.entry_id.startsWith('derived:')"
                    size="compact"
                    @update:model-value="setInput(entry, 'cooldown_seconds', $event)"
                  />
                  <AppButton
                    variant="link"
                    size="inline"
                    :disabled="entry.entry_id.startsWith('derived:')"
                    @click="clearField(entry, 'cooldown_seconds')"
                    >{{ text('clear') }}</AppButton
                  >
                  <small
                    v-if="invalidInputs[fieldKey(entry, 'cooldown_seconds')]"
                    class="schedule-entry__validation"
                    role="alert"
                  >
                    {{ text('invalidValue') }}
                  </small>
                </label>
                <AppButton
                  variant="link"
                  size="inline"
                  :disabled="entry.entry_id.startsWith('derived:')"
                  @click="clearBreaker(entry)"
                  >{{ text('breaker') }}</AppButton
                >
              </div>

              <dl class="schedule-entry__facts">
                <div>
                  <dt>{{ text('share') }}</dt>
                  <dd>
                    {{ (shareValue(group.group_id, entry) * 100).toFixed(1) }}%
                    <small v-if="isDraftShare(group.group_id, entry)">{{
                      text('draftPreview')
                    }}</small>
                    <small v-else-if="hasWeightDraft">{{ text('observedShare') }}</small>
                  </dd>
                </div>
                <div>
                  <dt>{{ text('fallback') }}</dt>
                  <dd>{{ entry.fallback ? 'P' + entry.priority : 'P1' }}</dd>
                </div>
                <div>
                  <dt>{{ text('reason') }}</dt>
                  <dd>
                    <code>{{ reasonLabel(entry) }}</code>
                  </dd>
                </div>
                <div>
                  <dt>{{ text('credentials') }}</dt>
                  <dd>{{ credentialSummary(entry) }}</dd>
                </div>
                <div>
                  <dt>{{ text('runtime') }}</dt>
                  <dd :class="runtimeTone(entry)">
                    {{ runtimeLabel(entry) }} · {{ entry.runtime.failure_count }}
                    {{ text('failures') }}
                  </dd>
                </div>
                <div v-if="entry.runtime.cooldown_until_ms !== null">
                  <dt>{{ text('cooldown') }}</dt>
                  <dd>{{ cooldownLabel(entry.runtime.cooldown_until_ms) }}</dd>
                </div>
              </dl>

              <div class="schedule-entry__breaker-summary">
                <span>{{ text('breaker') }}</span>
                <span
                  >{{ text('effective') }}
                  {{ entry.circuit_breaker.effective.blacklist_threshold ?? '-' }} /
                  {{ entry.circuit_breaker.effective.cooldown_seconds ?? '-' }}s</span
                >
                <span
                  >{{ text('configured') }}
                  {{ entry.circuit_breaker.configured.blacklist_threshold ?? '-' }} /
                  {{ entry.circuit_breaker.configured.cooldown_seconds ?? '-' }}s</span
                >
                <span
                  >{{ text('inherited') }}
                  {{ breakerSourceLabel(entry.circuit_breaker.sources.blacklist_threshold) }}/{{
                    breakerSourceLabel(entry.circuit_breaker.sources.cooldown_seconds)
                  }}</span
                >
              </div>
              <AppButton
                v-if="entry.runtime.state !== 'available'"
                variant="secondary"
                size="compact"
                :busy="recovering === entryKey(group.group_id, entry.entry_id)"
                :disabled="entry.entry_id.startsWith('derived:')"
                @click="recover(group.group_id, entry.entry_id)"
                >{{ text('recover') }}</AppButton
              >
            </article>
          </div>
        </article>
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
.schedule-detail__header,
.schedule-group__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-3);
}
.schedule-detail__eyebrow {
  margin: 0 0 4px;
  overflow: hidden;
  color: var(--color-action);
  font-family: var(--font-mono);
  font-size: var(--text-meta);
  text-overflow: ellipsis;
  white-space: nowrap;
}
.schedule-detail h2,
.schedule-group h3 {
  margin: 0;
  color: var(--color-text);
}
.schedule-detail h2 {
  font-size: var(--text-lg);
}
.schedule-group h3 {
  font-size: var(--text-md);
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
  border-radius: var(--radius-card);
  color: var(--color-text-muted);
  padding: 30px;
  text-align: center;
}
.schedule-detail__groups {
  display: grid;
  gap: var(--space-4);
}
.schedule-group {
  min-width: 0;
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-card);
  background: var(--color-surface);
  overflow: hidden;
}
.schedule-group__header {
  border-bottom: 1px solid var(--color-border-subtle);
  background: var(--color-surface-sunken);
  padding: 14px 16px;
}
.schedule-group__header p {
  margin: 5px 0 0;
  color: var(--color-text-muted);
  font-size: var(--text-meta);
}
.schedule-group__weight {
  color: var(--color-text-muted);
  font-size: var(--text-meta);
  white-space: nowrap;
}
.schedule-group__weight strong {
  color: var(--color-text);
  font-family: var(--font-mono);
}
.schedule-entries {
  display: grid;
}
.schedule-entry {
  display: grid;
  grid-template-columns: minmax(170px, 1fr) minmax(360px, 2.5fr) minmax(240px, 1.5fr) auto;
  min-width: 0;
  align-items: start;
  gap: var(--space-3);
  border-bottom: 1px solid var(--color-border-subtle);
  padding: 16px;
}
.schedule-entry:last-child {
  border-bottom: 0;
}
.schedule-entry--excluded {
  background: color-mix(in srgb, var(--color-surface-sunken) 50%, var(--color-surface));
}
.schedule-entry__identity,
.schedule-entry__editors {
  display: grid;
  min-width: 0;
  gap: 6px;
}
.schedule-entry__name {
  display: grid;
  min-width: 0;
  gap: 2px;
}
.schedule-entry__name strong,
.schedule-entry__model,
.schedule-entry__identity code,
.schedule-entry__readonly {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.schedule-entry__name strong {
  color: var(--color-text);
}
.schedule-entry__model,
.schedule-entry__identity code {
  color: var(--color-text-muted);
  font-size: var(--text-meta);
}
.schedule-entry__identity code,
.schedule-entry__facts code {
  font-family: var(--font-mono);
}
.schedule-entry__readonly {
  color: var(--color-warning);
  font-size: 10px;
}
.schedule-entry__editors {
  grid-template-columns: repeat(2, minmax(110px, 1fr));
}
.schedule-entry__editors label {
  display: grid;
  min-width: 0;
  gap: 3px;
  color: var(--color-text-muted);
  font-size: 10px;
}
.schedule-entry__editors label :deep(.app-button) {
  justify-self: start;
  color: var(--color-text-faint);
  font-size: 10px;
}
.schedule-entry__facts {
  display: grid;
  min-width: 0;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 7px 12px;
  margin: 0;
  font-size: var(--text-meta);
}
.schedule-entry__facts div {
  min-width: 0;
}
.schedule-entry__facts dt {
  color: var(--color-text-faint);
  font-size: 10px;
}
.schedule-entry__facts dd {
  overflow: hidden;
  margin: 2px 0 0;
  color: var(--color-text-muted);
  text-overflow: ellipsis;
  white-space: nowrap;
}
.schedule-entry__facts dd small {
  display: block;
  color: var(--color-text-faint);
  font-size: 10px;
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
.schedule-entry__validation {
  color: var(--color-danger);
  font-size: 10px;
}
.schedule-entry__breaker-summary {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px 8px;
  color: var(--color-text-faint);
  font-size: 10px;
}
.schedule-entry__breaker-summary span:first-child {
  color: var(--color-text-muted);
  font-weight: 650;
}
.schedule-entry > :deep(.app-button) {
  align-self: end;
}
.schedule-detail :deep(.sticky-save-bar) {
  position: sticky;
  z-index: 2;
  bottom: 12px;
}
@media (max-width: 1100px) {
  .schedule-entry {
    grid-template-columns: minmax(170px, 1fr) minmax(300px, 2fr);
  }
  .schedule-entry__facts,
  .schedule-entry__breaker-summary,
  .schedule-entry > :deep(.app-button) {
    grid-column: 1 / -1;
  }
}
@media (max-width: 620px) {
  .schedule-detail__header,
  .schedule-group__header {
    display: grid;
  }
  .schedule-detail__observed {
    justify-content: flex-start;
    text-align: left;
  }
  .schedule-entry {
    grid-template-columns: minmax(0, 1fr);
    padding: 14px;
  }
  .schedule-entry__editors {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .schedule-entry__facts,
  .schedule-entry__breaker-summary,
  .schedule-entry > :deep(.app-button) {
    grid-column: auto;
  }
}
</style>
