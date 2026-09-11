<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'

import { useApiClient } from '@/api/client-context'
import { useToast } from '@/app/toast'
import { accessKeyOptionsQueryOptions } from '@/app/resources/access-keys'
import type { ModelProbeTargetDto } from '@/app/resources/model-probe'
import { monitorLocation } from '@/app/route-locations'
import {
  modelRouteScheduleDetailQueryOptions,
  modelRouteScheduleIndexQueryOptions,
  type ModelRouteScheduleIndexItemDto,
} from '@/app/resources/model-route-schedule'
import AppSelect from '@/components/ui/AppSelect.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import ModelProbeDialog from '@/features/models/ModelProbeDialog.vue'
import { useModelProbe } from '@/features/models/use-model-probe'

import { type ScheduleDrafts, type ScheduleMode } from './monitor-route'
import SchedulePanelDetail, { type SchedulePanelDetailLabels } from './SchedulePanelDetail.vue'

export interface SchedulePanelLabels {
  model: string
  mode: string
  selectModel: string
  loadingOptions: string
  optionsFailed: string
  contextRequired: string
  kicker?: string
  contextReady?: string
  context?: string
  retry?: string
  indexFailed?: string
  detailFailed?: string
  modeLabels?: Partial<Record<ScheduleMode, string>>
  detail?: Partial<SchedulePanelDetailLabels>
}

const props = withDefaults(
  defineProps<{
    externalModel?: string
    mode?: ScheduleMode
    selectedRow?: string
    drafts?: ScheduleDrafts
    labels?: Partial<SchedulePanelLabels>
    locale?: string
  }>(),
  {
    externalModel: '',
    mode: 'all',
    selectedRow: undefined,
    drafts: () => ({}),
    labels: () => ({}),
    locale: 'en-US',
  },
)
const emit = defineEmits<{
  'change-context': [context: { externalModel?: string; mode: ScheduleMode }]
  'draft-change': [drafts: ScheduleDrafts]
  'row-change': [row: string | undefined]
  refresh: []
  saved: [snapshotRevision: number]
  recovered: [groupId: number, entryId: string]
}>()

const client = useApiClient()
const { t } = useI18n()
const toast = useToast()
const selectedModel = ref(props.externalModel)
const selectedMode = ref<ScheduleMode>(props.mode)
const selectedAccessKeyID = ref<number>()
const indexQuery = useQuery(modelRouteScheduleIndexQueryOptions(client))
const accessKeyQuery = useQuery(accessKeyOptionsQueryOptions(client))
const indexItems = computed<ModelRouteScheduleIndexItemDto[]>(
  () => indexQuery.data.value?.items ?? [],
)
const selectedIndexItem = computed(() =>
  indexItems.value.find(({ external_model }) => external_model === selectedModel.value),
)
const detailRequest = computed(() => {
  const indexItem = selectedIndexItem.value
  if (!indexItem || selectedAccessKeyID.value === undefined) return undefined
  return {
    protocol: indexItem.protocol,
    external_model: selectedModel.value,
    access_key_id: selectedAccessKeyID.value,
    operation: indexItem.operation,
  }
})
const detailQuery = useQuery(modelRouteScheduleDetailQueryOptions(client, detailRequest))
const modelOptions = computed(() => [
  { value: '', label: text('selectModel') },
  ...indexItems.value.map((item) => ({ value: item.external_model, label: item.external_model })),
])
const modeOptions = computed(() =>
  (['all', 'primary', 'fallback'] as const).map((mode) => ({
    value: mode,
    label: props.labels.modeLabels?.[mode] ?? mode,
  })),
)
const optionsError = computed(() => accessKeyQuery.error.value ?? undefined)
const indexError = computed(() =>
  indexQuery.error.value ? errorMessage(indexQuery.error.value, text('indexFailed')) : '',
)
const detailError = computed(() =>
  detailQuery.error.value ? errorMessage(detailQuery.error.value, text('detailFailed')) : '',
)

const text = (key: keyof SchedulePanelLabels): string => {
  const value = props.labels[key]
  return typeof value === 'string' ? value : ''
}

watch(
  () => props.externalModel,
  (value) => {
    selectedModel.value = value ?? ''
  },
)
watch(
  () => props.mode,
  (value) => {
    selectedMode.value = value
  },
)
watch(
  () => accessKeyQuery.data.value,
  (options) => {
    if (selectedAccessKeyID.value !== undefined || !options?.length) return
    selectedAccessKeyID.value = (
      options.find((option) => option.status === 'active') ?? options[0]
    ).id
  },
  { immediate: true },
)

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback
}

function commitContext(): void {
  emit('change-context', {
    externalModel: selectedModel.value || undefined,
    mode: selectedMode.value,
  })
}

function setModel(value: string): void {
  selectedModel.value = value
  commitContext()
}

function setMode(value: string): void {
  if (value !== 'all' && value !== 'primary' && value !== 'fallback') return
  selectedMode.value = value
  commitContext()
}

async function refreshIndex(): Promise<void> {
  await indexQuery.refetch()
  emit('refresh')
}

async function refreshDetail(): Promise<void> {
  if (detailRequest.value) await detailQuery.refetch()
  await indexQuery.refetch()
  emit('refresh')
}

async function refreshAll(): Promise<void> {
  await Promise.all([indexQuery.refetch(), detailQuery.refetch()])
  emit('refresh')
}

async function onSaved(revision: number): Promise<void> {
  emit('saved', revision)
  await refreshAll()
}

async function onRecovered(groupID: number, entryID: string): Promise<void> {
  emit('recovered', groupID, entryID)
  await refreshAll()
}

const router = useRouter()
const {
  open: probeOpen,
  pending: probePending,
  failed: probeFailed,
  stopped: probeStopped,
  results: probeResults,
  disabledGroupIds: probeDisabledGroupIds,
  total: probeTotal,
  completed: probeCompleted,
  start: startProbe,
  stop: stopProbe,
  close: closeProbe,
} = useModelProbe()

function probeRow(groupID: number, modelID: string, disabled: boolean): void {
  void startProbe([{ group_id: groupID, model: modelID }], {
    disabledGroupIds: disabled ? [groupID] : [],
  })
}

function probeRows(targets: ModelProbeTargetDto[], disabledGroupIds: number[]): void {
  void startProbe(targets, { disabledGroupIds })
}

function handleProbeOpen(value: boolean): void {
  if (!value) closeProbe()
}

function viewProbeLog(logID: string): void {
  void router.push(monitorLocation({ tab: 'logs', selected_request_id: logID }))
}

const probeApplying = ref(false)
const groupEnabledById = computed(
  () => new Map((detailQuery.data.value?.groups ?? []).map((g) => [g.group_id, g.enabled])),
)
const detailRef = ref<InstanceType<typeof SchedulePanelDetail> | null>(null)

async function onApplyProbeEnabled(changes: Map<number, boolean>): Promise<void> {
  probeApplying.value = true
  try {
    const target = detailRef.value
    if (!target) {
      toast.show({ message: t('monitor.modelProbe.toggle.applyFailed'), tone: 'danger' })
      return
    }
    const ok = await target.applyProbeEnabled(changes)
    if (ok) {
      const groups = detailQuery.data.value?.groups ?? []
      const count = [...changes.keys()].filter((id) => groups.some((g) => g.group_id === id)).length
      toast.show({ message: t('monitor.modelProbe.toggle.applied', { count }), tone: 'success' })
      closeProbe()
      return
    }
    const groups = detailQuery.data.value?.groups ?? []
    const hadRealCall = [...changes.keys()].some((id) => groups.some((g) => g.group_id === id))
    if (!hadRealCall) {
      toast.show({ message: t('monitor.modelProbe.toggle.applyFailed'), tone: 'danger' })
    }
  } finally {
    probeApplying.value = false
  }
}
</script>

<template>
  <section class="schedule-panel" :aria-label="text('model')">
    <div class="schedule-panel__filters" :aria-label="text('context')">
      <label>
        <span>{{ text('mode') }}</span>
        <AppSelect
          :model-value="selectedMode"
          :label="text('mode')"
          :options="modeOptions"
          size="compact"
          @update:model-value="setMode"
        />
      </label>
      <label>
        <span>{{ text('model') }}</span>
        <AppSelect
          :model-value="selectedModel"
          :label="text('model')"
          :options="modelOptions"
          :disabled="indexQuery.isPending.value || indexQuery.isError.value"
          size="compact"
          @update:model-value="setModel"
        />
      </label>
    </div>

    <QueryFeedback
      v-if="accessKeyQuery.isError.value"
      state="error"
      :message="errorMessage(optionsError, text('optionsFailed'))"
      :retry-label="text('retry')"
      @retry="accessKeyQuery.refetch"
    />
    <QueryFeedback
      v-else-if="accessKeyQuery.isPending.value"
      state="loading"
      :message="text('loadingOptions')"
    />
    <QueryFeedback
      v-if="indexQuery.isPending.value"
      state="loading"
      :message="text('loadingOptions')"
    />
    <QueryFeedback
      v-else-if="indexError"
      state="error"
      :message="indexError"
      :retry-label="text('retry')"
      @retry="refreshIndex"
    />

    <SchedulePanelDetail
      ref="detailRef"
      :detail="detailQuery.data.value"
      :loading="
        detailQuery.isPending.value &&
        detailQuery.data.value === undefined &&
        Boolean(detailRequest)
      "
      :error="detailRequest ? detailError : ''"
      :refreshing="detailQuery.isFetching.value && detailQuery.data.value !== undefined"
      :locale="locale"
      :mode="selectedMode"
      :selected-row="selectedRow"
      :drafts="drafts"
      :labels="labels?.detail"
      @refresh="refreshDetail"
      @saved="onSaved"
      @recovered="onRecovered"
      @probe="probeRow"
      @probe-all="probeRows"
      @draft-change="emit('draft-change', $event)"
      @row-change="emit('row-change', $event)"
    />
    <ModelProbeDialog
      :open="probeOpen"
      :pending="probePending"
      :failed="probeFailed"
      :stopped="probeStopped"
      :results="probeResults"
      :disabled-group-ids="probeDisabledGroupIds"
      :total="probeTotal"
      :completed="probeCompleted"
      :group-enabled-by-id="groupEnabledById"
      :applying="probeApplying"
      @update:open="handleProbeOpen"
      @stop="stopProbe"
      @view-log="viewProbeLog"
      @apply-enabled="onApplyProbeEnabled"
    />
  </section>
</template>

<style scoped>
.schedule-panel {
  display: grid;
  min-width: 0;
  gap: var(--space-4);
}
/* 卡片向外铺满舞台：外扩量必须等于舞台自身的内边距（stage-padding-*），
   用 sheet 的内边距会多撑出 8px(桌面) / 6px(窄屏)，让整页出现横向滚动。 */
:global(.monitor-page.ledger-sheet--padded) {
  width: min(calc(100% + var(--stage-padding-inline) * 2), var(--content-max));
  margin-inline: calc(0px - var(--stage-padding-inline));
}
.schedule-panel__filters {
  display: grid;
  grid-template-columns: minmax(150px, 0.5fr) minmax(220px, 1fr);
  gap: var(--space-3);
  max-width: 620px;
}
.schedule-panel__filters label {
  display: grid;
  min-width: 0;
  gap: 5px;
  color: var(--color-text-muted);
  font-size: var(--text-meta);
  font-weight: 650;
}
.schedule-panel__filters label :deep(.app-select__trigger) {
  width: 100%;
}
@media (max-width: 860px) {
  :global(.monitor-page.ledger-sheet--padded) {
    width: calc(100% + var(--stage-padding-inline-compact) * 2);
    margin-inline: calc(0px - var(--stage-padding-inline-compact));
  }
}
@media (max-width: 620px) {
  .schedule-panel__filters {
    grid-template-columns: 1fr;
  }
}
</style>
