<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import { computed, ref, watch } from 'vue'

import { useApiClient } from '@/api/client-context'
import { enabledDataProtocols } from '@/api/control/protocols'
import type { AccessProtocol } from '@/api/control/types'
import { ApiError } from '@/api/errors'
import { accessKeyOptionsQueryOptions } from '@/app/resources/access-keys'
import {
  modelRouteScheduleDetailQueryOptions,
  modelRouteScheduleIndexQueryOptions,
  type ModelRouteScheduleIndexItemDto,
} from '@/app/resources/model-route-schedule'
import { type RouteInspectOperation } from '@/app/resources/route-inspection'
import AppSelect from '@/components/ui/AppSelect.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'

import SchedulePanelDetail, { type SchedulePanelDetailLabels } from './SchedulePanelDetail.vue'
import SchedulePanelIndex, { type SchedulePanelIndexLabels } from './SchedulePanelIndex.vue'

export interface SchedulePanelLabels {
  model: string
  protocol: string
  operation: string
  accessKey: string
  selectModel: string
  selectProtocol: string
  selectAccessKey: string
  loadingOptions: string
  optionsFailed: string
  contextRequired: string
  kicker?: string
  contextReady?: string
  context?: string
  disabled?: string
  retry?: string
  indexFailed?: string
  detailFailed?: string
  protocolLabels?: Partial<Record<AccessProtocol, string>>
  operationLabels?: Partial<Record<RouteInspectOperation, string>>
  index?: Partial<SchedulePanelIndexLabels>
  detail?: Partial<SchedulePanelDetailLabels>
}

const props = withDefaults(
  defineProps<{
    externalModel?: string
    protocol?: AccessProtocol
    operation?: RouteInspectOperation
    accessKeyId?: number
    labels?: Partial<SchedulePanelLabels>
    locale?: string
  }>(),
  {
    externalModel: '',
    protocol: 'openai-completions',
    operation: undefined,
    accessKeyId: undefined,
    labels: () => ({}),
    locale: 'en-US',
  },
)
const emit = defineEmits<{
  'update:externalModel': [value: string]
  'update:protocol': [value: AccessProtocol]
  'update:operation': [value: RouteInspectOperation | undefined]
  'update:accessKeyId': [value: number | undefined]
  'change-context': [
    context: {
      externalModel: string
      protocol: AccessProtocol
      operation?: RouteInspectOperation
      accessKeyId?: number
    },
  ]
  selectModel: [model: string]
  refresh: []
  saved: [snapshotRevision: number]
  recovered: [groupId: number, entryId: string]
}>()

const client = useApiClient()
const protocolOperations: Record<AccessProtocol, readonly RouteInspectOperation[]> = {
  'openai-completions': ['chat_completion'],
  'openai-responses': [
    'responses_create',
    'responses_retrieve',
    'responses_delete',
    'responses_cancel',
    'responses_input_items',
    'responses_compact',
    'responses_input_tokens',
    'responses_passthrough',
  ],
  'openai-images': ['images_generate', 'images_edit'],
  'openai-embeddings': ['embeddings_create'],
  anthropic: ['chat_completion', 'count_tokens'],
  gemini: ['chat_completion', 'count_tokens'],
}

function operationOptionsForProtocol(protocol: AccessProtocol): readonly RouteInspectOperation[] {
  return protocolOperations[protocol] ?? []
}

function correctedOperation(
  protocol: AccessProtocol,
  operation: RouteInspectOperation | undefined,
): RouteInspectOperation | undefined {
  const options = operationOptionsForProtocol(protocol)
  return operation !== undefined && options.includes(operation) ? operation : options[0]
}

const selectedModel = ref(props.externalModel)
const selectedProtocol = ref<AccessProtocol>(props.protocol)
const selectedOperation = ref<RouteInspectOperation | undefined>(
  correctedOperation(props.protocol, props.operation),
)
const selectedAccessKeyID = ref(props.accessKeyId)
const indexQuery = useQuery(modelRouteScheduleIndexQueryOptions(client))
const accessKeyQuery = useQuery(accessKeyOptionsQueryOptions(client))
const detailRequest = computed(() => {
  if (!selectedModel.value || selectedAccessKeyID.value === undefined) return undefined
  const request: {
    protocol: AccessProtocol
    external_model: string
    access_key_id: number
    operation?: RouteInspectOperation
  } = {
    protocol: selectedProtocol.value,
    external_model: selectedModel.value,
    access_key_id: selectedAccessKeyID.value,
  }
  if (selectedOperation.value !== undefined) request.operation = selectedOperation.value
  return request
})
const detailQuery = useQuery(modelRouteScheduleDetailQueryOptions(client, detailRequest))

const text = (key: keyof SchedulePanelLabels): string => {
  const value = props.labels[key]
  return typeof value === 'string' ? value : ''
}
const indexItems = computed<ModelRouteScheduleIndexItemDto[]>(
  () => indexQuery.data.value?.items ?? [],
)
const protocolOptions = computed(() => [
  { value: '', label: text('selectProtocol') },
  ...enabledDataProtocols.map((value) => ({
    value,
    label: props.labels.protocolLabels?.[value] ?? value,
  })),
])
const operationOptions = computed(() =>
  operationOptionsForProtocol(selectedProtocol.value).map((value) => ({
    value,
    label: props.labels.operationLabels?.[value] ?? value,
  })),
)
const accessKeyOptions = computed(() => [
  { value: '', label: text('selectAccessKey') },
  ...(accessKeyQuery.data.value ?? []).map((key) => ({
    value: String(key.id),
    label: `${key.name} (#${key.id})${key.status === 'disabled' ? ` - ${text('disabled')}` : ''}`,
  })),
])
const optionsError = computed(() => accessKeyQuery.error.value ?? undefined)
const indexError = computed(() => errorMessage(indexQuery.error.value, text('indexFailed')))
const detailError = computed(() => errorMessage(detailQuery.error.value, text('detailFailed')))

watch(
  () => props.externalModel,
  (value) => {
    if (value !== undefined) selectedModel.value = value
  },
)
watch(
  () => props.protocol,
  (value) => {
    if (value === undefined) return
    selectedProtocol.value = value
    const nextOperation = correctedOperation(value, selectedOperation.value)
    if (nextOperation !== selectedOperation.value) {
      selectedOperation.value = nextOperation
      emit('update:operation', nextOperation)
    }
  },
)
watch(
  () => props.operation,
  (value) => {
    const nextOperation = correctedOperation(selectedProtocol.value, value)
    if (nextOperation !== selectedOperation.value) selectedOperation.value = nextOperation
  },
)
watch(
  () => props.accessKeyId,
  (value) => {
    selectedAccessKeyID.value = value
  },
)
watch(
  () => accessKeyQuery.data.value,
  (options) => {
    if (selectedAccessKeyID.value !== undefined || !options?.length) return
    const active = options.find((option) => option.status === 'active') ?? options[0]
    selectedAccessKeyID.value = active.id
    emit('update:accessKeyId', active.id)
  },
  { immediate: true },
)

function errorMessage(error: unknown, fallback: string): string {
  if (!error) return ''
  if (error instanceof ApiError && error.message && error.message !== error.code)
    return error.message
  return fallback
}

function announceContext(): void {
  if (selectedAccessKeyID.value === undefined || !selectedModel.value) return
  emit('change-context', {
    externalModel: selectedModel.value,
    protocol: selectedProtocol.value,
    ...(selectedOperation.value === undefined ? {} : { operation: selectedOperation.value }),
    accessKeyId: selectedAccessKeyID.value,
  })
}

function selectModel(model: string): void {
  selectedModel.value = model
  emit('update:externalModel', model)
  emit('selectModel', model)
  announceContext()
}

function setProtocol(value: string): void {
  if (!enabledDataProtocols.includes(value as AccessProtocol)) return
  selectedProtocol.value = value as AccessProtocol
  const nextOperation = correctedOperation(selectedProtocol.value, selectedOperation.value)
  selectedOperation.value = nextOperation
  emit('update:protocol', selectedProtocol.value)
  emit('update:operation', nextOperation)
  announceContext()
}

function setOperation(value: string): void {
  if (!operationOptionsForProtocol(selectedProtocol.value).includes(value as RouteInspectOperation))
    return
  selectedOperation.value = value as RouteInspectOperation
  emit('update:operation', selectedOperation.value)
  announceContext()
}

function setAccessKey(value: string): void {
  if (!/^\d+$/.test(value)) {
    selectedAccessKeyID.value = undefined
    emit('update:accessKeyId', undefined)
    return
  }
  const id = Number(value)
  if (!Number.isSafeInteger(id) || id <= 0) return
  selectedAccessKeyID.value = id
  emit('update:accessKeyId', id)
  announceContext()
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
</script>

<template>
  <section class="schedule-panel" aria-labelledby="schedule-panel-title">
    <header class="schedule-panel__heading">
      <div>
        <p class="schedule-panel__kicker">{{ text('kicker') }}</p>
        <h1 id="schedule-panel-title">{{ text('model') }}</h1>
      </div>
      <span class="schedule-panel__context-state" :data-ready="Boolean(detailRequest)">
        {{ detailRequest ? text('contextReady') : text('contextRequired') }}
      </span>
    </header>

    <div class="schedule-panel__context" :aria-label="text('context')">
      <label>
        <span>{{ text('model') }}</span>
        <span class="schedule-panel__model-value">{{ selectedModel || text('selectModel') }}</span>
      </label>
      <label>
        <span>{{ text('protocol') }}</span>
        <AppSelect
          :model-value="selectedProtocol"
          :label="text('protocol')"
          :options="protocolOptions"
          size="compact"
          @update:model-value="setProtocol"
        />
      </label>
      <label v-if="operationOptions.length">
        <span>{{ text('operation') }}</span>
        <AppSelect
          :model-value="selectedOperation"
          :label="text('operation')"
          :options="operationOptions"
          size="compact"
          @update:model-value="setOperation"
        />
      </label>
      <label>
        <span>{{ text('accessKey') }}</span>
        <AppSelect
          :model-value="selectedAccessKeyID === undefined ? '' : String(selectedAccessKeyID)"
          :label="text('accessKey')"
          :options="accessKeyOptions"
          :disabled="accessKeyQuery.isPending.value || accessKeyQuery.isError.value"
          size="compact"
          @update:model-value="setAccessKey"
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

    <div class="schedule-panel__layout">
      <SchedulePanelIndex
        :items="indexItems"
        :selected-model="selectedModel"
        :loading="indexQuery.isPending.value"
        :refreshing="indexQuery.isFetching.value"
        :error="indexError"
        :labels="labels?.index"
        @select-model="selectModel"
        @refresh="refreshIndex"
      />
      <SchedulePanelDetail
        :detail="detailQuery.data.value"
        :loading="
          detailQuery.isPending.value &&
          detailQuery.data.value === undefined &&
          Boolean(detailRequest)
        "
        :error="detailRequest ? detailError : ''"
        :refreshing="detailQuery.isFetching.value && detailQuery.data.value !== undefined"
        :locale="locale"
        :labels="labels?.detail"
        @refresh="refreshDetail"
        @saved="onSaved"
        @recovered="onRecovered"
      />
    </div>
  </section>
</template>

<style scoped>
.schedule-panel {
  display: grid;
  min-width: 0;
  gap: var(--space-5);
}
.schedule-panel__heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-4);
  border-bottom: 1px solid var(--color-border-subtle);
  padding-bottom: var(--space-4);
}
.schedule-panel__kicker {
  margin: 0 0 5px;
  color: var(--color-action);
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.14em;
}
.schedule-panel h1 {
  margin: 0;
  color: var(--color-text);
  font-size: clamp(1.35rem, 2vw, 1.85rem);
}
.schedule-panel__context-state {
  color: var(--color-warning);
  font-size: var(--text-meta);
}
.schedule-panel__context-state[data-ready='true'] {
  color: var(--color-success);
}
.schedule-panel__context {
  display: grid;
  grid-template-columns: minmax(180px, 1.4fr) repeat(3, minmax(150px, 1fr));
  align-items: end;
  gap: var(--space-3);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-card);
  background: var(--color-surface-sunken);
  padding: 14px;
}
.schedule-panel__context label {
  display: grid;
  min-width: 0;
  gap: 6px;
  color: var(--color-text-muted);
  font-size: var(--text-meta);
  font-weight: 600;
}
.schedule-panel__context label :deep(.app-select__trigger) {
  width: 100%;
}
.schedule-panel__model-value {
  display: flex;
  min-height: var(--control-compact);
  align-items: center;
  overflow: hidden;
  border: 1px solid var(--color-border-control);
  border-radius: var(--radius-control);
  background: var(--color-surface);
  color: var(--color-text);
  padding: 0 10px;
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  text-overflow: ellipsis;
  white-space: nowrap;
}
.schedule-panel__layout {
  display: grid;
  min-width: 0;
  grid-template-columns: minmax(230px, 0.65fr) minmax(0, 2fr);
  align-items: start;
  gap: var(--space-6);
}
@media (max-width: 1120px) {
  .schedule-panel__context {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .schedule-panel__context label:first-child {
    grid-column: 1 / -1;
  }
  .schedule-panel__layout {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 620px) {
  .schedule-panel__heading {
    display: grid;
    align-items: start;
  }
  .schedule-panel__context {
    grid-template-columns: 1fr;
  }
  .schedule-panel__context label:first-child {
    grid-column: auto;
  }
}
</style>
