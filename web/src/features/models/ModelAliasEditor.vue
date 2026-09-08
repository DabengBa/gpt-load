<script setup lang="ts" generic="T extends ModelDraftValue">
import { ChevronDown, ChevronRight, Plus, X } from '@lucide/vue'
import { computed, nextTick, ref, useId, watch } from 'vue'

import LedgerRecordList from '@/components/collection/LedgerRecordList.vue'
import AppButton from '@/components/ui/AppButton.vue'
import AppSearchInput from '@/components/ui/AppSearchInput.vue'
import AppTextInput from '@/components/ui/AppTextInput.vue'
import CompactFieldError from '@/components/ui/CompactFieldError.vue'
import IconButton from '@/components/ui/IconButton.vue'

import {
  clientModel,
  modelDraftValidity,
  routeEntryShares,
  type ModelAliasEditorLabels,
  type ModelDraftKey,
  type ModelDraftValue,
  type ModelNameConflict,
} from './model-draft'

const props = withDefaults(
  defineProps<{
    modelValue: T[]
    conflicts: readonly ModelNameConflict[]
    labels: ModelAliasEditorLabels
    createRow?: () => T
    disabled?: boolean
    searchable?: boolean
    addable?: boolean
    search?: string
    validationMode?: 'immediate' | 'blur'
    showAllErrors?: boolean
  }>(),
  {
    createRow: undefined,
    disabled: false,
    searchable: true,
    addable: true,
    search: undefined,
    validationMode: 'immediate',
    showAllErrors: false,
  },
)
const emit = defineEmits<{
  'update:modelValue': [value: T[]]
  'update:search': [value: string]
  'visible-validation-change': [indexes: Set<number>]
}>()

const instanceId = useId()
const root = ref<HTMLElement>()
const internalSearch = ref(props.search ?? '')
const touchedModelIDs = ref<Set<ModelDraftKey>>(new Set())
const touchedAliases = ref<Set<ModelDraftKey>>(new Set())
const searchValue = computed({
  get: () => internalSearch.value,
  set: (value: string) => {
    internalSearch.value = value
    emit('update:search', value)
  },
})
watch(
  () => props.search,
  (value) => {
    internalSearch.value = value ?? ''
  },
)
const validity = computed(() => modelDraftValidity(props.modelValue, props.conflicts))
const shares = computed(() => routeEntryShares(props.modelValue))

interface VisibleRow {
  item: T
  index: number
}

interface VisibleGroup {
  clientModel: string
  rows: VisibleRow[]
}

const visibleGroups = computed<VisibleGroup[]>(() => {
  const query = searchValue.value.trim().toLocaleLowerCase()
  const rows = props.modelValue.flatMap<VisibleRow>((item, index) =>
    !query || `${item.id} ${item.name} ${item.alias}`.toLocaleLowerCase().includes(query)
      ? [{ item, index }]
      : [],
  )
  const groups: VisibleGroup[] = []
  const byName = new Map<string, VisibleGroup>()
  for (const row of rows) {
    const name = clientModel(row.item) || '\u0000'
    let group = byName.get(name)
    if (!group) {
      group = { clientModel: name, rows: [] }
      byName.set(name, group)
      groups.push(group)
    }
    group.rows.push(row)
  }
  return groups
})

const visibleRowCount = computed(() =>
  visibleGroups.value.reduce((total, group) => total + group.rows.length, 0),
)
const groupHeaderCount = computed(
  () => visibleGroups.value.filter((group) => group.rows.length > 1).length,
)

const collapsedGroups = ref<Set<string>>(new Set())

function isCollapsed(name: string): boolean {
  return collapsedGroups.value.has(name)
}

function toggleGroup(name: string): void {
  const next = new Set(collapsedGroups.value)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  collapsedGroups.value = next
}

type RenderItem<T> =
  | { kind: 'header'; key: string; group: VisibleGroup }
  | { kind: 'row'; key: ModelDraftKey; item: T; index: number; nested: boolean }

const renderList = computed<RenderItem<T>[]>(() => {
  const list: RenderItem<T>[] = []
  for (const group of visibleGroups.value) {
    if (group.rows.length > 1) {
      list.push({ kind: 'header', key: `group:${group.clientModel}`, group })
      if (isCollapsed(group.clientModel)) continue
    }
    const nested = group.rows.length > 1
    for (const row of group.rows) {
      list.push({ kind: 'row', key: row.item.key, item: row.item, index: row.index, nested })
    }
  }
  return list
})

function sharePercent(index: number): string {
  const share = shares.value[index] ?? 0
  return `${Math.round(share * 1_000) / 10}%`
}

function routeCountText(value: number | null): string {
  return value === null ? '' : String(value)
}

type RouteCountField = 'weight' | 'priority'

function updateRow(
  index: number,
  patch: Partial<Pick<ModelDraftValue, 'id' | 'alias' | 'alias_enabled' | RouteCountField>>,
): void {
  emit(
    'update:modelValue',
    props.modelValue.map((item, current) =>
      current === index
        ? ({
            ...item,
            ...patch,
            alias: patch.alias_enabled === false ? '' : (patch.alias ?? item.alias),
          } as T)
        : ({ ...item, sources: [...item.sources] } as T),
    ),
  )
}

function updateRouteCount(index: number, field: RouteCountField, raw: string): void {
  const trimmed = raw.trim()
  if (trimmed === '') {
    updateRow(index, { [field]: null })
    return
  }
  updateRow(index, { [field]: Number(trimmed) })
}

function removeRow(index: number): void {
  emit(
    'update:modelValue',
    props.modelValue
      .filter((_, current) => current !== index)
      .map((item) => ({ ...item, sources: [...item.sources] })),
  )
}

async function addManual(): Promise<void> {
  if (props.disabled || !props.createRow) return
  searchValue.value = ''
  const index = props.modelValue.length
  emit('update:modelValue', [
    ...props.modelValue.map((item) => ({ ...item, sources: [...item.sources] })),
    props.createRow(),
  ])
  await nextTick()
  root.value?.querySelector<HTMLInputElement>(`[data-model-id-index="${index}"]`)?.focus()
}

function conflictMessage(index: number): string {
  const conflict = props.conflicts.find((item) => item.indexes.includes(index))
  return conflict ? props.labels.nameConflict(conflict.client_model) : ''
}

function modelIDError(item: ModelDraftValue, index: number): string {
  if (validity.value.emptyIDIndexes.has(index)) return props.labels.manualIdRequired
  if (!item.alias_enabled && validity.value.conflictIndexes.has(index)) {
    return conflictMessage(index)
  }
  return ''
}

function modelAliasError(item: ModelDraftValue, index: number): string {
  if (!item.alias_enabled) return ''
  if (validity.value.emptyAliasIndexes.has(index)) return props.labels.aliasRequired
  return validity.value.conflictIndexes.has(index) ? conflictMessage(index) : ''
}

function visibleWeightError(index: number): string {
  if (validity.value.invalidWeightIndexes.has(index)) return props.labels.invalidWeight
  if (validity.value.zeroShareIndexes.has(index)) return props.labels.zeroShare
  return ''
}

function visiblePriorityError(index: number): string {
  return validity.value.invalidPriorityIndexes.has(index) ? props.labels.invalidPriority : ''
}

function visibleModelIDError(item: ModelDraftValue, index: number): string {
  const error = modelIDError(item, index)
  if (!error) return ''
  if (
    props.validationMode === 'immediate' ||
    props.showAllErrors ||
    !item.editable_id ||
    touchedModelIDs.value.has(item.key)
  ) {
    return error
  }
  return ''
}

function visibleModelAliasError(item: ModelDraftValue, index: number): string {
  const error = modelAliasError(item, index)
  if (!error) return ''
  if (
    props.validationMode === 'immediate' ||
    props.showAllErrors ||
    touchedAliases.value.has(item.key)
  ) {
    return error
  }
  return ''
}

const visibleInvalidIndexes = computed(
  () =>
    new Set(
      props.modelValue.flatMap((item, index) =>
        visibleModelIDError(item, index) ||
        visibleModelAliasError(item, index) ||
        visibleWeightError(index) ||
        visiblePriorityError(index)
          ? [index]
          : [],
      ),
    ),
)

watch(visibleInvalidIndexes, (indexes) => emit('visible-validation-change', new Set(indexes)), {
  immediate: true,
})

function touchModelID(key: ModelDraftKey): void {
  if (touchedModelIDs.value.has(key)) return
  touchedModelIDs.value = new Set(touchedModelIDs.value).add(key)
}

function touchAlias(key: ModelDraftKey): void {
  if (touchedAliases.value.has(key)) return
  touchedAliases.value = new Set(touchedAliases.value).add(key)
}

async function setAliasEnabled(index: number, enabled: boolean): Promise<void> {
  const item = props.modelValue[index]
  if (enabled && item && props.validationMode === 'blur') {
    const nextTouched = new Set(touchedAliases.value)
    nextTouched.delete(item.key)
    touchedAliases.value = nextTouched
  }
  updateRow(index, { alias_enabled: enabled })
  if (!enabled) return

  await nextTick()
  root.value?.querySelector<HTMLInputElement>(`[data-alias-input-index="${index}"]`)?.focus()
}

async function focusFirstInvalid(): Promise<void> {
  const index = Math.min(...validity.value.invalidIndexes)
  if (!Number.isFinite(index)) return
  searchValue.value = ''
  const item = props.modelValue[index]
  const targetsModelID =
    validity.value.emptyIDIndexes.has(index) ||
    (validity.value.conflictIndexes.has(index) && item?.editable_id && !item.alias_enabled)
  const targetsWeight =
    validity.value.invalidWeightIndexes.has(index) || validity.value.zeroShareIndexes.has(index)
  const targetsPriority = validity.value.invalidPriorityIndexes.has(index)
  if (item) {
    if (targetsModelID) touchModelID(item.key)
    else if (item.alias_enabled && !targetsWeight && !targetsPriority) touchAlias(item.key)
  }
  await nextTick()
  const selector = targetsModelID
    ? `[data-model-id-index="${index}"]`
    : targetsWeight
      ? `[data-model-weight-index="${index}"]`
      : targetsPriority
        ? `[data-model-priority-index="${index}"]`
        : item?.alias_enabled
          ? `[data-alias-input-index="${index}"]`
          : `[data-alias-toggle-index="${index}"]`
  root.value?.querySelector<HTMLInputElement>(selector)?.focus()
}

defineExpose({ addManual, focusFirstInvalid })
</script>

<template>
  <div ref="root" class="model-alias-editor">
    <div v-if="searchable" class="model-alias-editor__toolbar">
      <label class="model-alias-editor__search">
        <span>{{ labels.searchLabel }}</span>
        <AppSearchInput
          v-model="searchValue"
          :label="labels.search"
          :placeholder="labels.search"
          :clear-label="labels.clearSearch"
          :disabled="disabled"
        />
      </label>
      <span class="model-alias-editor__count" aria-live="polite">
        {{ labels.count(modelValue.length) }}
      </span>
    </div>

    <LedgerRecordList
      :label="labels.tableLabel"
      :row-count="visibleRowCount + groupHeaderCount + 1"
      grid-class="model-alias-editor__grid"
    >
      <template #header>
        <span role="columnheader">{{ labels.id }}</span>
        <span role="columnheader">{{ labels.alias }}</span>
        <span role="columnheader">{{ labels.weight }} / {{ labels.priority }}</span>
        <span role="columnheader">{{ labels.thirdColumn }}</span>
        <span role="columnheader"
          ><span class="sr-only">{{ labels.actions }}</span></span
        >
      </template>

      <template v-for="(render, renderIndex) in renderList" :key="render.key">
        <article
          v-if="render.kind === 'header'"
          class="ledger-record-list__record model-alias-editor__group"
          role="row"
          :aria-rowindex="renderIndex + 2"
        >
          <div class="ledger-record-list__cell model-alias-editor__group-cell" role="cell">
            <button
              type="button"
              class="model-alias-editor__group-toggle"
              :aria-expanded="!isCollapsed(render.group.clientModel)"
              :disabled="disabled"
              @click="toggleGroup(render.group.clientModel)"
            >
              <ChevronDown
                v-if="!isCollapsed(render.group.clientModel)"
                :size="14"
                aria-hidden="true"
              />
              <ChevronRight v-else :size="14" aria-hidden="true" />
              <strong>{{ render.group.clientModel }}</strong>
              <span class="model-alias-editor__group-count">{{ render.group.rows.length }}</span>
            </button>
            <div class="model-alias-editor__distribution" aria-hidden="true">
              <i
                v-for="row in render.group.rows"
                :key="row.item.key"
                class="model-alias-editor__distribution-segment"
                :class="{
                  'model-alias-editor__distribution-segment--zero': (row.item.weight ?? 1) === 0,
                }"
                :style="{ width: sharePercent(row.index) }"
              />
            </div>
          </div>
        </article>
        <article
          v-else
          class="ledger-record-list__record model-alias-editor__record"
          :class="{
            'model-alias-editor__record--invalid': visibleInvalidIndexes.has(render.index),
            'model-alias-editor__record--nested': render.nested,
            'model-alias-editor__record--disabled': (render.item.weight ?? 1) === 0,
          }"
          role="row"
          :aria-rowindex="renderIndex + 2"
        >
          <div class="ledger-record-list__cell model-alias-editor__id" role="cell">
            <span class="model-alias-editor__mobile-label">{{ labels.id }}</span>
            <CompactFieldError
              :id="`${instanceId}-model-id-${render.index}`"
              class="model-alias-editor__id-field"
              :error="visibleModelIDError(render.item, render.index)"
            >
              <template #default="{ invalid, describedBy }">
                <AppTextInput
                  v-if="render.item.editable_id"
                  :id="`${instanceId}-model-id-${render.index}`"
                  :model-value="render.item.id"
                  appearance="surface"
                  size="compact"
                  monospace
                  :label="labels.id"
                  :placeholder="labels.manualId"
                  :invalid="invalid"
                  :described-by="describedBy"
                  :data-model-id-index="render.index"
                  :spellcheck="false"
                  :disabled="disabled"
                  @update:model-value="updateRow(render.index, { id: $event })"
                  @blur="touchModelID(render.item.key)"
                />
                <code v-else :aria-describedby="describedBy">{{ render.item.id }}</code>
              </template>
            </CompactFieldError>
          </div>

          <div class="ledger-record-list__cell model-alias-editor__alias-cell" role="cell">
            <span class="model-alias-editor__mobile-label">{{ labels.alias }}</span>
            <div class="model-alias-editor__alias-control">
              <label
                class="model-alias-editor__alias-toggle"
                :class="{ 'model-alias-editor__alias-toggle--disabled': disabled }"
              >
                <span class="sr-only">{{ labels.aliasEnabledFor(render.item.id) }}</span>
                <input
                  :data-alias-toggle-index="render.index"
                  type="checkbox"
                  :checked="render.item.alias_enabled"
                  :disabled="disabled"
                  @change="
                    setAliasEnabled(render.index, ($event.target as HTMLInputElement).checked)
                  "
                />
              </label>
              <CompactFieldError
                v-if="render.item.alias_enabled"
                :id="`${instanceId}-model-alias-${render.index}`"
                class="model-alias-editor__alias-field"
                :error="visibleModelAliasError(render.item, render.index)"
              >
                <template #default="{ invalid, describedBy }">
                  <AppTextInput
                    :id="`${instanceId}-model-alias-${render.index}`"
                    :model-value="render.item.alias"
                    appearance="surface"
                    size="compact"
                    :label="labels.aliasFor(render.item.id)"
                    :disabled="disabled"
                    :placeholder="labels.aliasPlaceholder"
                    :invalid="invalid"
                    :described-by="describedBy"
                    :data-alias-input-index="render.index"
                    :spellcheck="false"
                    @update:model-value="updateRow(render.index, { alias: $event })"
                    @blur="touchAlias(render.item.key)"
                  />
                </template>
              </CompactFieldError>
            </div>
          </div>

          <div class="ledger-record-list__cell model-alias-editor__route" role="cell">
            <span class="model-alias-editor__mobile-label">
              {{ labels.weight }} / {{ labels.priority }}
            </span>
            <div class="model-alias-editor__route-inputs">
              <CompactFieldError
                :id="`${instanceId}-model-weight-${render.index}`"
                class="model-alias-editor__route-field"
                :error="visibleWeightError(render.index)"
              >
                <template #default="{ invalid, describedBy }">
                  <AppTextInput
                    :id="`${instanceId}-model-weight-${render.index}`"
                    class="model-alias-editor__route-input"
                    :model-value="routeCountText(render.item.weight ?? null)"
                    appearance="surface"
                    size="compact"
                    monospace
                    :label="labels.weight"
                    placeholder="1"
                    :invalid="invalid"
                    :described-by="describedBy"
                    :data-model-weight-index="render.index"
                    :spellcheck="false"
                    :disabled="disabled"
                    @update:model-value="updateRouteCount(render.index, 'weight', $event)"
                  />
                </template>
              </CompactFieldError>
              <CompactFieldError
                :id="`${instanceId}-model-priority-${render.index}`"
                class="model-alias-editor__route-field"
                :error="visiblePriorityError(render.index)"
              >
                <template #default="{ invalid, describedBy }">
                  <AppTextInput
                    :id="`${instanceId}-model-priority-${render.index}`"
                    class="model-alias-editor__route-input"
                    :model-value="routeCountText(render.item.priority ?? null)"
                    appearance="surface"
                    size="compact"
                    monospace
                    :label="labels.priority"
                    placeholder="1"
                    :invalid="invalid"
                    :described-by="describedBy"
                    :data-model-priority-index="render.index"
                    :spellcheck="false"
                    :disabled="disabled"
                    @update:model-value="updateRouteCount(render.index, 'priority', $event)"
                  />
                </template>
              </CompactFieldError>
            </div>
            <div class="model-alias-editor__route-meta">
              <span
                v-if="(render.item.priority ?? 1) >= 2"
                class="model-alias-editor__route-flag"
                >{{ labels.priorityFallback }}</span
              >
              <span
                v-if="(render.item.weight ?? 1) === 0"
                class="model-alias-editor__route-flag model-alias-editor__route-flag--muted"
                >{{ labels.weightDisabled }}</span
              >
              <span v-else class="model-alias-editor__route-share">{{
                sharePercent(render.index)
              }}</span>
            </div>
          </div>

          <div class="ledger-record-list__cell model-alias-editor__third-column" role="cell">
            <span class="model-alias-editor__mobile-label">{{ labels.thirdColumn }}</span>
            <slot name="third-column" :item="render.item" :index="render.index" />
          </div>

          <div class="ledger-record-list__cell model-alias-editor__actions" role="cell">
            <IconButton
              variant="ghost"
              size="compact"
              :disabled="disabled"
              :label="labels.removeFor(render.item.id || labels.manualId)"
              @click="removeRow(render.index)"
            >
              <X :size="16" aria-hidden="true" />
            </IconButton>
          </div>
        </article>
      </template>

      <div
        v-if="visibleRowCount === 0"
        class="ledger-record-list__record model-alias-editor__empty"
        role="row"
        aria-rowindex="2"
      >
        <span class="ledger-record-list__cell" role="cell">
          {{ modelValue.length ? labels.noMatches : labels.empty }}
        </span>
      </div>
    </LedgerRecordList>

    <AppButton
      v-if="addable"
      class="model-alias-editor__add"
      variant="link"
      size="inline"
      :disabled="disabled || !createRow"
      @click="addManual"
    >
      <Plus :size="16" aria-hidden="true" />{{ labels.addInline }}
    </AppButton>
  </div>
</template>

<style scoped>
.model-alias-editor {
  min-width: 0;
}

.model-alias-editor__toolbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-3);
  padding: 15px 0 13px;
}

.model-alias-editor__search {
  display: grid;
  width: min(100%, 420px);
  min-width: 0;
  gap: 5px;
  color: var(--color-text-faint);
  font-size: var(--text-meta);
}

.model-alias-editor__count {
  flex: none;
  color: var(--color-text-faint);
  font-family: var(--font-mono);
  font-size: 10.8px;
}

.model-alias-editor__grid {
  --ledger-record-list-record-min-height: 58px;
  --ledger-record-list-record-padding: 9px 0;
  --ledger-record-list-grid: minmax(170px, 22fr) minmax(230px, 36fr) minmax(200px, 22fr)
    minmax(110px, 15fr) 40px;
  --ledger-record-list-column-gap: 16px;
}

.model-alias-editor__grid :deep(.ledger-record-list__header) {
  font-size: var(--text-label-xs);
  font-weight: 500;
}

.model-alias-editor__record--invalid {
  background: var(--color-danger-bg);
}

.model-alias-editor__record--nested {
  border-left: 2px solid var(--color-border-subtle);
  padding-left: 10px;
}

.model-alias-editor__record--disabled .model-alias-editor__id,
.model-alias-editor__record--disabled .model-alias-editor__alias-cell {
  opacity: 0.55;
}

.model-alias-editor__group-cell {
  grid-column: 1 / -1;
  display: grid;
  gap: 6px;
}

.model-alias-editor__group-toggle {
  display: inline-flex;
  width: fit-content;
  align-items: center;
  gap: 7px;
  padding: 2px 0;
  border: 0;
  background: none;
  color: var(--color-text);
  cursor: pointer;
  font: inherit;
  font-weight: 600;
}

.model-alias-editor__group-toggle:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}

.model-alias-editor__group-count {
  color: var(--color-text-faint);
  font-family: var(--font-mono);
  font-size: var(--text-meta);
  font-weight: 500;
}

.model-alias-editor__distribution {
  display: flex;
  width: min(100%, 360px);
  height: 6px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--color-border-subtle);
}

.model-alias-editor__distribution-segment {
  height: 100%;
  background: var(--color-action);
}

.model-alias-editor__distribution-segment--zero {
  background: var(--color-border-subtle);
}

.model-alias-editor__route {
  display: grid;
  min-width: 0;
  align-content: center;
  gap: 4px;
}

.model-alias-editor__route-inputs {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.model-alias-editor__route-field {
  width: min(96px, 100%);
}

.model-alias-editor__route-input {
  width: 100%;
}

.model-alias-editor__route-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 16px;
}

.model-alias-editor__route-flag {
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--color-warning-bg, var(--color-border-subtle));
  color: var(--color-text);
  font-size: var(--text-meta);
  font-weight: 560;
}

.model-alias-editor__route-flag--muted {
  background: var(--color-border-subtle);
  color: var(--color-text-faint);
}

.model-alias-editor__route-share {
  color: var(--color-text-faint);
  font-family: var(--font-mono);
  font-size: var(--text-meta);
}

.model-alias-editor__id,
.model-alias-editor__third-column {
  min-width: 0;
}

.model-alias-editor__id {
  display: grid;
  align-content: center;
  gap: 5px;
}

.model-alias-editor__id code {
  display: block;
  width: 100%;
  padding-inline-end: 38px;
  overflow-wrap: anywhere;
  font-size: var(--text-sm);
}

.model-alias-editor__id-field {
  display: flex;
  width: 100%;
  min-height: var(--control-sm);
  max-width: 300px;
  align-items: center;
  padding-left: 6px;
}

.model-alias-editor__id-field :deep(.app-text-input) {
  width: 100%;
}

.model-alias-editor__alias-cell {
  display: grid;
  gap: var(--space-1);
}

.model-alias-editor__alias-control {
  display: flex;
  min-height: var(--control-sm);
  align-items: center;
  gap: 9px;
}

.model-alias-editor__alias-toggle {
  display: grid;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  place-items: center;
  cursor: pointer;
}

.model-alias-editor__alias-toggle input {
  width: 16px;
  height: 16px;
  margin: 0;
  accent-color: var(--color-action);
  cursor: pointer;
}

.model-alias-editor__alias-toggle--disabled,
.model-alias-editor__alias-toggle input:disabled {
  cursor: not-allowed;
}

.model-alias-editor__alias-toggle input:disabled {
  opacity: 0.55;
}

.model-alias-editor__alias-field {
  width: min(100%, 300px);
  min-width: 0;
  flex: 1;
}

.model-alias-editor__actions {
  display: flex;
  justify-content: flex-end;
}

.model-alias-editor__actions :deep(.icon-button:hover:not(:disabled)) {
  border-color: var(--color-danger);
  color: var(--color-danger);
}

.model-alias-editor__mobile-label {
  display: none;
  color: var(--color-text-faint);
  font-size: var(--text-label-xs);
  font-weight: 560;
}

.model-alias-editor__empty {
  grid-template-columns: minmax(0, 1fr);
  min-height: 58px;
  color: var(--color-text-faint);
  font-size: var(--text-sm);
  text-align: center;
}

.model-alias-editor__empty .ledger-record-list__cell {
  grid-column: 1 / -1;
}

.model-alias-editor__add {
  width: fit-content;
  min-height: 36px;
  justify-content: flex-start;
  margin-top: 10px;
  padding: 4px 1px;
  font-size: var(--text-sm);
  font-weight: 600;
}

@media (max-width: 860px) {
  .model-alias-editor__grid {
    --ledger-record-list-card-grid: minmax(0, 0.7fr) minmax(0, 1.3fr);
  }

  .model-alias-editor__record {
    padding-right: 58px;
  }

  .model-alias-editor__id,
  .model-alias-editor__alias-cell,
  .model-alias-editor__route {
    grid-column: 1 / -1;
  }

  .model-alias-editor__id,
  .model-alias-editor__alias-cell,
  .model-alias-editor__route,
  .model-alias-editor__third-column {
    display: grid;
    align-content: start;
    gap: 5px;
  }

  .model-alias-editor__alias-cell {
    border-top: 1px solid var(--color-border-subtle);
    padding-top: 11px;
  }

  .model-alias-editor__actions {
    position: absolute;
    top: 4px;
    right: 4px;
  }

  .model-alias-editor__actions :deep(.icon-button) {
    width: var(--touch-target);
    height: var(--touch-target);
  }

  .model-alias-editor__mobile-label {
    display: inline;
  }

  .model-alias-editor__alias-toggle {
    width: var(--touch-target);
    height: var(--touch-target);
    flex-basis: var(--touch-target);
  }

  .model-alias-editor__id-field,
  .model-alias-editor__alias-field {
    max-width: none;
  }

  .model-alias-editor__id-field {
    min-height: var(--touch-target);
  }
}

@media (max-width: 640px) {
  .model-alias-editor__toolbar {
    align-items: stretch;
    flex-direction: column;
  }

  .model-alias-editor__search {
    width: 100%;
  }
}
</style>
