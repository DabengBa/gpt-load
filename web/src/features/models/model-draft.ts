import type { ModelPricingStatus } from '@/api/control/types'
import type { GroupModelUpdateDto } from '@/app/resources/groups'
import type { ModelCandidate, ModelCandidateSource } from '@/app/resources/providers'

export type ModelDraftKey = string | number

export interface ModelDraftValue extends GroupModelUpdateDto {
  key: ModelDraftKey
  editable_id?: boolean
  name: string
  sources: ModelCandidateSource[]
  pricing_status: ModelPricingStatus
  /** Price record ID for direct navigation to the price editor; absent when no record exists. */
  price_id?: number
}

export interface ModelNameConflict {
  client_model: string
  indexes: number[]
}

export interface ModelAliasEditorLabels {
  tableLabel: string
  id: string
  alias: string
  thirdColumn: string
  actions: string
  search: string
  searchLabel: string
  clearSearch: string
  aliasEnabledFor: (id: string) => string
  aliasFor: (id: string) => string
  aliasPlaceholder: string
  aliasRequired: string
  removeFor: (id: string) => string
  manualId: string
  manualIdRequired: string
  add: string
  addInline: string
  count: (count: number) => string
  empty: string
  noMatches: string
  nameConflict: (name: string) => string
  weight: string
  priority: string
  priorityFallback: string
  weightDisabled: string
  invalidWeight: string
  invalidPriority: string
  zeroShare: string
}

export interface ModelDiscoveryDrawerLabels {
  title: string
  description: string
  close: string
  loading: string
  search: string
  clearSearch: string
  filterLabel: string
  filterUnadded: string
  filterAll: string
  alreadyAdded: string
  unadded: string
  noMatches: string
  empty: string
  selected: (count: number) => string
  selectAll: string
  deselectAll: string
  retry: string
  cancel: string
  confirm: string
  pricingStatus: Record<ModelPricingStatus, string>
  pricingDiscovered: (source: string) => string
  sources: Record<ModelCandidateSource, string>
}

export function mergeCandidateMetadata<T extends ModelDraftValue>(
  draft: readonly T[],
  candidates: readonly ModelCandidate[],
): T[] {
  const byID = new Map(candidates.map((candidate) => [candidate.id, candidate] as const))
  return draft.map((item) => {
    const candidate = byID.get(item.id.trim())
    return candidate
      ? ({
          ...item,
          name: candidate.name,
          sources: [...candidate.sources],
          pricing_status: candidate.pricing_status,
        } as T)
      : ({ ...item, sources: [...item.sources] } as T)
  })
}

export function appendSelectedCandidates<T extends ModelDraftValue>(
  draft: readonly T[],
  selected: readonly ModelCandidate[],
  create: (candidate: ModelCandidate) => T,
): T[] {
  const result = mergeCandidateMetadata(draft, selected)
  const present = new Set(result.map(({ id }) => id.trim()).filter(Boolean))
  for (const candidate of selected) {
    if (present.has(candidate.id)) continue
    present.add(candidate.id)
    result.push(create(candidate))
  }
  return result
}

export function readModelNameConflicts(value: unknown): ModelNameConflict[] {
  if (typeof value !== 'object' || value === null || !('conflicts' in value)) return []
  const conflicts = (value as { conflicts?: unknown }).conflicts
  if (!Array.isArray(conflicts)) return []

  return conflicts.flatMap((item) => {
    if (
      typeof item !== 'object' ||
      item === null ||
      typeof (item as { client_model?: unknown }).client_model !== 'string' ||
      !Array.isArray((item as { indexes?: unknown }).indexes) ||
      !(item as { indexes: unknown[] }).indexes.every(
        (index) => typeof index === 'number' && Number.isSafeInteger(index) && index >= 0,
      )
    ) {
      return []
    }
    return [item as ModelNameConflict]
  })
}

export function normalizeModel(model: GroupModelUpdateDto): GroupModelUpdateDto | undefined {
  const id = model.id.trim()
  if (!id) return undefined
  const alias = model.alias_enabled ? model.alias.trim() : ''
  const normalized: GroupModelUpdateDto = {
    id,
    alias,
    alias_enabled: model.alias_enabled,
    weight: model.weight ?? null,
    priority: model.priority ?? null,
  }
  if (Object.prototype.hasOwnProperty.call(model, 'entry_id')) {
    normalized.entry_id = model.entry_id
  }
  if (Object.prototype.hasOwnProperty.call(model, 'circuit_breaker')) {
    normalized.circuit_breaker = model.circuit_breaker
  }
  return normalized
}

export function clientModel(model: GroupModelUpdateDto): string {
  const normalized = normalizeModel(model)
  return normalized === undefined ? '' : normalized.alias_enabled ? normalized.alias : normalized.id
}

/** Client names are intentionally exact and case sensitive, matching the API contract. */
export function findModelNameConflicts(
  models: readonly GroupModelUpdateDto[],
): ModelNameConflict[] {
  // 同一对外名可映射多个上游模型（设计 §3）；仅当同一 (对外名, 上游模型) 重复时冲突。
  const byPair = new Map<string, number[]>()
  const names = new Map<string, string>()
  for (const [index, model] of models.entries()) {
    const name = clientModel(model)
    if (!name) continue
    const pair = `${name}\u0000${model.id.trim()}`
    byPair.set(pair, [...(byPair.get(pair) ?? []), index])
    names.set(pair, name)
  }
  return [...byPair.entries()]
    .filter(([, indexes]) => indexes.length > 1)
    .map(([pair, indexes]) => ({ client_model: names.get(pair) ?? '', indexes }))
}

export function indexesWithConflicts(conflicts: readonly ModelNameConflict[]): Set<number> {
  return new Set(conflicts.flatMap((conflict) => conflict.indexes))
}

export function indexesWithEmptyAliases(models: readonly GroupModelUpdateDto[]): Set<number> {
  return new Set(
    models.flatMap((model, index) => (model.alias_enabled && !model.alias.trim() ? [index] : [])),
  )
}

export function indexesWithEmptyIDs(models: readonly GroupModelUpdateDto[]): Set<number> {
  return new Set(models.flatMap((model, index) => (!model.id.trim() ? [index] : [])))
}

/** 条目权重上限与后端 state.MaxWeight 对齐。 */
export const MODEL_ROUTE_MAX_WEIGHT = 100

function isInvalidRouteCount(value: number | null | undefined, minimum: number): boolean {
  return value !== null && value !== undefined && (!Number.isSafeInteger(value) || value < minimum)
}

export function indexesWithInvalidWeights(models: readonly GroupModelUpdateDto[]): Set<number> {
  return new Set(
    models.flatMap((model, index) =>
      isInvalidRouteCount(model.weight, 0) || (model.weight ?? 0) > MODEL_ROUTE_MAX_WEIGHT
        ? [index]
        : [],
    ),
  )
}

export function indexesWithInvalidPriorities(models: readonly GroupModelUpdateDto[]): Set<number> {
  return new Set(
    models.flatMap((model, index) => (isInvalidRouteCount(model.priority, 1) ? [index] : [])),
  )
}

export function effectiveEntryWeight(model: GroupModelUpdateDto): number {
  return model.weight ?? 1
}

/**
 * 每个条目在其对外名分组内的占比（null 权重按 1 参与，设计 §3）。
 * 权重合计为 0 的分组内所有条目占比均为 0。
 */
export function routeEntryShares(models: readonly GroupModelUpdateDto[]): number[] {
  const totals = new Map<string, number>()
  for (const model of models) {
    const name = clientModel(model)
    if (!name) continue
    totals.set(name, (totals.get(name) ?? 0) + effectiveEntryWeight(model))
  }
  return models.map((model) => {
    const name = clientModel(model)
    const total = name ? (totals.get(name) ?? 0) : 0
    return total > 0 ? effectiveEntryWeight(model) / total : 0
  })
}

/** 同一对外名下条目权重合计为 0（该对外模型完全不可分流，设计 §3 V2）。 */
export function indexesWithZeroShare(models: readonly GroupModelUpdateDto[]): Set<number> {
  const totals = new Map<string, number>()
  for (const model of models) {
    const name = clientModel(model)
    if (!name) continue
    totals.set(name, (totals.get(name) ?? 0) + effectiveEntryWeight(model))
  }
  const zeroModels = new Set(
    [...totals.entries()].filter(([, total]) => total === 0).map(([name]) => name),
  )
  return new Set(
    models.flatMap((model, index) => (zeroModels.has(clientModel(model)) ? [index] : [])),
  )
}

export function modelDraftValidity(
  models: readonly GroupModelUpdateDto[],
  conflicts: readonly ModelNameConflict[] = findModelNameConflicts(models),
): {
  conflictIndexes: Set<number>
  emptyIDIndexes: Set<number>
  emptyAliasIndexes: Set<number>
  invalidWeightIndexes: Set<number>
  invalidPriorityIndexes: Set<number>
  zeroShareIndexes: Set<number>
  invalidIndexes: Set<number>
} {
  const conflictIndexes = indexesWithConflicts(conflicts)
  const emptyIDIndexes = indexesWithEmptyIDs(models)
  const emptyAliasIndexes = indexesWithEmptyAliases(models)
  const invalidWeightIndexes = indexesWithInvalidWeights(models)
  const invalidPriorityIndexes = indexesWithInvalidPriorities(models)
  const zeroShareIndexes = indexesWithZeroShare(models)
  return {
    conflictIndexes,
    emptyIDIndexes,
    emptyAliasIndexes,
    invalidWeightIndexes,
    invalidPriorityIndexes,
    zeroShareIndexes,
    invalidIndexes: new Set([
      ...conflictIndexes,
      ...emptyIDIndexes,
      ...emptyAliasIndexes,
      ...invalidWeightIndexes,
      ...invalidPriorityIndexes,
      ...zeroShareIndexes,
    ]),
  }
}

export function createModelDraft<T extends GroupModelUpdateDto>(
  items: readonly T[],
): Array<T & { key: number }>
export function createModelDraft<T extends GroupModelUpdateDto, K extends ModelDraftKey>(
  items: readonly T[],
  createKey: (item: T, index: number) => K,
): Array<T & { key: K }>
export function createModelDraft<T extends GroupModelUpdateDto, K extends ModelDraftKey>(
  items: readonly T[],
  createKey?: (item: T, index: number) => K,
): Array<T & { key: K | number }> {
  return items.map((item, index) => ({
    ...item,
    key: createKey ? createKey(item, index) : index,
  }))
}

export function normalizedModels(draft: readonly GroupModelUpdateDto[]): GroupModelUpdateDto[] {
  return draft.flatMap((item) => {
    const normalized = normalizeModel(item)
    return normalized === undefined ? [] : [normalized]
  })
}

export function sameModels(
  left: readonly GroupModelUpdateDto[],
  right: readonly GroupModelUpdateDto[],
): boolean {
  return JSON.stringify(normalizedModels(left)) === JSON.stringify(normalizedModels(right))
}
