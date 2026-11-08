import type { GroupModelItemDto } from '@shared/control/types'
import type { ModelCandidate } from '@shared/control/resources/providers'
import {
  findModelNameConflicts,
  normalizedModels as normalizeSharedModels,
  sameModels as sameSharedModels,
  type ModelDraftItem,
  type ModelNameConflict,
} from '@shared/domain/models/model-draft'

export type { ModelDraftItem } from '@shared/domain/models/model-draft'

export type ModelSyncMode = 'cleanup' | 'add' | 'full'

export interface ModelSyncDiff {
  additions: ModelCandidate[]
  removals: ModelDraftItem[]
}

export type { ModelNameConflict }
export { findModelNameConflicts }

export function createModelDraft(items: readonly GroupModelItemDto[]): ModelDraftItem[] {
  return items.map((item, index) => {
    const draft: ModelDraftItem = {
      id: item.id,
      name: item.id,
      sources: [],
      alias: item.alias,
      alias_enabled: item.alias_enabled,
      test_alias: item.test_alias,
      weight: item.weight ?? null,
      priority: item.priority ?? null,
      concurrency_limit: item.concurrency_limit,
      pricing_status: item.pricing_status,
      price_id: item.price_id,
      reasoning_effort: item.reasoning_effort ?? '',
      enabled: item.enabled ?? true,
      key: index,
    }

    if (Object.prototype.hasOwnProperty.call(item, 'entry_id')) {
      draft.entry_id = item.entry_id
    }
    if (Object.prototype.hasOwnProperty.call(item, 'circuit_breaker')) {
      draft.circuit_breaker = item.circuit_breaker
    }
    return draft
  })
}

/** Three-way reconcile; competing edits require explicit review rather than a retry. */
export function reconcileUncertainModels(
  baseline: readonly ModelDraftItem[],
  draft: readonly ModelDraftItem[],
  remote: readonly ModelDraftItem[],
  operation: 'sync' | 'save',
): { saved: ModelDraftItem[]; draft: ModelDraftItem[]; conflict: boolean } {
  if (operation === 'sync')
    return { saved: [...remote], draft: remote.map((item) => ({ ...item })), conflict: false }
  const previousEntries = new Set(baseline.map((item) => item.entry_id).filter(Boolean))
  const candidates = remote.filter((item) => item.entry_id && !previousEntries.has(item.entry_id))
  let conflict = false
  const keys = new Map<string, number>()
  for (const item of draft) {
    if (item.entry_id) continue
    const matches = candidates.filter(
      (candidate) => candidate.id === item.id && candidate.alias === item.alias,
    )
    const competing = draft.filter(
      (other) => !other.entry_id && other.id === item.id && other.alias === item.alias,
    )
    if (matches.length === 1 && competing.length === 1 && matches[0].entry_id) {
      keys.set(matches[0].entry_id, item.key)
      if (
        !sameModels(
          [
            {
              ...item,
              entry_id: matches[0].entry_id,
              concurrency_limit: item.concurrency_limit ?? 0,
              circuit_breaker: item.circuit_breaker ?? null,
            },
          ],
          [
            {
              ...matches[0],
              concurrency_limit: matches[0].concurrency_limit ?? 0,
              circuit_breaker: matches[0].circuit_breaker ?? null,
            },
          ],
        )
      )
        conflict = true
    } else conflict = true
  }
  const saved = remote.map((item) => ({
    ...item,
    key: item.entry_id ? (keys.get(item.entry_id) ?? item.key) : item.key,
  }))
  const remoteByKey = new Map(saved.map((item) => [item.key, item]))
  const originalByKey = new Map(baseline.map((item) => [item.key, item]))
  const draftByKey = new Map(draft.map((item) => [item.key, item]))
  const originalOrder = baseline.filter((item) => remoteByKey.has(item.key)).map((item) => item.key)
  const remoteOrder = saved.filter((item) => originalByKey.has(item.key)).map((item) => item.key)
  if (JSON.stringify(originalOrder) !== JSON.stringify(remoteOrder)) conflict = true
  const fields = [
    'id',
    'alias',
    'alias_enabled',
    'weight',
    'priority',
    'concurrency_limit',
    'circuit_breaker',
    'reasoning_effort',
    'enabled',
  ] as const
  const equal = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null)
  function mergeField<K extends (typeof fields)[number]>(
    out: ModelDraftItem,
    original: ModelDraftItem,
    local: ModelDraftItem,
    fresh: ModelDraftItem,
    field: K,
  ): void {
    if (!equal(local[field], original[field])) {
      if (!equal(fresh[field], original[field]) && !equal(fresh[field], local[field]))
        conflict = true
      out[field] = local[field]
    }
  }
  const next: ModelDraftItem[] = []
  for (const item of draft) {
    const original = originalByKey.get(item.key)
    const fresh = remoteByKey.get(item.key)
    if (!original) {
      next.push(
        fresh
          ? {
              ...item,
              entry_id: fresh.entry_id,
              concurrency_limit: item.concurrency_limit ?? fresh.concurrency_limit,
              circuit_breaker: item.circuit_breaker ?? fresh.circuit_breaker,
            }
          : { ...item },
      )
      continue
    }
    if (!fresh) {
      if (!sameModels([original], [item])) {
        conflict = true
        next.push({ ...item })
      }
      continue
    }
    const merged = { ...fresh }
    for (const field of fields) mergeField(merged, original, item, fresh, field)
    next.push(merged)
  }
  for (const original of baseline) {
    if (draftByKey.has(original.key)) continue
    const fresh = remoteByKey.get(original.key)
    if (fresh && !sameModels([fresh], [original])) conflict = true
  }
  for (const fresh of saved) {
    if (!originalByKey.has(fresh.key) && !draftByKey.has(fresh.key)) next.push({ ...fresh })
  }
  return { saved, draft: next, conflict }
}

export function normalizedModels(
  draft: readonly ModelDraftItem[],
  baseline?: readonly ModelDraftItem[],
) {
  if (!baseline) return normalizeSharedModels(draft)
  const byKey = new Map(baseline.map((item) => [item.key, item]))
  return draft.flatMap((item) => {
    const normalized = normalizeSharedModels([item])[0]
    if (!normalized) return []
    const original = byKey.get(item.key)
    if (original) {
      if ((item.weight ?? null) === (original.weight ?? null)) delete normalized.weight
      if ((item.priority ?? null) === (original.priority ?? null)) delete normalized.priority
      if (item.concurrency_limit === original.concurrency_limit) delete normalized.concurrency_limit
      if (JSON.stringify(item.circuit_breaker) === JSON.stringify(original.circuit_breaker))
        delete normalized.circuit_breaker
    }
    const reasoning = item.reasoning_effort ?? ''
    if (reasoning !== (original?.reasoning_effort ?? ''))
      normalized.reasoning_effort = reasoning || null
    const enabled = item.enabled ?? true
    if (enabled !== (original?.enabled ?? true)) normalized.enabled = enabled
    return [normalized]
  })
}

export function sameModels(
  left: readonly ModelDraftItem[],
  right: readonly ModelDraftItem[],
): boolean {
  return (
    sameSharedModels(left, right) &&
    left.length === right.length &&
    left.every(
      (item, index) =>
        (item.reasoning_effort ?? '') === (right[index]?.reasoning_effort ?? '') &&
        (item.enabled ?? true) === (right[index]?.enabled ?? true),
    )
  )
}

export function createModelSyncDiff(
  current: readonly ModelDraftItem[],
  candidates: readonly ModelCandidate[],
): ModelSyncDiff {
  const liveCandidates = candidates.filter(({ sources }) => sources.includes('live'))
  const liveIDs = new Set(liveCandidates.map(({ id }) => id))
  const currentIDs = new Set(current.map(({ id }) => id.trim()).filter(Boolean))
  return {
    additions: liveCandidates.filter(({ id }) => !currentIDs.has(id)),
    removals: current.filter(({ id }) => !liveIDs.has(id.trim())),
  }
}

export function syncedModels(
  current: readonly ModelDraftItem[],
  diff: ModelSyncDiff,
  mode: ModelSyncMode,
) {
  const removalKeys = new Set(diff.removals.map(({ key }) => key))
  const retained = mode === 'add' ? current : current.filter(({ key }) => !removalKeys.has(key))
  const additions =
    mode === 'cleanup'
      ? []
      : diff.additions.map(({ id }) => ({
          id,
          alias: '',
          alias_enabled: false,
          weight: null,
          priority: null,
        }))
  return [...normalizedModels(retained, current), ...normalizeSharedModels(additions)]
}
