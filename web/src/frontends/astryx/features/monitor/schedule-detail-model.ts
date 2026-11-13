import type {
  ModelRouteScheduleDetailDto,
  ModelRouteScheduleEntryDto,
  ModelRouteScheduleGroupDto,
  ModelRouteSchedulePatchUpdate,
} from '@shared/control/resources/model-route-schedule'
import type { ScheduleDraft, ScheduleDrafts } from '@shared/routing/monitor-route'
import { pagePath } from '@shared/routing/page-routes'

export type EditableField = 'weight' | 'priority' | 'concurrency_limit'
export type Draft = ScheduleDraft

export interface ScheduleRow {
  readonly group: ModelRouteScheduleGroupDto
  readonly entry: ModelRouteScheduleEntryDto
}

export type DetailKey = readonly [revision: number, externalModel: string | null, protocol: string]

export const EMPTY_DETAIL_KEY: DetailKey = [0, '', '']

// Use the saved, unrounded share rather than the editable preview: drafts must
// not move an input while it is being edited. Retain the existing key order for ties.
export function scheduleRows(detail: ModelRouteScheduleDetailDto | undefined): ScheduleRow[] {
  return (detail?.groups ?? [])
    .flatMap((group) => group.entries.map((entry) => ({ group, entry })))
    .sort(
      (a, b) =>
        a.entry.priority - b.entry.priority ||
        (b.group.enabled && b.entry.enabled ? b.entry.configured_share : 0) -
          (a.group.enabled && a.entry.enabled ? a.entry.configured_share : 0) ||
        a.group.group_id - b.group.group_id ||
        (a.entry.entry_id < b.entry.entry_id ? -1 : a.entry.entry_id > b.entry.entry_id ? 1 : 0),
    )
}

export function rowKey(groupID: number, entryID: string): string {
  return `${groupID}:${entryID}`
}

export function draftKey(groupID: number, entryID: string): string {
  return rowKey(groupID, entryID)
}

export function fieldKey(groupID: number, entryID: string, field: EditableField): string {
  return `${draftKey(groupID, entryID)}\u0000${field}`
}

export function hasOwn(source: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(source, key)
}

export function configuredValue(
  entry: ModelRouteScheduleEntryDto,
  field: EditableField,
): number | null {
  if (field === 'concurrency_limit') return entry.concurrency_limit
  if (field === 'weight') return null
  return entry.priority === 1 ? null : entry.priority
}

export function effectiveValue(entry: ModelRouteScheduleEntryDto, field: EditableField): number {
  return entry[field]
}

// Accepted numeric domain per editable field. Rendering and validation read this
// one table so the input bounds can never drift from what save() accepts.
export const FIELD_BOUNDS: Readonly<
  Record<EditableField, { readonly min: number; readonly max?: number }>
> = {
  concurrency_limit: { min: 0, max: 2147483647 },
  priority: { min: 1 },
  weight: { min: 0, max: 100 },
}

export function isValidValue(field: EditableField, value: number): boolean {
  if (!Number.isSafeInteger(value)) return false
  const { min, max } = FIELD_BOUNDS[field]
  return value >= min && (max === undefined || value <= max)
}

// Row-level numeric controls, in the order the settings cell renders them.
export const ROW_NUMBER_FIELDS = ['priority', 'weight'] as const

export function draftFingerprint(source: ScheduleDrafts): string {
  return JSON.stringify(
    Object.keys(source)
      .sort()
      .map((key) => [
        key,
        Object.entries(source[key] ?? {}).sort(([a], [b]) => a.localeCompare(b)),
      ]),
  )
}

export function cloneDrafts(source: ScheduleDrafts): ScheduleDrafts {
  return Object.fromEntries(
    Object.entries(source).map(([key, draft]) => [key, draft === undefined ? {} : { ...draft }]),
  )
}

// hydrateDraftState: resets the local edit surface, then re-populates drafts
// for keys that still resolve to rendered rows; raw inputs mirror the draft
// values so a URL-decode round trip keeps the same visible text.
export function buildHydration(
  rows: ScheduleRow[],
  source: ScheduleDrafts,
): { draftMap: Record<string, Draft>; rawInputs: Record<string, string> } {
  const rowsByKey = new Map(
    rows.map(({ group, entry }) => [draftKey(group.group_id, entry.entry_id), { group, entry }]),
  )
  const nextDraftMap: Record<string, Draft> = {}
  const nextRawInputs: Record<string, string> = {}
  for (const [key, draft] of Object.entries(source)) {
    const row = rowsByKey.get(key)
    if (row === undefined) continue
    const next = { ...draft }
    nextDraftMap[key] = next
    for (const field of ['weight', 'priority', 'concurrency_limit'] as const) {
      if (!hasOwn(next, field)) continue
      const value = next[field]
      nextRawInputs[fieldKey(row.group.group_id, row.entry.entry_id, field)] =
        value === null || value === undefined ? '' : String(value)
    }
  }
  return { draftMap: nextDraftMap, rawInputs: nextRawInputs }
}

export function buildPatchUpdates(
  detail: ModelRouteScheduleDetailDto | undefined,
  draftMap: Record<string, Draft>,
): ModelRouteSchedulePatchUpdate[] {
  const result: ModelRouteSchedulePatchUpdate[] = []
  for (const group of detail?.groups ?? []) {
    for (const entry of group.entries) {
      const key = draftKey(group.group_id, entry.entry_id)
      const draft = draftMap[key]
      if (draft === undefined || entry.entry_id.startsWith('derived:')) {
        continue
      }
      const update: ModelRouteSchedulePatchUpdate = {
        group_id: group.group_id,
        entry_id: entry.entry_id,
      }
      if (draft !== undefined && hasOwn(draft, 'weight')) update.weight = draft.weight
      if (draft !== undefined && hasOwn(draft, 'priority')) update.priority = draft.priority
      if (draft?.concurrency_limit !== undefined && draft.concurrency_limit !== null) {
        update.concurrency_limit = draft.concurrency_limit
      }
      if (hasOwn(draft, 'reasoning_effort')) {
        update.reasoning_effort = draft.reasoning_effort ?? null
      }
      if (Object.keys(update).length > 2) result.push(update)
    }
  }
  return result
}

export function groupDetailHref(id: number): string {
  return `${pagePath('groups')}/${id}`
}

export function revealScheduleRow(container: HTMLElement | null, key: string): void {
  if (container === null) return
  container
    .querySelector(`[data-row-key="${CSS.escape(key)}"]`)
    ?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
}

export function withSetValue<T>(set: ReadonlySet<T>, value: T): ReadonlySet<T> {
  return new Set(set).add(value)
}

export function withoutSetValue<T>(set: ReadonlySet<T>, value: T): ReadonlySet<T> {
  if (!set.has(value)) return set
  const next = new Set(set)
  next.delete(value)
  return next
}

export function withMapValue<K, V>(map: ReadonlyMap<K, V>, key: K, value: V): ReadonlyMap<K, V> {
  return new Map(map).set(key, value)
}

export function withoutMapValue<K, V>(map: ReadonlyMap<K, V>, key: K): ReadonlyMap<K, V> {
  if (!map.has(key)) return map
  const next = new Map(map)
  next.delete(key)
  return next
}

// Classic `await nextTick()` after the mutation: give the invalidation-driven
// parent refresh a macrotask to land before the preserve markers clear.
export function waitNextTick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0))
}
