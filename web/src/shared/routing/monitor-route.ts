import {
  defaultUsageBreakdownSort,
  defaultUsageBreakdownSortDirectionFor,
  normalizeUsageBreakdownSort,
  normalizeUsageBreakdownSortDirection,
  type UsageFilters,
} from '@shared/control/resources/usage'
import { defaultTimeRange } from '@shared/lib/time'
import {
  normalizeUsageGroupID,
  normalizeUsageModel,
  normalizeUsagePage,
  normalizeUsagePageSize,
  parseAppliedUsageFilters,
} from '@shared/domain/monitor/usage-filters'
import { normalizeMonitorText } from '@shared/domain/monitor/filter-validation'

import type { SharedRouteQuery, SharedRouteQueryRaw } from './route-query'

// Framework-free port of the classic monitor route codec. /monitor is the
// usage & cost surface; /schedule reuses the schedule draft surface of the
// same query space.

export type ScheduleDraftField = 'weight' | 'priority'
export type ScheduleDrafts = Record<string, Partial<Record<ScheduleDraftField, number | null>>>

export interface ScheduleMonitorState {
  externalModel?: string
  selectedRow?: string
  /** 来源分组行定位提示（分组模型页跳转）；详情加载后解析为 selectedRow。 */
  sourceGroupId?: number
  /** 价格抽屉深链接（正整数 price id）；access-key 分支清除。 */
  selectedPriceID?: number
  drafts: ScheduleDrafts
}

export function normalizeMonitorQuery(query: SharedRouteQuery): SharedRouteQueryRaw {
  return usageMonitorQuery(parseAppliedUsageFilters(query))
}

export function scopeAccessKeyUsageFilters(filters: UsageFilters): UsageFilters {
  const scoped = { ...filters }
  delete scoped.group_id
  if (scoped.breakdown_sort === 'group' || scoped.breakdown_sort === 'channel') {
    scoped.breakdown_sort = 'model'
    scoped.breakdown_sort_direction = 'asc'
  }
  return scoped
}

export function normalizeAccessKeyMonitorQuery(query: SharedRouteQuery): SharedRouteQueryRaw {
  return usageMonitorQuery(scopeAccessKeyUsageFilters(parseAppliedUsageFilters(query)))
}

const scheduleRowPattern = /^\d+:\S{1,512}$/u

export function parseScheduleMonitorState(query: SharedRouteQuery): ScheduleMonitorState {
  return {
    externalModel: scalarText(query.schedule_model),
    selectedRow: scalarScheduleRow(query.schedule_row),
    sourceGroupId: scalarPositiveNumber(query.schedule_group),
    selectedPriceID: scalarPositiveNumber(query.selected_price_id),
    drafts: parseScheduleDrafts(query.schedule_draft),
  }
}

// access-key 只读分支：清除价格与调度上下文，保留只读模型选择与草稿容器。
export function scopeAccessKeyScheduleMonitorState(
  state: ScheduleMonitorState,
): ScheduleMonitorState {
  return {
    externalModel: state.externalModel,
    drafts: {},
  }
}

// /schedule 页面用路径表达页面身份，query 只保留调度上下文与草稿。
export function scheduleMonitorQuery(state: ScheduleMonitorState): SharedRouteQueryRaw {
  const normalized: SharedRouteQueryRaw = {}
  const model = scalarText(state.externalModel)
  if (model !== undefined) normalized.schedule_model = model
  if (state.selectedRow !== undefined) normalized.schedule_row = state.selectedRow
  if (state.sourceGroupId !== undefined) normalized.schedule_group = String(state.sourceGroupId)
  if (state.selectedPriceID !== undefined) {
    normalized.selected_price_id = String(state.selectedPriceID)
  }
  const drafts = serializeScheduleDrafts(state.drafts)
  if (drafts !== undefined) normalized.schedule_draft = drafts
  return normalized
}

export function usageMonitorQuery(
  filters: UsageFilters = { range: defaultTimeRange },
): SharedRouteQueryRaw {
  const normalized: SharedRouteQueryRaw = {
    range: filters.range,
  }
  const groupID = normalizeUsageGroupID(filters.group_id)
  const upstreamModel = normalizeUsageModel(filters.upstream_model)
  if (groupID !== undefined) normalized.group_id = String(groupID)
  if (upstreamModel !== undefined) normalized.upstream_model = upstreamModel
  const breakdownPage = normalizeUsagePage(filters.breakdown_page)
  const breakdownPageSize = normalizeUsagePageSize(filters.breakdown_page_size)
  if (breakdownPage !== 1) normalized.breakdown_page = String(breakdownPage)
  if (breakdownPageSize !== 20) normalized.breakdown_page_size = String(breakdownPageSize)
  const rawBreakdownSort = filters.breakdown_sort
  const breakdownSort = normalizeUsageBreakdownSort(rawBreakdownSort)
  const breakdownSortDirection =
    rawBreakdownSort === breakdownSort
      ? normalizeUsageBreakdownSortDirection(filters.breakdown_sort_direction, breakdownSort)
      : defaultUsageBreakdownSortDirectionFor(breakdownSort)
  if (breakdownSort !== defaultUsageBreakdownSort) {
    normalized.breakdown_sort = breakdownSort
  }
  if (breakdownSortDirection !== defaultUsageBreakdownSortDirectionFor(breakdownSort)) {
    normalized.breakdown_sort_direction = breakdownSortDirection
  }
  return normalized
}

export function sameMonitorQuery(left: SharedRouteQuery, right: SharedRouteQueryRaw): boolean {
  const leftKeys = Object.keys(left)
  const rightKeys = Object.keys(right)
  if (leftKeys.length !== rightKeys.length) return false

  return leftKeys.every(
    (key) =>
      Object.prototype.hasOwnProperty.call(right, key) &&
      typeof left[key] === 'string' &&
      left[key] === right[key],
  )
}

function scalarText(raw: unknown): string | undefined {
  return normalizeMonitorText(raw)
}

function scalarPositiveNumber(raw: unknown): number | undefined {
  if (typeof raw !== 'string' || !/^\d+$/.test(raw)) return undefined
  const value = Number(raw)
  return Number.isSafeInteger(value) && value > 0 ? value : undefined
}

function scalarScheduleRow(raw: unknown): string | undefined {
  return typeof raw === 'string' && scheduleRowPattern.test(raw) ? raw : undefined
}

function parseScheduleDrafts(raw: unknown): ScheduleDrafts {
  if (typeof raw !== 'string' || raw.length > 20_000) return {}
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return {}
    const drafts: ScheduleDrafts = {}
    for (const [row, draft] of Object.entries(value)) {
      if (!scheduleRowPattern.test(row) || typeof draft !== 'object' || draft === null) continue
      const next: Partial<Record<ScheduleDraftField, number | null>> = {}
      for (const field of ['weight', 'priority'] as const) {
        const candidate = (draft as Record<string, unknown>)[field]
        if (candidate === null) {
          next[field] = null
        } else if (
          typeof candidate === 'number' &&
          Number.isSafeInteger(candidate) &&
          (field === 'weight' ? candidate >= 0 && candidate <= 100 : candidate >= 1)
        ) {
          next[field] = candidate
        }
      }
      if (Object.keys(next).length > 0) drafts[row] = next
    }
    return drafts
  } catch {
    return {}
  }
}

function serializeScheduleDrafts(drafts: ScheduleDrafts): string | undefined {
  const normalized: ScheduleDrafts = {}
  for (const row of Object.keys(drafts).sort()) {
    const draft = drafts[row]
    if (!scheduleRowPattern.test(row) || draft === undefined) continue
    const next: Partial<Record<ScheduleDraftField, number | null>> = {}
    for (const field of ['weight', 'priority'] as const) {
      const value = draft[field]
      if (value === null || (typeof value === 'number' && Number.isSafeInteger(value))) {
        next[field] = value
      }
    }
    if (Object.keys(next).length > 0) normalized[row] = next
  }
  return Object.keys(normalized).length > 0 ? JSON.stringify(normalized) : undefined
}
