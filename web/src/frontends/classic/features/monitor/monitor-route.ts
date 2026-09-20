import type { LocationQueryRaw } from 'vue-router'

import { enabledDataProtocols } from '@/api/control/protocols'
import {
  defaultUsageBreakdownSort,
  defaultUsageBreakdownSortDirectionFor,
  normalizeUsageBreakdownSort,
  normalizeUsageBreakdownSortDirection,
  type UsageFilters,
} from '@/app/resources/usage'
import { defaultTimeRange } from '@/lib/time'

import {
  normalizeUsageGroupID,
  normalizeUsageChannelID,
  normalizeUsageModel,
  normalizeUsagePage,
  normalizeUsagePageSize,
  parseAppliedUsageFilters,
} from './usage-filters'
import { normalizeMonitorText } from './filter-validation'

export type MonitorTab = 'health' | 'inspector' | 'usage'
export interface HealthMonitorState {
  groupsExpanded: boolean
}

export type UsageTrendMetric = 'tokens' | 'cost'

export interface UsageMonitorState {
  filtersOpen: boolean
  seriesExpanded: boolean
  metric: UsageTrendMetric
}

export type ScheduleMode = 'all' | 'primary' | 'fallback'
export type ScheduleDraftField = 'weight' | 'priority'
export type ScheduleDrafts = Record<string, Partial<Record<ScheduleDraftField, number | null>>>

export interface ScheduleMonitorState {
  mode: ScheduleMode
  externalModel?: string
  selectedRow?: string
  drafts: ScheduleDrafts
}

export interface InspectorMonitorState {
  protocol?: string
  externalModel?: string
  accessKeyID?: string
  run: boolean
  expandedGroupIDs: number[]
}

export function normalizeMonitorTab(raw: unknown): MonitorTab {
  return raw === 'inspector' || raw === 'usage' || raw === 'health' ? raw : 'health'
}

export function normalizeMonitorQuery(query: Record<string, unknown>): LocationQueryRaw {
  const tab = normalizeMonitorTab(query.tab)
  if (tab === 'health') return healthMonitorQuery(parseHealthMonitorState(query))
  if (tab === 'inspector') return inspectorMonitorQuery(parseInspectorMonitorState(query))
  if (tab === 'usage') {
    return usageMonitorQuery(parseAppliedUsageFilters(query), parseUsageMonitorState(query))
  }
  return healthMonitorQuery(parseHealthMonitorState(query))
}

export function scopeAccessKeyUsageFilters(filters: UsageFilters): UsageFilters {
  const scoped = { ...filters }
  delete scoped.group_id
  delete scoped.channel_id
  delete scoped.credential_id
  if (scoped.breakdown_sort === 'group' || scoped.breakdown_sort === 'channel') {
    scoped.breakdown_sort = 'model'
    scoped.breakdown_sort_direction = 'asc'
  }
  return scoped
}

export function normalizeAccessKeyMonitorQuery(query: Record<string, unknown>): LocationQueryRaw {
  if (query.tab === 'logs') return usageMonitorQuery()
  return usageMonitorQuery(
    scopeAccessKeyUsageFilters(parseAppliedUsageFilters(query)),
    parseUsageMonitorState(query),
  )
}

const scheduleRowPattern = /^\d+:\S{1,512}$/u

export function parseScheduleMonitorState(query: Record<string, unknown>): ScheduleMonitorState {
  return {
    mode:
      query.schedule_mode === 'primary' || query.schedule_mode === 'fallback'
        ? query.schedule_mode
        : 'all',
    externalModel: scalarText(query.schedule_model),
    selectedRow: scalarScheduleRow(query.schedule_row),
    drafts: parseScheduleDrafts(query.schedule_draft),
  }
}

// /schedule 页面用路径表达页面身份，query 只保留调度上下文与草稿。
export function scheduleMonitorQuery(state: ScheduleMonitorState): LocationQueryRaw {
  const normalized: LocationQueryRaw = {}
  if (state.mode !== 'all') normalized.schedule_mode = state.mode
  const model = scalarText(state.externalModel)
  if (model !== undefined) normalized.schedule_model = model
  if (state.selectedRow !== undefined) normalized.schedule_row = state.selectedRow
  const drafts = serializeScheduleDrafts(state.drafts)
  if (drafts !== undefined) normalized.schedule_draft = drafts
  return normalized
}

export function parseHealthMonitorState(query: Record<string, unknown>): HealthMonitorState {
  return { groupsExpanded: query.groups === 'expanded' }
}

export function healthMonitorQuery(state: HealthMonitorState): LocationQueryRaw {
  return state.groupsExpanded ? { tab: 'health', groups: 'expanded' } : { tab: 'health' }
}

export function usageMonitorQuery(
  filters: UsageFilters = { range: defaultTimeRange },
  state: UsageMonitorState = {
    filtersOpen: false,
    seriesExpanded: false,
    metric: 'tokens',
  },
): LocationQueryRaw {
  const normalized: LocationQueryRaw = {
    tab: 'usage',
    range: filters.range,
    metric: normalizeUsageTrendMetric(state.metric),
  }
  const groupID = normalizeUsageGroupID(filters.group_id)
  const channelID = normalizeUsageChannelID(filters.channel_id)
  const credentialID = normalizeUsageGroupID(filters.credential_id)
  const upstreamModel = normalizeUsageModel(filters.upstream_model)
  if (groupID !== undefined) normalized.group_id = String(groupID)
  if (channelID !== undefined) normalized.channel_id = channelID
  if (credentialID !== undefined) normalized.credential_id = String(credentialID)
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
  if (state.filtersOpen) normalized.panel = 'filters'
  if (state.seriesExpanded) normalized.series = 'expanded'
  return normalized
}

export function parseUsageMonitorState(query: Record<string, unknown>): UsageMonitorState {
  return {
    filtersOpen: query.panel === 'filters',
    seriesExpanded: query.series === 'expanded',
    metric: normalizeUsageTrendMetric(query.metric),
  }
}

function normalizeUsageTrendMetric(raw: unknown): UsageTrendMetric {
  return raw === 'tokens' || raw === 'cost' ? raw : 'tokens'
}

export function sameMonitorQuery(left: LocationQueryRaw, right: LocationQueryRaw): boolean {
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

export function parseInspectorMonitorState(query: Record<string, unknown>): InspectorMonitorState {
  const protocol = scalarEnum(query.protocol, enabledDataProtocols)
  const externalModel = scalarText(query.external_model)
  const accessKeyID = scalarPositiveID(query.access_key_id)

  return {
    protocol,
    externalModel,
    accessKeyID,
    run:
      query.run === '1' &&
      protocol !== undefined &&
      externalModel !== undefined &&
      accessKeyID !== undefined,
    expandedGroupIDs: parsePositiveIDList(query.expanded_groups),
  }
}

export function inspectorMonitorQuery(state: InspectorMonitorState): LocationQueryRaw {
  const normalized: LocationQueryRaw = { tab: 'inspector' }

  if (state.protocol !== undefined) normalized.protocol = state.protocol
  if (state.externalModel !== undefined) normalized.external_model = state.externalModel
  if (state.accessKeyID !== undefined) normalized.access_key_id = state.accessKeyID
  if (
    state.run &&
    state.protocol !== undefined &&
    state.externalModel !== undefined &&
    state.accessKeyID !== undefined
  ) {
    normalized.run = '1'
  }
  const expandedGroups = serializePositiveIDList(state.expandedGroupIDs)
  if (expandedGroups !== undefined) normalized.expanded_groups = expandedGroups
  return normalized
}

function scalarText(raw: unknown): string | undefined {
  return normalizeMonitorText(raw)
}

function scalarPositiveID(raw: unknown): string | undefined {
  if (typeof raw !== 'string' || !/^\d+$/.test(raw)) return undefined
  const value = Number(raw)
  return Number.isSafeInteger(value) && value > 0 ? String(value) : undefined
}

function scalarEnum<T extends string>(raw: unknown, values: readonly T[]): T | undefined {
  return typeof raw === 'string' && values.includes(raw as T) ? (raw as T) : undefined
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

function parsePositiveIDList(raw: unknown): number[] {
  if (typeof raw !== 'string' || raw === '') return []
  const values = raw.split(',').map(Number)
  if (
    values.some((value) => !Number.isSafeInteger(value) || value <= 0) ||
    new Set(values).size !== values.length
  ) {
    return []
  }
  return [...values].sort((left, right) => left - right)
}

function serializePositiveIDList(values: readonly number[]): string | undefined {
  const normalized = [...new Set(values)]
    .filter((value) => Number.isSafeInteger(value) && value > 0)
    .sort((left, right) => left - right)
  return normalized.length > 0 ? normalized.join(',') : undefined
}
