import type { LocationQueryRaw } from 'vue-router'

import type { RequestLogFilters } from '@/app/resources/request-logs'

import {
  parseAppliedLogFilterState,
  parseAppliedLogFilters,
  serializeAppliedLogFilters,
} from '../monitor/log-filters'

export interface LogsMonitorState {
  filtersOpen: boolean
  cursorHistory: string[]
  selectedRequestID?: string
  invalidAffinityKey?: string
}

const requestIDPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const logCursorPattern = /^[A-Za-z0-9_-]{1,512}$/u
const accessKeyForbiddenLogFilters: readonly (keyof RequestLogFilters)[] = [
  'group_id',
  'channel_id',
  'credential_id',
  'upstream_model',
  'access_key_id',
  'attempt_status_code',
  'failure_category',
  'error_code',
  'retry_state',
  'retry_count_min',
  'retry_count_max',
  'affinity_key',
]

export function scopeAccessKeyLogFilters(filters: RequestLogFilters): RequestLogFilters {
  const scoped = { ...filters }
  for (const field of accessKeyForbiddenLogFilters) delete scoped[field]
  return scoped
}

export function parseLogsMonitorState(query: Record<string, unknown>): LogsMonitorState {
  return {
    filtersOpen: query.panel === 'filters',
    cursorHistory: parseLogCursorHistory(query.log_cursors),
    selectedRequestID: parseSelectedRequestID(query),
    invalidAffinityKey: parseAppliedLogFilterState(query).invalidAffinityKey,
  }
}

export function logsMonitorQuery(
  filters: ReturnType<typeof parseAppliedLogFilters>,
  state: LogsMonitorState = { filtersOpen: false, cursorHistory: [] },
): LocationQueryRaw {
  const normalized: LocationQueryRaw = { ...serializeAppliedLogFilters(filters) }
  if (state.filtersOpen) normalized.panel = 'filters'
  const cursorHistory = serializeLogCursorHistory(state.cursorHistory)
  if (cursorHistory !== undefined) normalized.log_cursors = cursorHistory
  if (state.selectedRequestID !== undefined) {
    normalized.selected_request_id = state.selectedRequestID
  }
  if (state.invalidAffinityKey !== undefined) {
    normalized.affinity_key = state.invalidAffinityKey
  }
  return normalized
}

export function parseSelectedRequestID(query: Record<string, unknown>): string | undefined {
  return scalarUUIDv4(query.selected_request_id)
}

function scalarUUIDv4(raw: unknown): string | undefined {
  return typeof raw === 'string' && requestIDPattern.test(raw) ? raw : undefined
}

function parseLogCursorHistory(raw: unknown): string[] {
  if (typeof raw !== 'string' || raw.length > 30_000) return []
  try {
    const parsed: unknown = JSON.parse(raw)
    if (
      !Array.isArray(parsed) ||
      parsed.length > 50 ||
      parsed.some((cursor) => typeof cursor !== 'string' || !logCursorPattern.test(cursor)) ||
      new Set(parsed).size !== parsed.length
    ) {
      return []
    }
    return parsed as string[]
  } catch {
    return []
  }
}

function serializeLogCursorHistory(cursors: readonly string[]): string | undefined {
  const normalized = cursors.filter((cursor) => logCursorPattern.test(cursor)).slice(-50)
  return normalized.length > 0 ? JSON.stringify(normalized) : undefined
}
