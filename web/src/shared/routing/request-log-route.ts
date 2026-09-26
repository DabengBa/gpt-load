import {
  scalarRouteQuery,
  type SharedRouteQuery,
  type SharedRouteQueryValue,
} from './route-query'
import { defaultTimeRange, timeRangeMilliseconds } from '../lib/time'

// The detail overlay is URL-driven so it deep-links and survives reload,
// mirroring the classic logs-route contract. Classic keeps its own copy in
// features/logs/logs-route.ts until the logs domain migrates (Phase 3).
export const selectedRequestIdParam = 'selected_request_id'

const requestIDPattern =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

export function parseSelectedRequestId(query: SharedRouteQuery): string | undefined {
  const value = scalarRouteQuery(query[selectedRequestIdParam])
  return value !== undefined && requestIDPattern.test(value) ? value : undefined
}

// Time-range contract mirrors classic `parseAppliedLogFilterState`: both
// params must be canonical non-negative integers with from < to — anything
// else drops the pair and the caller falls back to `defaultLogRange`.
const canonicalNonNegativeInteger = /^(?:0|[1-9]\d*)$/

function parseRangeMs(
  raw: SharedRouteQueryValue | readonly SharedRouteQueryValue[],
): number | undefined {
  const value = scalarRouteQuery(raw)
  if (value === undefined || !canonicalNonNegativeInteger.test(value)) return undefined
  const number = Number(value)
  return Number.isSafeInteger(number) ? number : undefined
}

export function parseLogRangeMs(
  query: SharedRouteQuery,
): { from_ms: number; to_ms: number } | undefined {
  const from = parseRangeMs(query.from_ms)
  const to = parseRangeMs(query.to_ms)
  return from !== undefined && to !== undefined && from < to
    ? { from_ms: from, to_ms: to }
    : undefined
}

// Same window as classic `defaultRange()`: a second-aligned now, from one
// default range ago through +24h so in-flight writes stay visible.
export function defaultLogRange(now = Date.now()): { from_ms: number; to_ms: number } {
  const aligned = Math.floor(now / 1000) * 1000
  return {
    from_ms: Math.max(0, aligned - timeRangeMilliseconds[defaultTimeRange]),
    to_ms: aligned + 24 * 60 * 60 * 1000,
  }
}
