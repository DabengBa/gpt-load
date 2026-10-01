export const timeRanges = ['1h', '24h', '3d', '7d', '15d', '30d'] as const
export type TimeRange = (typeof timeRanges)[number]
export const dateTimePresets = ['today', 'yesterday', ...timeRanges] as const
export type DateTimePreset = (typeof dateTimePresets)[number]

export const defaultTimeRange: TimeRange = '24h'

const hourMilliseconds = 60 * 60 * 1000

export const timeRangeMilliseconds: Record<TimeRange, number> = {
  '1h': hourMilliseconds,
  '24h': 24 * hourMilliseconds,
  '3d': 3 * 24 * hourMilliseconds,
  '7d': 7 * 24 * hourMilliseconds,
  '15d': 15 * 24 * hourMilliseconds,
  '30d': 30 * 24 * hourMilliseconds,
}

export function isTimeRange(value: unknown): value is TimeRange {
  return typeof value === 'string' && timeRanges.some((range) => range === value)
}

export function isDateTimePreset(value: unknown): value is DateTimePreset {
  return value === 'today' || value === 'yesterday' || isTimeRange(value)
}

export function resolveDateTimePreset(
  preset: DateTimePreset,
  now = Date.now(),
): { from_ms: number; to_ms: number } {
  let range: { from_ms: number; to_ms: number }
  if (preset === 'today' || preset === 'yesterday') {
    const midnight = new Date(now)
    midnight.setHours(0, 0, 0, 0)
    if (preset === 'today') {
      range = { from_ms: midnight.getTime(), to_ms: now }
    } else {
      const to = midnight.getTime()
      midnight.setDate(midnight.getDate() - 1)
      range = { from_ms: midnight.getTime(), to_ms: to }
    }
  } else {
    range = {
      from_ms: Math.max(0, now - timeRangeMilliseconds[preset]),
      to_ms: now,
    }
  }
  // A zero/negative-width range (e.g. "today" hit exactly at midnight) is
  // rejected by range validation downstream; keep a 1ms floor.
  if (range.from_ms >= range.to_ms) range = { from_ms: range.from_ms, to_ms: range.from_ms + 1 }
  return range
}

export function normalizeSeconds(value: number): number {
  return Math.max(1, Math.ceil(Number.isFinite(value) ? value : 0))
}

export function localDateTimeInput(milliseconds: number): string {
  const date = new Date(milliseconds)
  if (!Number.isSafeInteger(milliseconds) || milliseconds < 0 || Number.isNaN(date.getTime())) {
    return ''
  }
  const pad = (part: number, length = 2) => String(part).padStart(length, '0')
  const fraction = date.getMilliseconds() ? `.${pad(date.getMilliseconds(), 3)}` : ''
  return `${pad(date.getFullYear(), 4)}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}:${pad(date.getSeconds())}${fraction}`
}

export function parseLocalDateTime(value: string): Date | undefined {
  const match = value.match(/^(\d{4,})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$/u)
  if (!match) return undefined
  const [year, month, day, hour, minute, second] = match.slice(1, 7).map(Number)
  const milliseconds = Number((match[7] ?? '').padEnd(3, '0'))
  const date = new Date(0)
  date.setFullYear(year!, month! - 1, day!)
  date.setHours(hour!, minute!, second!, milliseconds)
  return date.getTime() >= 0 &&
    date.getFullYear() === year &&
    date.getMonth() === month! - 1 &&
    date.getDate() === day &&
    date.getHours() === hour &&
    date.getMinutes() === minute &&
    date.getSeconds() === second &&
    date.getMilliseconds() === milliseconds
    ? date
    : undefined
}

export function currentTimeZone(): string {
  try {
    return new Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
  } catch {
    return 'UTC'
  }
}

export function serverClockOffset(serverNowMs: number, clientNowMs: number = Date.now()): number {
  assertMilliseconds(serverNowMs)
  assertMilliseconds(clientNowMs)
  return serverNowMs - clientNowMs
}

export function serverNow(offsetMs: number, clientNowMs: number = Date.now()): number {
  if (!Number.isFinite(offsetMs)) {
    throw new RangeError('Server clock offset must be finite')
  }
  assertMilliseconds(clientNowMs)
  const resolved = clientNowMs + offsetMs
  assertMilliseconds(resolved)
  return resolved
}

function assertMilliseconds(value: number): void {
  if (!Number.isSafeInteger(value)) {
    throw new RangeError('Epoch milliseconds must be a safe integer')
  }
}
