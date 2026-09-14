export const REQUEST_LOG_AFFINITY_KEY_PATTERN = /^[0-9a-f]{16}\*{4}[0-9a-f]{16}$/u

export type RequestLogAffinityKeyParseResult =
  { kind: 'missing' } | { kind: 'valid'; value: string } | { kind: 'invalid'; raw: unknown }

export function isValidRequestLogAffinityKey(value: unknown): value is string {
  return typeof value === 'string' && REQUEST_LOG_AFFINITY_KEY_PATTERN.test(value)
}

export function projectRequestLogAffinityKey(value: unknown): string | null {
  return isValidRequestLogAffinityKey(value) ? value : null
}

export function parseRequestLogAffinityKey(value: unknown): RequestLogAffinityKeyParseResult {
  if (value === undefined) return { kind: 'missing' }
  return isValidRequestLogAffinityKey(value)
    ? { kind: 'valid', value }
    : { kind: 'invalid', raw: value }
}

export function serializeRequestLogAffinityKey(value: string): string {
  if (!isValidRequestLogAffinityKey(value)) {
    throw new TypeError('invalid affinity key')
  }
  return value
}
