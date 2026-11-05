// Structural equivalents of a framework router's LocationQuery/LocationQueryRaw
// so the rules stay framework-free; both routers' query shapes assign to them.
// Numbers appear when a router hands back already-typed values (e.g. TanStack
// Router re-validates the validated match.search, which carries numbers), so
// parsers must accept them to stay idempotent.
export type SharedRouteQueryValue = string | number | null | undefined
export type SharedRouteQuery = Record<
  string,
  SharedRouteQueryValue | readonly SharedRouteQueryValue[]
>
export type SharedRouteQueryRaw = Record<
  string,
  SharedRouteQueryValue | number | (SharedRouteQueryValue | number)[]
>

const maxCollectionSearchCodePoints = 200

export function scalarRouteQuery(
  value: SharedRouteQueryValue | readonly SharedRouteQueryValue[],
): string | undefined {
  if (typeof value === 'string') return value
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return undefined
}

export function parsePositiveRouteInteger(
  value: SharedRouteQueryValue | readonly SharedRouteQueryValue[],
): number | undefined {
  const candidate = scalarRouteQuery(value)
  if (candidate === undefined || !/^[1-9]\d*$/u.test(candidate)) return undefined
  const parsed = Number(candidate)
  return Number.isSafeInteger(parsed) ? parsed : undefined
}

export function normalizeCollectionSearch(value: string | undefined): string | undefined {
  const trimmed = value?.trim()
  if (!trimmed) return undefined
  return Array.from(trimmed).length <= maxCollectionSearchCodePoints ? trimmed : undefined
}

export function constrainCollectionSearch(value: string | undefined): string | undefined {
  const trimmed = value?.trim()
  if (!trimmed) return undefined
  return Array.from(trimmed).slice(0, maxCollectionSearchCodePoints).join('')
}

export function isCanonicalRouteQuery(
  query: SharedRouteQuery,
  canonical: SharedRouteQueryRaw,
): boolean {
  const actualKeys = Object.keys(query)
  const canonicalKeys = Object.keys(canonical)
  if (actualKeys.length !== canonicalKeys.length) return false
  return canonicalKeys.every((key) => sameRouteQueryValue(query[key], canonical[key]))
}

function sameRouteQueryValue(
  actual: SharedRouteQueryValue | readonly SharedRouteQueryValue[],
  canonical: SharedRouteQueryRaw[string],
): boolean {
  if (Array.isArray(actual) || Array.isArray(canonical)) {
    if (!Array.isArray(actual) || !Array.isArray(canonical)) return false
    return (
      actual.length === canonical.length &&
      actual.every((value, index) => value === canonical[index])
    )
  }
  return actual === canonical
}
