import type { SharedRouteQuery } from '@shared/routing/route-query'

// TanStack's default search codec JSON-decodes each value (?page=2 → 2),
// which the shared vue-router-shaped query contract (strings, duplicate keys
// as arrays) would silently drop. URLSearchParams can't be used: it maps '+'
// to space while vue-router keeps it literal — decodeURIComponent by hand.
export function parseSharedRouteSearch(searchStr: string): SharedRouteQuery {
  const query: SharedRouteQuery = {}
  const str = searchStr.startsWith('?') ? searchStr.slice(1) : searchStr
  if (str === '') return query
  for (const pair of str.split('&')) {
    if (pair === '') continue
    const eq = pair.indexOf('=')
    const rawKey = eq === -1 ? pair : pair.slice(0, eq)
    const rawValue = eq === -1 ? '' : pair.slice(eq + 1)
    let key = rawKey
    let value = rawValue
    try {
      key = decodeURIComponent(rawKey)
      value = decodeURIComponent(rawValue)
    } catch {
      // Malformed escapes keep the raw pair — same as vue-router's fallback.
    }
    const existing = query[key]
    query[key] =
      existing === undefined
        ? value
        : Array.isArray(existing)
          ? [...existing, value]
          : [existing, value]
  }
  return query
}

export function stringifySharedRouteSearch(query: Record<string, unknown>): string {
  const pairs: string[] = []
  for (const [key, value] of Object.entries(query)) {
    if (value === null || value === undefined) continue
    for (const item of Array.isArray(value) ? value : [value]) {
      if (item === null || item === undefined) continue
      pairs.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(item))}`)
    }
  }
  return pairs.length === 0 ? '' : `?${pairs.join('&')}`
}
