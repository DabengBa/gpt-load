import type { SharedRouteQuery } from '@shared/routing/route-query'

// TanStack's default search codec JSON-decodes each value (?page=2 → 2),
// which the shared vue-router-shaped query contract (strings, duplicate keys
// as arrays) would silently drop. This mirrors vue-router's parseQuery: '+'
// maps to space before the '=' split, key and value decode independently,
// malformed escapes keep the raw side, and bare keys read as null. The
// null-prototype result keeps '__proto__'-style keys as plain own properties
// instead of mutating the object's prototype.
export function parseSharedRouteSearch(searchStr: string): SharedRouteQuery {
  const query: Record<string, unknown> = Object.create(null)
  const str = searchStr.startsWith('?') ? searchStr.slice(1) : searchStr
  if (str === '') return query as SharedRouteQuery
  for (const pair of str.split('&')) {
    if (pair === '') continue
    const decoded = pair.replace(/\+/g, ' ')
    const eq = decoded.indexOf('=')
    const key = safeDecode(eq === -1 ? decoded : decoded.slice(0, eq))
    const value = eq === -1 ? null : safeDecode(decoded.slice(eq + 1))
    const existing = query[key]
    query[key] =
      existing === undefined
        ? value
        : Array.isArray(existing)
          ? [...existing, value]
          : [existing, value]
  }
  return query as SharedRouteQuery
}

function safeDecode(text: string): string {
  try {
    return decodeURIComponent(text)
  } catch {
    return text
  }
}

export function stringifySharedRouteSearch(query: Record<string, unknown>): string {
  const pairs: string[] = []
  for (const [key, value] of Object.entries(query)) {
    // vue-router's stringifyQuery: undefined drops the key, null emits it bare.
    if (value === undefined) continue
    const encodedKey = encodeURIComponent(key)
    for (const item of Array.isArray(value) ? value : [value]) {
      if (item === undefined) continue
      pairs.push(
        item === null ? encodedKey : `${encodedKey}=${encodeURIComponent(String(item))}`,
      )
    }
  }
  return pairs.length === 0 ? '' : `?${pairs.join('&')}`
}
