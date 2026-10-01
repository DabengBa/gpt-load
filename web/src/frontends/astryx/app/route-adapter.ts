import type { PageRouteEntry } from '@shared/routing/page-routes'

// Pure manifest -> TanStack Router mapping. Kept free of runtime imports so
// node:test (scripts/astryx-routes.test.ts) can exercise it without Vite's
// alias resolution; page-routes.ts itself cannot load under
// --experimental-strip-types because of its JSON import.

export interface AstryxRoutePath {
  readonly name: string
  readonly path: string
}

export function toTanStackPath(path: string): string {
  return path
    .split('/')
    .map((segment) => (segment.startsWith(':') ? `$${segment.slice(1)}` : segment))
    .join('/')
}

export function astryxRoutePaths(entries: readonly PageRouteEntry[]): readonly AstryxRoutePath[] {
  return entries.map((entry) => ({ name: entry.name, path: toTanStackPath(entry.path) }))
}
