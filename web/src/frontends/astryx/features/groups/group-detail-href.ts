import { pagePath } from '@shared/routing/page-routes'

// Manifest-derived hrefs; the dynamic route tree keeps `to`/`search` loosely
// typed, so plain `string` paths are the contract here.
export function groupDetailHref(id: number): string {
  return `${pagePath('groups')}/${id}`
}
