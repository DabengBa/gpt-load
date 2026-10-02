import {
  normalizeCollectionSearch,
  scalarRouteQuery,
  type SharedRouteQuery,
  type SharedRouteQueryRaw,
} from './route-query'

export type GroupTab = 'credentials' | 'models' | 'settings'
export type GroupModelDiscoveryFilter = 'unadded' | 'all'

export interface GroupModelsRouteState {
  discoveryOpen: boolean
  discoverySearch?: string
  discoveryFilter: GroupModelDiscoveryFilter
}

export function parsePositiveId(raw: unknown): number | undefined {
  if (typeof raw !== 'string' || !/^\d+$/u.test(raw)) return undefined
  const value = Number(raw)
  return Number.isSafeInteger(value) && value > 0 ? value : undefined
}

export function normalizeGroupTab(raw: unknown): GroupTab {
  return raw === 'models' || raw === 'settings' || raw === 'credentials' ? raw : 'credentials'
}

export function parseGroupModelsRouteQuery(query: SharedRouteQuery): GroupModelsRouteState {
  const discoveryOpen = scalarRouteQuery(query.panel) === 'discovery'
  return {
    discoveryOpen,
    discoverySearch: discoveryOpen
      ? normalizeCollectionSearch(scalarRouteQuery(query.discovery_q))
      : undefined,
    discoveryFilter:
      discoveryOpen && scalarRouteQuery(query.discovery_filter) === 'all' ? 'all' : 'unadded',
  }
}

export function serializeGroupModelsRouteQuery(state: GroupModelsRouteState): SharedRouteQueryRaw {
  const query: SharedRouteQueryRaw = { tab: 'models' }
  if (state.discoveryOpen) {
    query.panel = 'discovery'
    const discoverySearch = normalizeCollectionSearch(state.discoverySearch)
    if (discoverySearch !== undefined) query.discovery_q = discoverySearch
    if (state.discoveryFilter === 'all') query.discovery_filter = 'all'
  }
  return query
}

export function normalizeGroupQuery(query: SharedRouteQuery): SharedRouteQueryRaw {
  const tab = normalizeGroupTab(query.tab)
  if (tab === 'credentials') return { tab: 'credentials' }
  if (tab === 'models') return serializeGroupModelsRouteQuery(parseGroupModelsRouteQuery(query))
  return { tab: 'settings' }
}
