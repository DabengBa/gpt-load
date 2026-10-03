import type { SharedRouteQuery, SharedRouteQueryRaw } from './route-query'
import type { AccessKeyCollectionFilters } from '../control/types'

import { isCanonicalRouteQuery, scalarRouteQuery } from './route-query'
import {
  parseAccessKeyCollectionRouteQuery,
  parseAccessKeyDrawerRoute,
  serializeAccessKeyCollectionRouteQuery,
  type AccessKeyDrawerRoute,
} from './access-key-collection-route'

export type SettingsRouteSection =
  | 'routing'
  | 'connection'
  | 'reliability'
  | 'browser-access'
  | 'credentials'
  | 'data-maintenance'
  | 'system'

const settingsRouteSections = new Set<SettingsRouteSection>([
  'routing',
  'connection',
  'reliability',
  'browser-access',
  'credentials',
  'data-maintenance',
  'system',
])

// The credentials section embeds the access-key collection + drawer query
// contract that used to live on the standalone /access-keys page.
export interface SettingsCredentialsRouteState {
  collection: AccessKeyCollectionFilters
  drawer?: AccessKeyDrawerRoute
}

export function parseSettingsRouteSection(query: SharedRouteQuery): SettingsRouteSection {
  const value = scalarRouteQuery(query.section)
  return value !== undefined && settingsRouteSections.has(value as SettingsRouteSection)
    ? (value as SettingsRouteSection)
    : 'routing'
}

export function parseSettingsCredentialsRoute(
  query: SharedRouteQuery,
): SettingsCredentialsRouteState {
  return {
    collection: parseAccessKeyCollectionRouteQuery(query),
    drawer: parseAccessKeyDrawerRoute(query),
  }
}

export function serializeSettingsRouteQuery(
  section: SettingsRouteSection,
  credentials?: SettingsCredentialsRouteState,
): SharedRouteQueryRaw {
  const query: SharedRouteQueryRaw = section === 'routing' ? {} : { section }
  if (section === 'credentials' && credentials !== undefined) {
    Object.assign(
      query,
      serializeAccessKeyCollectionRouteQuery(credentials.collection, credentials.drawer),
    )
  }
  return query
}

export function isCanonicalSettingsRouteQuery(
  query: SharedRouteQuery,
  section: SettingsRouteSection,
  credentials?: SettingsCredentialsRouteState,
): boolean {
  return isCanonicalRouteQuery(query, serializeSettingsRouteQuery(section, credentials))
}
