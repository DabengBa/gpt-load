import type { SharedRouteQuery, SharedRouteQueryRaw } from './route-query'

import { isCanonicalRouteQuery, scalarRouteQuery } from './route-query'

export type SettingsRouteSection =
  'routing' | 'connection' | 'reliability' | 'browser-access' | 'data-maintenance' | 'system'

const settingsRouteSections = new Set<SettingsRouteSection>([
  'routing',
  'connection',
  'reliability',
  'browser-access',
  'data-maintenance',
  'system',
])

export function parseSettingsRouteSection(query: SharedRouteQuery): SettingsRouteSection {
  const value = scalarRouteQuery(query.section)
  return value !== undefined && settingsRouteSections.has(value as SettingsRouteSection)
    ? (value as SettingsRouteSection)
    : 'routing'
}

export function serializeSettingsRouteQuery(section: SettingsRouteSection): SharedRouteQueryRaw {
  return section === 'routing' ? {} : { section }
}

export function isCanonicalSettingsRouteQuery(
  query: SharedRouteQuery,
  section: SettingsRouteSection,
): boolean {
  return isCanonicalRouteQuery(query, serializeSettingsRouteQuery(section))
}
