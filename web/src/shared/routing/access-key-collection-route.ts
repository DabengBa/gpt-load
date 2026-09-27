import type { AccessKeyCollectionFilters, AccessKeyCollectionStatus } from '@shared/control/types'

import {
  constrainCollectionSearch,
  isCanonicalRouteQuery,
  normalizeCollectionSearch,
  parsePositiveRouteInteger,
  scalarRouteQuery,
  type SharedRouteQuery,
  type SharedRouteQueryRaw,
} from './route-query'

// Framework-free port of the classic access-key collection route codec. The
// drawer is part of the canonical surface: ?action=create|edit&access_key_id=.

export const defaultAccessKeyCollectionFilters: AccessKeyCollectionFilters = {
  page: 1,
  page_size: 20,
}

const statuses = new Set<AccessKeyCollectionStatus>(['active', 'disabled'])

export type AccessKeyDrawerRoute = { mode: 'create' } | { mode: 'edit'; accessKeyID: number }

export function parseAccessKeyDrawerRoute(
  query: SharedRouteQuery,
): AccessKeyDrawerRoute | undefined {
  const action = scalarRouteQuery(query.action)
  if (action === 'create') return { mode: 'create' }
  const accessKeyID = parsePositiveRouteInteger(query.access_key_id)
  return action === 'edit' && accessKeyID !== undefined ? { mode: 'edit', accessKeyID } : undefined
}

export function normalizeAccessKeyCollectionSearchQuery(
  value: string | undefined,
): string | undefined {
  return normalizeCollectionSearch(value)
}

export function constrainAccessKeyCollectionSearchQuery(
  value: string | undefined,
): string | undefined {
  return constrainCollectionSearch(value)
}

export function parseAccessKeyCollectionRouteQuery(
  query: SharedRouteQuery,
): AccessKeyCollectionFilters {
  const filters: AccessKeyCollectionFilters = { ...defaultAccessKeyCollectionFilters }
  const q = normalizeAccessKeyCollectionSearchQuery(scalarRouteQuery(query.q))
  const status = scalarRouteQuery(query.status)
  const page = parsePositiveRouteInteger(query.page)

  if (q) filters.q = q
  if (status !== undefined && statuses.has(status as AccessKeyCollectionStatus)) {
    filters.status = status as AccessKeyCollectionStatus
  }
  if (page !== undefined) filters.page = page
  return filters
}

export function serializeAccessKeyCollectionRouteQuery(
  filters: AccessKeyCollectionFilters,
  drawer?: AccessKeyDrawerRoute,
): SharedRouteQueryRaw {
  const query: SharedRouteQueryRaw = {}
  const q = normalizeAccessKeyCollectionSearchQuery(filters.q)
  if (q) query.q = q
  if (filters.status !== undefined) query.status = filters.status
  if (filters.page !== defaultAccessKeyCollectionFilters.page) query.page = String(filters.page)
  if (drawer?.mode === 'create') query.action = 'create'
  if (drawer?.mode === 'edit') {
    query.action = 'edit'
    query.access_key_id = String(drawer.accessKeyID)
  }
  return query
}

export function isCanonicalAccessKeyCollectionRouteQuery(
  query: SharedRouteQuery,
  filters: AccessKeyCollectionFilters,
  drawer?: AccessKeyDrawerRoute,
): boolean {
  return isCanonicalRouteQuery(
    query,
    serializeAccessKeyCollectionRouteQuery(filters, drawer),
  )
}
