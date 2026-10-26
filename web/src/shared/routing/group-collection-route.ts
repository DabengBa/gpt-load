import type { SharedRouteQuery, SharedRouteQueryRaw } from './route-query'

import type {
  GroupCollectionFilters,
  GroupCollectionSort,
  GroupCollectionStatus,
} from '@shared/control/types'
import {
  constrainCollectionSearch,
  isCanonicalRouteQuery,
  normalizeCollectionSearch,
  parsePositiveRouteInteger,
  scalarRouteQuery,
} from './route-query'

const defaultFilters: GroupCollectionFilters = {
  sort: 'recent',
  page: 1,
  page_size: 100,
}
const statuses = new Set<GroupCollectionStatus>(['available', 'unavailable', 'disabled'])
const sorts = new Set<GroupCollectionSort>(['recent', 'status', 'name', 'created'])

export function normalizeGroupCollectionSearchQuery(value: string | undefined): string | undefined {
  return normalizeCollectionSearch(value)
}

export function constrainGroupCollectionSearchQuery(value: string | undefined): string | undefined {
  return constrainCollectionSearch(value)
}

export function parseGroupCollectionRouteQuery(query: SharedRouteQuery): GroupCollectionFilters {
  const filters: GroupCollectionFilters = { ...defaultFilters }
  const q = normalizeGroupCollectionSearchQuery(scalarRouteQuery(query.q))
  const status = scalarRouteQuery(query.status)
  const sort = scalarRouteQuery(query.sort)
  const page = parsePositiveRouteInteger(query.page)

  if (q) filters.q = q
  if (status !== undefined && statuses.has(status as GroupCollectionStatus)) {
    filters.status = status as GroupCollectionStatus
  }
  if (sort !== undefined && sorts.has(sort as GroupCollectionSort)) {
    filters.sort = sort as GroupCollectionSort
  }
  if (page !== undefined) filters.page = page

  return filters
}

export function serializeGroupCollectionRouteQuery(
  filters: GroupCollectionFilters,
): SharedRouteQueryRaw {
  const query: SharedRouteQueryRaw = {}
  const q = normalizeGroupCollectionSearchQuery(filters.q)
  if (q) query.q = q
  if (filters.status !== undefined) query.status = filters.status
  if (filters.sort !== defaultFilters.sort) query.sort = filters.sort
  if (filters.page !== defaultFilters.page) query.page = String(filters.page)
  return query
}

export function isCanonicalGroupCollectionRouteQuery(
  query: SharedRouteQuery,
  filters: GroupCollectionFilters,
): boolean {
  return isCanonicalRouteQuery(query, serializeGroupCollectionRouteQuery(filters))
}
