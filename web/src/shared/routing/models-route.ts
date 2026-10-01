import type {
  ModelCollectionFilters,
  ModelCollectionGroupStatus,
  ModelCollectionPricingStatus,
} from '@shared/control/resources/models'
import type { SharedRouteQuery, SharedRouteQueryRaw } from './route-query'

import {
  isCanonicalRouteQuery,
  normalizeCollectionSearch,
  parsePositiveRouteInteger,
  scalarRouteQuery,
} from './route-query'

export interface ModelsRouteState {
  filters: ModelCollectionFilters
  selectedPriceID?: number
}

export const defaultModelsFilters: ModelCollectionFilters = {
  group_status: 'enabled',
  pricing_status: 'all',
  page: 1,
  page_size: 10,
}
const groupStatuses = new Set<ModelCollectionGroupStatus>(['enabled', 'all'])
const pricingStatuses = new Set<ModelCollectionPricingStatus>(['pending', 'configured', 'all'])

export function parseModelsRouteQuery(query: SharedRouteQuery): ModelsRouteState {
  const q = normalizeCollectionSearch(scalarRouteQuery(query.q))
  const rawGroupStatus = scalarRouteQuery(query.group_status)
  const rawPricingStatus = scalarRouteQuery(query.pricing_status)
  const filters: ModelCollectionFilters = {
    ...defaultModelsFilters,
    ...(q === undefined ? {} : { q }),
    group_status:
      rawGroupStatus !== undefined &&
      groupStatuses.has(rawGroupStatus as ModelCollectionGroupStatus)
        ? (rawGroupStatus as ModelCollectionGroupStatus)
        : defaultModelsFilters.group_status,
    pricing_status:
      rawPricingStatus !== undefined &&
      pricingStatuses.has(rawPricingStatus as ModelCollectionPricingStatus)
        ? (rawPricingStatus as ModelCollectionPricingStatus)
        : defaultModelsFilters.pricing_status,
    page: parsePositiveRouteInteger(query.page) ?? defaultModelsFilters.page,
  }
  return {
    filters,
    selectedPriceID: parsePositiveRouteInteger(query.selected_price_id),
  }
}

export function serializeModelsRouteQuery(state: ModelsRouteState): SharedRouteQueryRaw {
  const query: SharedRouteQueryRaw = {}
  const q = normalizeCollectionSearch(state.filters.q)
  if (q !== undefined) query.q = q
  if (state.filters.group_status !== defaultModelsFilters.group_status) {
    query.group_status = state.filters.group_status
  }
  if (state.filters.pricing_status !== defaultModelsFilters.pricing_status) {
    query.pricing_status = state.filters.pricing_status
  }
  if (state.filters.page !== defaultModelsFilters.page) query.page = String(state.filters.page)
  if (state.selectedPriceID !== undefined) {
    query.selected_price_id = String(state.selectedPriceID)
  }
  return query
}

export function isCanonicalModelsRouteQuery(
  query: SharedRouteQuery,
  state: ModelsRouteState,
): boolean {
  return isCanonicalRouteQuery(query, serializeModelsRouteQuery(state))
}
