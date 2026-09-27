import type { SharedRouteQuery, SharedRouteQueryRaw } from './route-query'
import type { GatewayClientID } from '@shared/domain/home/gateway-clients'

import { gatewayClients } from '@shared/domain/home/gateway-clients'

import {
  isCanonicalRouteQuery,
  parsePositiveRouteInteger,
  scalarRouteQuery,
} from './route-query'

export interface HomeRouteState {
  accessKeyID?: number
  client: GatewayClientID
}

export const defaultHomeClient: GatewayClientID = 'cc-switch'

export function parseHomeRouteQuery(query: SharedRouteQuery): HomeRouteState {
  const rawClient = scalarRouteQuery(query.client)
  return {
    accessKeyID: parsePositiveRouteInteger(query.access_key_id),
    client: gatewayClients.find(({ id }) => id === rawClient)?.id ?? defaultHomeClient,
  }
}

export function serializeHomeRouteQuery(state: HomeRouteState): SharedRouteQueryRaw {
  const query: SharedRouteQueryRaw = {}
  if (state.accessKeyID !== undefined) query.access_key_id = String(state.accessKeyID)
  if (state.client !== defaultHomeClient) query.client = state.client
  return query
}

export function isCanonicalHomeRouteQuery(
  query: SharedRouteQuery,
  state: HomeRouteState,
): boolean {
  return isCanonicalRouteQuery(query, serializeHomeRouteQuery(state))
}
