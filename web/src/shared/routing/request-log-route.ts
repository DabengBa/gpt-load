import { scalarRouteQuery, type SharedRouteQuery } from './route-query'

// The detail overlay is URL-driven so it deep-links and survives reload,
// mirroring the classic logs-route contract. Classic keeps its own copy in
// features/logs/logs-route.ts until the logs domain migrates (Phase 3).
export const selectedRequestIdParam = 'selected_request_id'

const requestIDPattern =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

export function parseSelectedRequestId(query: SharedRouteQuery): string | undefined {
  const value = scalarRouteQuery(query[selectedRequestIdParam])
  return value !== undefined && requestIDPattern.test(value) ? value : undefined
}
