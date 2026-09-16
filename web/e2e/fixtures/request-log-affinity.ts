import type { Page, Request } from '@playwright/test'

export const ADMIN_KEY = 'e2e-admin-key'
export const ACCESS_KEY = 'e2e-access-key'
export const AFFINITY_KEY = '0123456789abcdef****fedcba9876543210'
export const DIFFERENT_AFFINITY_KEY = 'aaaaaaaaaaaaaaaa****bbbbbbbbbbbbbbbb'
export const EMPTY_AFFINITY_KEY = 'eeeeeeeeeeeeeeee****ffffffffffffffff'

const requestIDs = {
  sameFirst: '11111111-1111-4111-8111-111111111111',
  sameSecond: '22222222-2222-4222-8222-222222222222',
  different: '33333333-3333-4333-8333-333333333333',
  historical: '44444444-4444-4444-8444-444444444444',
} as const

function requestLogItem(
  requestID: string,
  completedAtMs: number,
  affinityKey: string | null,
  clientModel = 'gpt-4o',
) {
  return {
    request_id: requestID,
    completed_at_ms: completedAtMs,
    access_key: { id: 7, name: 'e2e access key', deleted: false },
    protocol: 'openai-completions',
    operation: 'chat_completion',
    upstream_protocol: 'openai-completions',
    client_model: clientModel,
    upstream_model: 'gpt-4o',
    upstream_reported_model: 'gpt-4o',
    model_consistency: 'match',
    reasoning: null,
    status: 'success',
    status_code: 200,
    stream: false,
    first_response_ms: null,
    duration_ms: 120,
    attempt_count: 1,
    error_code: '',
    error_summary: '',
    affinity_hit: Boolean(affinityKey),
    continuity_hit: false,
    affinity_source: affinityKey ? 'prompt_cache_key' : 'none',
    affinity_state: affinityKey ? 'hit' : 'no_signal',
    affinity_key: affinityKey,
    group_id: null,
    channel_id: null,
    credential_id: null,
    credential_name: '',
    route_mode: 'native',
    usage_state: 'not_applicable',
    cost_state: 'not_applicable',
    pricing_completeness: 'not_applicable',
    pricing_mode: null,
    context_threshold_tokens: null,
    input_tokens: '0',
    cache_read_tokens: '0',
    cache_write_5m_tokens: '0',
    cache_write_1h_tokens: '0',
    cache_write_unknown_tokens: '0',
    output_tokens: '0',
    estimated_cost_nano_usd: '0',
  }
}

const rows = [
  requestLogItem(requestIDs.sameFirst, 1700003000000, AFFINITY_KEY),
  requestLogItem(requestIDs.sameSecond, 1700002900000, AFFINITY_KEY),
  requestLogItem(requestIDs.different, 1700002800000, DIFFERENT_AFFINITY_KEY),
  requestLogItem(requestIDs.historical, 1700001000000, null),
]

function envelope(data: unknown) {
  return { code: 0, message: 'OK', data }
}

function response(data: unknown, status = 200) {
  return {
    status,
    contentType: 'application/json',
    body: JSON.stringify(envelope(data)),
  }
}

function isAccessKeyRequest(request: Request): boolean {
  return request.headers()['authorization'] === `Bearer ${ACCESS_KEY}`
}

export interface RequestLogAffinityRoutes {
  readonly logRequests: URL[]
  readonly logRequestPaths: string[]
}

export async function installRequestLogAffinityRoutes(
  page: Page,
  principal: 'admin' | 'access_key' = 'admin',
): Promise<RequestLogAffinityRoutes> {
  const key = principal === 'admin' ? ADMIN_KEY : ACCESS_KEY
  await page.addInitScript((authKey) => {
    window.localStorage.setItem('gpt-load.auth-key', authKey)
  }, key)

  const logRequests: URL[] = []
  await page.route(
    (url) => url.pathname === '/api' || url.pathname.startsWith('/api/'),
    async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      const path = url.pathname

      if (path === '/api/auth/session') {
        const requestPrincipal = isAccessKeyRequest(request) ? 'access_key' : 'admin'
        await route.fulfill(response({ authenticated: true, principal_type: requestPrincipal }))
        return
      }
      if (path === '/api/groups/options') {
        await route.fulfill(response([]))
        return
      }
      if (path === '/api/channels') {
        await route.fulfill(response({ items: [], total: 0 }))
        return
      }
      if (path === '/api/access-keys/options') {
        await route.fulfill(response([{ id: 7, name: 'e2e access key', status: 'active' }]))
        return
      }
      if (path === '/api/logs') {
        logRequests.push(url)
        const affinityKey = url.searchParams.get('affinity_key')
        const items =
          affinityKey === EMPTY_AFFINITY_KEY
            ? []
            : affinityKey === AFFINITY_KEY
              ? rows.filter((row) => row.affinity_key === AFFINITY_KEY)
              : rows.map((row) =>
                  isAccessKeyRequest(request) ? { ...row, affinity_key: null } : row,
                )
        await route.fulfill(response({ items, next_cursor: null }))
        return
      }
      if (path.startsWith('/api/logs/')) {
        const item = rows[0]
        await route.fulfill(response({ ...item, attempts: [] }))
        return
      }

      await route.fulfill(response({}, 404))
    },
  )

  return {
    logRequests,
    get logRequestPaths() {
      return logRequests.map((request) => request.pathname + request.search)
    },
  }
}

export async function openRequestLogs(
  page: Page,
  routes: RequestLogAffinityRoutes,
  query = '',
): Promise<void> {
  const normalizedQuery = query.startsWith('&') ? `?${query.slice(1)}` : query
  await page.goto(`/logs${normalizedQuery}`)
  await page.locator('.logs-tab').waitFor()
  await page.waitForLoadState('networkidle')
  if (routes.logRequests.length === 0) {
    await page.waitForTimeout(50)
  }
}

export { requestIDs }
