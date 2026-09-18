import type { Page } from '@playwright/test'

export const ADMIN_KEY = 'e2e-admin-key'
export const PROVIDER_URL = 'https://provider-alpha.example.com'

export const requestIDs = {
  mapped: 'aaaaaaaa-1111-4111-8111-111111111111',
  plain: 'bbbbbbbb-2222-4222-8222-222222222222',
} as const

function baseLogItem(requestID: string) {
  return {
    request_id: requestID,
    completed_at_ms: 1700003000000,
    access_key: { id: 7, name: 'e2e access key', deleted: false },
    protocol: 'openai-completions',
    operation: 'chat_completion',
    upstream_protocol: 'openai-completions',
    model_consistency: 'match',
    reasoning: null,
    status: 'success',
    status_code: 200,
    stream: true,
    attempt_count: 1,
    error_code: '',
    error_summary: '',
    affinity_hit: false,
    continuity_hit: false,
    affinity_source: 'none',
    affinity_state: 'no_signal',
    affinity_key: null,
    channel_id: 'openai_compatible',
    route_mode: 'native',
    usage_state: 'complete',
    cost_state: 'priced',
    pricing_completeness: 'complete',
    pricing_mode: null,
    context_threshold_tokens: null,
    input_tokens: '120',
    cache_read_tokens: '0',
    cache_write_5m_tokens: '0',
    cache_write_1h_tokens: '0',
    cache_write_unknown_tokens: '0',
    output_tokens: '42',
    estimated_cost_nano_usd: '5200000',
  }
}

// Row 1 exercises every new display affordance at once: provider link, alias
// mapping and a first response over the 15s threshold. Row 2 is the control row.
const rows = [
  {
    ...baseLogItem(requestIDs.mapped),
    client_model: 'worker',
    upstream_model: 'gpt-5.6-luna',
    upstream_reported_model: 'gpt-5.6-luna',
    first_response_ms: 16_000,
    duration_ms: 24_000,
    group_id: 1,
    credential_id: 3,
    credential_name: 'key-a',
  },
  {
    ...baseLogItem(requestIDs.plain),
    client_model: 'gpt-4o',
    upstream_model: 'gpt-4o',
    upstream_reported_model: 'gpt-4o',
    first_response_ms: 500,
    duration_ms: 800,
    group_id: 2,
    credential_id: 4,
    credential_name: 'key-b',
  },
]

const groupOptions = [
  {
    id: 1,
    name: 'alpha',
    channel_id: 'openai_compatible',
    connection_type: 'api_key',
    params: { base_url: 'https://alpha.example/v1' },
    provider_url: PROVIDER_URL,
    enabled: true,
    models: ['worker', 'gpt-4o'],
  },
  {
    id: 2,
    name: 'beta',
    channel_id: 'openai_compatible',
    connection_type: 'api_key',
    params: { base_url: 'https://beta.example/v1' },
    provider_url: null,
    enabled: true,
    models: ['gpt-5'],
  },
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

export interface RequestLogDisplayRoutes {
  readonly logRequests: URL[]
}

export async function installRequestLogDisplayRoutes(
  page: Page,
): Promise<RequestLogDisplayRoutes> {
  await page.addInitScript((authKey) => {
    window.localStorage.setItem('gpt-load.auth-key', authKey)
  }, ADMIN_KEY)

  const logRequests: URL[] = []
  await page.route(
    (url) => url.pathname === '/api' || url.pathname.startsWith('/api/'),
    async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      const path = url.pathname

      if (path === '/api/auth/session') {
        await route.fulfill(response({ authenticated: true, principal_type: 'admin' }))
        return
      }
      if (path === '/api/groups/options') {
        await route.fulfill(response(groupOptions))
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
        await route.fulfill(response({ items: rows, next_cursor: null }))
        return
      }
      if (path.startsWith('/api/logs/')) {
        const requestID = path.slice('/api/logs/'.length)
        const item = rows.find((row) => row.request_id === requestID) ?? rows[0]
        await route.fulfill(response({ ...item, attempts: [] }))
        return
      }

      await route.fulfill(response({}, 404))
    },
  )

  return { logRequests }
}

export async function openRequestLogs(page: Page, query = ''): Promise<void> {
  const normalizedQuery = query.startsWith('&') ? `?${query.slice(1)}` : query
  await page.goto(`/logs${normalizedQuery}`)
  await page.locator('.logs-tab').waitFor()
  await page.waitForLoadState('networkidle')
}
