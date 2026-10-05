import { expect, test, type Page, type Route } from '@playwright/test'

// Monitor-domain coverage for the Astryx entry. /monitor is the usage & cost
// surface behind a canonical filter query; access_key principals get the
// scoped usage view and must not issue the admin-only /api/health call.

const now = 1730000000000
const hourMs = 3_600_000

function usageAggregate(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  // The strict projector requires success+failure === request_count and
  // total_tokens === sum(token parts) — keep the fixture honest by default.
  return {
    request_count: 0,
    success_count: 0,
    failure_count: 0,
    uncached_input_tokens: 0,
    cache_read_tokens: 0,
    cache_write_5m_tokens: 0,
    cache_write_1h_tokens: 0,
    cache_write_unknown_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    duration_ms_total: 0,
    duration_sample_count: 0,
    first_response_ms_total: 0,
    first_response_sample_count: 0,
    usage_missing_count: 0,
    partial_count: 0,
    unpriced_request_count: 0,
    pricing_partial_count: 0,
    estimated_cost_nano_usd: '0',
    ...overrides,
  }
}

// 24h contract: 24 UTC-aligned hourly buckets, observed_at inside the last one.
function usagePayload(options: { scope?: 'admin' | 'access_key' } = {}) {
  const scope = options.scope ?? 'admin'
  const to = Math.ceil((now + 1) / hourMs) * hourMs
  const from = to - 24 * hourMs
  const summary = usageAggregate({
    request_count: 4,
    success_count: 3,
    failure_count: 1,
    uncached_input_tokens: 1000,
    cache_read_tokens: 200,
    output_tokens: 800,
    total_tokens: 2000,
    duration_ms_total: 4000,
    duration_sample_count: 4,
    first_response_ms_total: 1200,
    first_response_sample_count: 4,
    estimated_cost_nano_usd: '1500000000',
  })
  const series = Array.from({ length: 24 }, (_, index) =>
    index === 20
      ? {
          ...summary,
          bucket_start_ms: from + index * hourMs,
          bucket_end_ms: from + (index + 1) * hourMs,
        }
      : {
          ...usageAggregate(),
          bucket_start_ms: from + index * hourMs,
          bucket_end_ms: from + (index + 1) * hourMs,
        },
  )
  const distItem = (identity: Record<string, unknown>) => ({
    ...identity,
    request_count: 4,
    total_tokens: 2000,
    estimated_cost_nano_usd: '1500000000',
  })
  const distribution = (dimension: string, metric: string, item: Record<string, unknown>) => ({
    dimension,
    metric,
    items: [item],
    other: null,
  })
  const breakdownRow = (model: string) => ({
    model,
    ...(scope === 'admin' ? { group_id: 1, channel_id: 'openai' } : {}),
    ...usageAggregate({
      request_count: 2,
      success_count: 2,
      failure_count: 0,
      uncached_input_tokens: 500,
      output_tokens: 500,
      total_tokens: 1000,
      estimated_cost_nano_usd: '750000000',
    }),
    attempt_count: 2,
    attempt_failure_count: 0,
    normal_attempt_count: 2,
    slow_attempt_count: 0,
    faulty_attempt_count: 0,
  })
  return {
    range: '24h',
    granularity: 'hour',
    bucket_width_ms: hourMs,
    from_ms: from,
    to_ms: to,
    observed_at_ms: now,
    summary,
    series,
    distributions: {
      model: {
        requests: distribution('model', 'requests', distItem({ model: 'gpt-4o' })),
        tokens: distribution('model', 'tokens', distItem({ model: 'gpt-4o' })),
        cost: distribution('model', 'cost', distItem({ model: 'gpt-4o' })),
      },
      group:
        scope === 'admin'
          ? {
              requests: distribution('group', 'requests', distItem({ group_id: 1 })),
              tokens: distribution('group', 'tokens', distItem({ group_id: 1 })),
              cost: distribution('group', 'cost', distItem({ group_id: 1 })),
            }
          : undefined,
      access_key:
        scope === 'admin'
          ? {
              requests: distribution('access_key', 'requests', distItem({ access_key_id: 1 })),
              tokens: distribution('access_key', 'tokens', distItem({ access_key_id: 1 })),
              cost: distribution('access_key', 'cost', distItem({ access_key_id: 1 })),
            }
          : undefined,
    },
    collection_health: {
      scope: scope === 'admin' ? 'current_process' : 'access_key',
      dropped_total: 0,
      write_failure_total: 0,
      last_write_failure_at_ms: null,
    },
    breakdown: {
      scope,
      rows: [breakdownRow('gpt-4o'), breakdownRow('gpt-4o-mini')],
      total: summary,
      attempt_total: {
        attempt_count: 4,
        attempt_failure_count: 0,
        normal_attempt_count: 4,
        slow_attempt_count: 0,
        faulty_attempt_count: 0,
      },
      pagination: { page: 1, page_size: 20, total_items: 2, total_pages: 1 },
    },
  }
}

const prodGroupOption = {
  id: 1,
  name: 'prod',
  channel_id: 'openai',
  connection_type: 'api_key',
  params: {},
  provider_url: null,
  enabled: true,
  models: ['gpt-4o'],
}

const openaiChannel = {
  channel_id: 'openai',
  name: 'OpenAI',
  mark: 'O',
  icon: 'openai',
  search_terms: [],
  description: '',
  default_base_url: 'https://api.openai.com',
  notices: [],
  param_fields: [],
  credential_fields: [],
  connection: { type: 'api_key', credential_input: 'batch_text', authorization_methods: [] },
  capabilities: {
    model_discovery: false,
    quota_observation: false,
    credential_actions: [],
    outbound_proxy: false,
  },
  routes: [
    {
      client_protocol: 'openai-completions',
      operation: 'chat_completion',
      route_mode: 'native',
      model_dependent: false,
      possible_modes: [],
    },
  ],
  client_protocols: ['openai-completions'],
}

interface MonitorRequests {
  usageQueries: URLSearchParams[]
  healthCalls: number
  inspectCalls: number
}

async function mockMonitor(
  page: Page,
  options: { principalType?: 'admin' | 'access_key' } = {},
): Promise<MonitorRequests> {
  const principalType = options.principalType ?? 'admin'
  const requests: MonitorRequests = { usageQueries: [], healthCalls: 0, inspectCalls: 0 }

  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')

  await page.route('**/api/**', async (route: Route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const fulfill = (data: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data }),
      })

    if (path === '/api/auth/session') {
      await fulfill({ authenticated: true, principal_type: principalType })
      return
    }
    if (path === '/api/health') {
      requests.healthCalls += 1
      await fulfill({})
      return
    }
    if (path === '/api/route/inspect') {
      requests.inspectCalls += 1
      await fulfill({})
      return
    }
    if (path === '/api/usage') {
      requests.usageQueries.push(url.searchParams)
      await fulfill(
        usagePayload({ scope: principalType === 'access_key' ? 'access_key' : 'admin' }),
      )
      return
    }
    if (path === '/api/groups/options') {
      await fulfill([prodGroupOption])
      return
    }
    if (path === '/api/channels') {
      await fulfill({ items: [openaiChannel], total: 1 })
      return
    }
    if (path === '/api/access-keys/options') {
      await fulfill([{ id: 1, name: 'prod key', status: 'active' }])
      return
    }
    await fulfill({})
  })
  return requests
}

async function expectAstryxDocument(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible()
}

test('canonicalizes the bare query to the usage filters and renders the page', async ({ page }) => {
  await mockMonitor(page)
  await page.goto('/monitor?tab=bogus&junk=1', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/monitor\?range=\w+$/)

  await expect(page.getByRole('heading', { name: 'Monitor', exact: true })).toBeVisible()
  // The retired tabs are gone: no tablist renders on the page.
  await expect(page.getByRole('tablist')).not.toBeVisible()
  await expect(page.getByRole('heading', { name: 'Model and route breakdown' })).toBeVisible()
})

test('usage renders the model and route breakdown', async ({ page }) => {
  const requests = await mockMonitor(page)
  await page.goto('/monitor', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/monitor\?range=24h$/)

  await expect(page.getByRole('heading', { name: 'Model and route breakdown' })).toBeVisible()
  await expect(page.getByText('gpt-4o-mini')).toBeVisible()
  // The request carried the canonical filter params.
  expect(requests.usageQueries.at(-1)?.get('range')).toBe('24h')
})

test('usage range selector applies through the filter bar', async ({ page }) => {
  const requests = await mockMonitor(page)
  await page.goto('/monitor?range=24h', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page.getByRole('heading', { name: 'Model and route breakdown' })).toBeVisible()

  // The bar holds the Range Selector as a draft; Apply commits the query.
  await page.getByRole('combobox', { name: 'Range' }).click()
  await page.getByRole('option', { name: '7 days' }).click()
  await page.getByRole('button', { name: 'Apply' }).click()
  await expect(page).toHaveURL(/\/monitor\?range=7d$/)
  await expect.poll(() => requests.usageQueries.at(-1)?.get('range')).toBe('7d')
})

test('usage filter bar applies filters through the canonical query', async ({ page }) => {
  await mockMonitor(page)
  await page.goto('/monitor', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page.getByRole('heading', { name: 'Model and route breakdown' })).toBeVisible()

  const bar = page.getByRole('form', { name: 'Usage report filters' })
  await expect(bar).toBeVisible()
  await expect(page.getByRole('button', { name: 'Filter' })).not.toBeVisible()

  await bar.getByRole('button', { name: 'Group', exact: true }).click()
  await page.getByRole('option', { name: 'prod' }).click()
  await bar.getByRole('button', { name: 'Upstream model', exact: true }).click()
  await page.getByRole('option', { name: 'gpt-4o' }).click()
  await bar.getByRole('button', { name: 'Apply' }).click()

  await expect(page).toHaveURL(/upstream_model=gpt-4o/)
  await expect(page).toHaveURL(/group_id=1/)
  await expect(page).not.toHaveURL(/channel_id=|credential_id=|panel=/)
})

test('access_key usage hides cross-principal filter fields', async ({ page }) => {
  await mockMonitor(page, { principalType: 'access_key' })
  await page.goto('/monitor', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page.getByRole('heading', { name: 'Model and route breakdown' })).toBeVisible()

  const bar = page.getByRole('form', { name: 'Usage report filters' })
  await expect(bar).toBeVisible()
  // selfScoped: the model filter is a free-text input; the Group selector is absent.
  await expect(bar.getByLabel('Upstream model')).toBeVisible()
  await expect(bar.getByRole('button', { name: 'Group', exact: true })).not.toBeVisible()
})

test('access_key principal never calls /api/health or /api/route/inspect', async ({ page }) => {
  const requests = await mockMonitor(page, { principalType: 'access_key' })
  await page.goto('/monitor?tab=health&groups=expanded', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/monitor\?range=\w+$/)
  expect(requests.healthCalls).toBe(0)
  expect(requests.inspectCalls).toBe(0)
})
