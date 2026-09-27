import { expect, test, type Page, type Route } from '@playwright/test'

// Phase 3 monitor-domain coverage for the Astryx entry. The /monitor route
// hosts health | usage | inspector behind a canonical `tab` query; access_key
// principals are pinned to usage and must not issue the admin-only /api/health
// call.
//
// This file covers Task B (host + health tab), Task C (usage tab), and Task D
// (inspector tab).

const now = 1730000000000

function problemCredential(
  credentialId: number,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    credential_id: credentialId,
    group_id: 1,
    group_name: 'prod',
    cooldown_until_ms: now + 3_600_000,
    failure_count: 4,
    recent_success_count: 0,
    recent_problem_count: 4,
    consecutive_problem_count: 4,
    recovery: {
      automatic: true,
      mode: 'cooldown_expiry',
      at_ms: now + 3_600_000,
    },
    identity: `sk-cred-${credentialId}`,
    last_failure_category: 'rate_limited',
    last_status_code: 429,
    ...overrides,
  }
}

function healthGroup(id: number, name: string): Record<string, unknown> {
  return {
    id,
    name,
    enabled: true,
    counts: { credentials: 2, available: 2, cooldown: 0, blacklisted: 0 },
  }
}

function requestLogHealth(): Record<string, unknown> {
  return {
    enqueued_total: 12,
    persisted_total: 10,
    dropped_not_running_total: 0,
    dropped_queue_full_total: 1,
    dropped_stopping_total: 0,
    dropped_persist_failed_total: 1,
    dropped_shutdown_total: 0,
    dropped_total: 2,
    write_failure_total: 1,
    access_quota_checkpoint_write_failure_total: 0,
    access_quota_checkpoint_degraded: false,
    retention_delete_failure_total: 0,
    queue_depth: 0,
    queue_capacity: 4096,
    last_write_failure_at_ms: now - 30_000,
    last_access_quota_checkpoint_write_failure_at_ms: null,
    last_retention_failure_at_ms: null,
  }
}

// Six groups so the collapsed collection truncates at the classic limit of 5
// and the show-all toggle is exercisable.
function healthPayload(): Record<string, unknown> {
  return {
    observed_at_ms: now,
    version: 'v1.2.3',
    uptime_seconds: 7200,
    snapshot_revision: 3,
    stats_window_seconds: 300,
    counts: { credentials: 12, available: 10, cooldown: 1, blacklisted: 1 },
    groups: [
      healthGroup(1, 'prod'),
      healthGroup(2, 'staging'),
      healthGroup(3, 'edge'),
      healthGroup(4, 'backup'),
      healthGroup(5, 'sandbox'),
      healthGroup(6, 'archive'),
    ],
    cooldown_credentials: [problemCredential(8)],
    blacklisted_credentials: [
      problemCredential(9, {
        cooldown_until_ms: null,
        recovery: {
          automatic: true,
          mode: 'scheduled_release',
          at_ms: now + 86_400_000,
        },
        last_failure_category: 'invalid_key',
        last_status_code: 401,
      }),
    ],
    low_quota_credentials: [],
    expiring_reset_credits: [],
    blocked_access_keys: [
      {
        access_key_id: 2,
        name: 'dev key',
        masked_key: 'sk-00000000****0002',
        recoverable: true,
        next_available_at_ms: now + 86_400_000,
        blocking_rules: [
          {
            id: 12,
            kind: 'periodic',
            limit_usd: '1.5',
            period_seconds: 86_400,
            used_usd: '1.5',
            remaining_usd: '0',
            status: 'exhausted',
            window_started_at_ms: now - 3_600_000,
            window_ends_at_ms: now + 82_800_000,
          },
        ],
      },
    ],
    request_log: requestLogHealth(),
    debug_capture: {
      enabled: false,
      running: false,
      retention_seconds: 0,
      active: 0,
      completed: 0,
      failed: 0,
      sweep_total: 0,
      removed_total: 0,
      sweep_failure_total: 0,
      error: '',
      last_sweep_at_ms: null,
      last_failure_at_ms: null,
    },
  }
}

interface MonitorRequests {
  healthCalls: number
}

async function mockMonitor(
  page: Page,
  options: { principalType?: 'admin' | 'access_key' } = {},
): Promise<MonitorRequests> {
  const principalType = options.principalType ?? 'admin'
  const requests: MonitorRequests = { healthCalls: 0 }

  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')

  await page.route('**/api/**', async (route: Route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
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
      await fulfill(healthPayload())
      return
    }
    await fulfill({})
  })
  return requests
}

async function expectAstryxDocument(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible()
}

test('canonicalizes the bare query to tab=health and renders health sections', async ({ page }) => {
  await mockMonitor(page)
  await page.goto('/monitor?tab=bogus&junk=1', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/monitor\?tab=health$/)

  await expect(page.getByRole('heading', { name: 'Monitor', exact: true })).toBeVisible()
  await expect(page.getByRole('tab', { name: 'Health' })).toHaveAttribute('aria-selected', 'true')

  await expect(page.getByRole('region', { name: 'Health overview' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Credentials that need attention' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Request-log collection' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Access key cost limits' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Group health' })).toBeVisible()
})

test('renders problem credentials, blocked access keys, and collapsed groups', async ({ page }) => {
  await mockMonitor(page)
  await page.goto('/monitor', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/monitor\?tab=health$/)

  // Problem rows render as links into the classic-owned group credentials
  // view; the tooltip description duplicates the name, so assert the link.
  await expect(page.getByRole('link', { name: 'sk-cred-8', exact: true })).toBeVisible()
  await expect(page.getByRole('link', { name: 'sk-cred-9', exact: true })).toBeVisible()

  // Blocked access key card.
  await expect(page.getByText('sk-00000000****0002')).toBeVisible()

  // Collapsed group collection shows 5 of 6 rows with a show-all toggle.
  const groupsTable = page.getByRole('table', { name: 'Group health list' })
  await expect(groupsTable.getByRole('link', { name: 'prod', exact: true })).toBeVisible()
  await expect(groupsTable.getByRole('link', { name: 'staging', exact: true })).not.toBeVisible()
  await expect(page.getByRole('button', { name: 'View all 6 Groups' })).toBeVisible()
})

test('groups=expanded deep link shows every group and collapses via toggle', async ({ page }) => {
  await mockMonitor(page)
  await page.goto('/monitor?tab=health&groups=expanded', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/groups=expanded/)

  const groupsTable = page.getByRole('table', { name: 'Group health list' })
  await expect(groupsTable.getByRole('link', { name: 'staging', exact: true })).toBeVisible()

  await page.getByRole('button', { name: 'Collapse Groups' }).click()
  await expect(page).toHaveURL(/\/monitor\?tab=health$/)
  await expect(groupsTable.getByRole('link', { name: 'staging', exact: true })).not.toBeVisible()
})

test('tab switch pushes the canonical tab query without a document reload', async ({ page }) => {
  await mockMonitor(page)
  await page.goto('/monitor?tab=health', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  await page.getByRole('tab', { name: 'Usage & cost' }).click()
  await expect(page).toHaveURL(/\/monitor\?tab=usage&range=\w+&metric=tokens$/)
  await expect(page.getByRole('tab', { name: 'Usage & cost' })).toHaveAttribute(
    'aria-selected',
    'true',
  )

  // SPA navigation: the astryx shell never reloaded.
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible()
})

test('refresh re-issues the health query', async ({ page }) => {
  const requests = await mockMonitor(page)
  await page.goto('/monitor?tab=health', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page.getByRole('region', { name: 'Health overview' })).toBeVisible()
  const before = requests.healthCalls
  expect(before).toBeGreaterThan(0)

  await page.getByRole('button', { name: 'Refresh' }).click()
  await expect.poll(() => requests.healthCalls).toBeGreaterThan(before)
})

test('access_key principal is pinned to usage and never calls /api/health', async ({ page }) => {
  const requests = await mockMonitor(page, { principalType: 'access_key' })
  await page.goto('/monitor?tab=health', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/monitor\?tab=usage&range=\w+&metric=tokens$/)
  expect(requests.healthCalls).toBe(0)
  await expect(page.getByRole('tab', { name: 'Health' })).not.toBeVisible()
})

// ---------------------------------------------------------------------------
// Task C: usage tab
// ---------------------------------------------------------------------------

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
      ? { ...summary, bucket_start_ms: from + index * hourMs, bucket_end_ms: from + (index + 1) * hourMs }
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
      attempt_total: { attempt_count: 4, attempt_failure_count: 0 },
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

interface UsageRequests {
  usageQueries: URLSearchParams[]
  healthCalls: number
}

async function mockUsage(
  page: Page,
  options: { principalType?: 'admin' | 'access_key' } = {},
): Promise<UsageRequests> {
  const principalType = options.principalType ?? 'admin'
  const requests: UsageRequests = { usageQueries: [], healthCalls: 0 }

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
      await fulfill(healthPayload())
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

test('usage tab renders summary, trend, quality, distribution, and breakdown', async ({
  page,
}) => {
  const requests = await mockUsage(page)
  await page.goto('/monitor?tab=usage', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/tab=usage/)

  await expect(
    page.getByRole('heading', { name: 'Token and cache trend' }),
  ).toBeVisible()
  await expect(
    page.getByRole('heading', { name: 'Usage and persistence quality' }),
  ).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Consumption distribution' })).toBeVisible()
  await expect(
    page.getByRole('heading', { name: 'Model and route breakdown' }),
  ).toBeVisible()
  await expect(page.getByText('gpt-4o-mini')).toBeVisible()
  // The request carried the canonical filter params.
  expect(requests.usageQueries.at(-1)?.get('range')).toBe('24h')
})

test('usage range selector and trend metric write the canonical query', async ({ page }) => {
  const requests = await mockUsage(page)
  await page.goto('/monitor?tab=usage&range=24h&metric=tokens', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(
    page.getByRole('heading', { name: 'Token and cache trend' }),
  ).toBeVisible()

  // Range Selector is a combobox; choosing 7 days rewrites the query.
  await page.getByRole('combobox', { name: 'Range' }).click()
  await page.getByRole('option', { name: '7 days' }).click()
  await expect(page).toHaveURL(/tab=usage&range=7d&metric=tokens/)
  await expect.poll(() => requests.usageQueries.at(-1)?.get('range')).toBe('7d')

  // Trend metric radio: tokens → cost.
  await page
    .getByRole('radiogroup', { name: 'Trend metric' })
    .getByRole('radio', { name: 'Cost' })
    .click()
  await expect(page).toHaveURL(/metric=cost/)
  await expect(page.getByRole('heading', { name: 'Estimated cost trend' })).toBeVisible()
})

test('usage filter panel applies filters through the canonical query', async ({ page }) => {
  await mockUsage(page)
  await page.goto('/monitor?tab=usage', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(
    page.getByRole('heading', { name: 'Token and cache trend' }),
  ).toBeVisible()

  await page.getByRole('button', { name: 'Filter' }).click()
  await expect(page).toHaveURL(/panel=filters/)
  const panel = page.getByRole('dialog', { name: 'Filter usage and cost' })
  await expect(panel).toBeVisible()

  await panel.getByLabel('Upstream model').fill('gpt-4o')
  await panel.getByRole('combobox', { name: 'Group' }).click()
  await page.getByRole('option', { name: 'prod' }).click()
  await panel.getByRole('button', { name: 'Apply' }).click()

  await expect(page).toHaveURL(/upstream_model=gpt-4o/)
  await expect(page).toHaveURL(/group_id=1/)
  await expect(page).not.toHaveURL(/panel=filters/)
})

test('access_key usage hides cross-principal filter fields', async ({ page }) => {
  await mockUsage(page, { principalType: 'access_key' })
  await page.goto('/monitor?tab=usage', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(
    page.getByRole('heading', { name: 'Token and cache trend' }),
  ).toBeVisible()

  await page.getByRole('button', { name: 'Filter' }).click()
  const panel = page.getByRole('dialog', { name: 'Filter usage and cost' })
  await expect(panel).toBeVisible()
  await expect(panel.getByLabel('Upstream model')).toBeVisible()
  // selfScoped: group/channel/credential fields are absent.
  await expect(panel.getByRole('combobox', { name: 'Group' })).not.toBeVisible()
  await expect(panel.getByLabel(/^Credential/)).not.toBeVisible()
})

// --- Inspector tab (Task D) --------------------------------------------------

// Strict projector contract: configured_share per priority tier must total ~1
// (or 0) and entry_cooldown_until_ms must be strictly after observed_at_ms.
function inspectPayload(): Record<string, unknown> {
  return {
    observed_at_ms: now,
    snapshot_revision: 7,
    route_strategy: 'native_first',
    protocol: 'openai-completions',
    operation: 'chat_completion',
    route_requirement: 'any',
    external_model: 'gpt-4o',
    access_key: { id: 1, name: 'prod key', status: 'active' },
    routable: true,
    reason_code: null,
    groups: [
      {
        group_id: 1,
        group_name: 'prod',
        channel_id: 'openai',
        route_mode: 'native',
        route_requirement_satisfied: true,
        entry_id: 'entry-1',
        upstream_model: 'gpt-4o',
        entry_weight: 50,
        priority: 1,
        configured_share: 1,
        effective_share: 1,
        entry_cooldown_until_ms: null,
        included: true,
        routable: true,
        reason_code: null,
        credentials: [
          { credential_id: 11, available: true, reason_code: null, cooldown_until_ms: null },
          {
            credential_id: 12,
            available: false,
            reason_code: 'credential_cooldown',
            cooldown_until_ms: now + 600_000,
          },
        ],
      },
      {
        group_id: 2,
        group_name: 'staging',
        channel_id: 'openai',
        route_mode: 'converted',
        route_requirement_satisfied: true,
        entry_id: 'entry-2',
        upstream_model: 'gpt-4o',
        entry_weight: 10,
        priority: 2,
        configured_share: 0,
        effective_share: 0,
        entry_cooldown_until_ms: null,
        included: false,
        routable: false,
        reason_code: 'entry_weight_zero',
        credentials: [],
      },
    ],
  }
}

interface InspectorRequests {
  inspectBodies: Record<string, unknown>[]
}

async function mockInspector(
  page: Page,
  options: { inspectStatus?: number } = {},
): Promise<InspectorRequests> {
  const requests: InspectorRequests = { inspectBodies: [] }
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')

  await page.route('**/api/**', async (route: Route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const fulfill = (data: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data }),
      })

    if (path === '/api/auth/session') {
      await fulfill({ authenticated: true, principal_type: 'admin' })
      return
    }
    if (path === '/api/health') {
      await fulfill(healthPayload())
      return
    }
    if (path === '/api/route/inspect') {
      requests.inspectBodies.push(request.postDataJSON() as Record<string, unknown>)
      const status = options.inspectStatus ?? 200
      if (status === 200) {
        await fulfill(inspectPayload())
      } else {
        await route.fulfill({
          status,
          contentType: 'application/json',
          body: JSON.stringify({ code: 1, message: 'inspect failed' }),
        })
      }
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

test('inspector form submits through the canonical query and renders the result', async ({
  page,
}) => {
  const requests = await mockInspector(page)
  await page.goto('/monitor?tab=inspector', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/tab=inspector/)

  await expect(
    page.getByRole('heading', { name: 'Enter conditions to inspect the route' }),
  ).toBeVisible()

  await page.getByRole('combobox', { name: 'Access key' }).click()
  await page.getByRole('option', { name: /prod key/ }).click()
  await page.getByRole('combobox', { name: 'Protocol' }).click()
  await page.getByRole('option', { name: 'openai-completions' }).click()
  await page.getByRole('combobox', { name: 'Client model' }).click()
  await page.getByRole('option', { name: 'gpt-4o', exact: true }).click()

  await page.getByRole('button', { name: 'Inspect current route' }).click()

  // The submit rewrites the canonical query; the auto-run watch then inspects.
  await expect(page).toHaveURL(/tab=inspector/)
  await expect(page).toHaveURL(/protocol=openai-completions/)
  await expect(page).toHaveURL(/external_model=gpt-4o/)
  await expect(page).toHaveURL(/access_key_id=1/)
  await expect(page).toHaveURL(/run=1/)
  await expect.poll(() => requests.inspectBodies.length).toBe(1)
  expect(requests.inspectBodies[0]).toEqual({
    protocol: 'openai-completions',
    external_model: 'gpt-4o',
    access_key_id: 1,
  })

  await expect(
    page.getByRole('heading', { name: 'The current request can be routed' }),
  ).toBeVisible()
  await expect(
    page.getByRole('heading', { name: 'Candidate Groups' }),
  ).toBeVisible()
  const excluded = page.locator('section[aria-labelledby="route-exclusions-title"]')
  await expect(excluded.getByText('staging', { exact: true }).first()).toBeVisible()
})

test('inspector deep link with run=1 inspects on mount', async ({ page }) => {
  const requests = await mockInspector(page)
  await page.goto(
    '/monitor?tab=inspector&protocol=openai-completions&external_model=gpt-4o&access_key_id=1&run=1',
    { waitUntil: 'load' },
  )
  await expectAstryxDocument(page)
  await expect.poll(() => requests.inspectBodies.length).toBe(1)
  await expect(
    page.getByRole('heading', { name: 'The current request can be routed' }),
  ).toBeVisible()
})

test('inspector validation blocks submission and reports field errors', async ({ page }) => {
  const requests = await mockInspector(page)
  await page.goto('/monitor?tab=inspector', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  await page.getByRole('button', { name: 'Inspect current route' }).click()
  await expect(page.getByText('Select a valid protocol.')).toBeVisible()
  await expect(page.getByText('Reselect an existing access key.')).toBeVisible()
  expect(requests.inspectBodies.length).toBe(0)
  await expect(page).not.toHaveURL(/run=1/)
})

test('inspector failure shows the error state and retries', async ({ page }) => {
  const requests = await mockInspector(page, { inspectStatus: 500 })
  await page.goto(
    '/monitor?tab=inspector&protocol=openai-completions&external_model=gpt-4o&access_key_id=1&run=1',
    { waitUntil: 'load' },
  )
  await expectAstryxDocument(page)
  await expect.poll(() => requests.inspectBodies.length).toBe(1)
  await expect(
    page.getByRole('heading', { name: 'Unable to inspect the current route.' }),
  ).toBeVisible()
})
