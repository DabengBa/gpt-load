import { expect, test, type Page, type Route } from '@playwright/test'

// Phase 2 home-domain coverage for the Astryx entry: the admin ledger
// (summary/attention/subscriptions/access key/gateway/spend), canonical
// `access_key_id`/`client` query state, credential-sensitive copy and
// quick-import flows, the empty welcome state, error + stale recovery, and
// the access-key read-only mode.
//
// The mock mirrors the strict home projectors: 30d statistics must keep
// day-aligned buckets whose counts sum to the summary, health recovery
// objects must match cooldown/blacklist invariants, and subscription
// credentials must satisfy the credential-item rules.

const allProtocols = [
  'openai-completions',
  'openai-responses',
  'openai-images',
  'openai-embeddings',
  'rerank',
  'anthropic',
  'gemini',
]

const dayMilliseconds = 86_400_000

function routeKey(id: number, name: string, maskedKey: string, protocols: string[]) {
  return { id, name, masked_key: maskedKey, protocols }
}

const prodKey = routeKey(1, 'prod-key', 'sk-p•••xyz', allProtocols)
const devKey = routeKey(2, 'dev-key', 'sk-d•••abc', ['openai-completions'])
const scopedKey = routeKey(1, 'my-key', 'sk-m•••key', [
  'openai-completions',
  'openai-responses',
])

function currentAccessKey() {
  return {
    id: 1,
    name: 'my-key',
    masked_key: 'sk-m•••key',
    status: 'active',
    filters: {
      groups: [1],
      protocols: ['openai-completions', 'openai-responses'],
      models: ['gpt-4o'],
      allowed_cidrs: [],
    },
    expires_at_ms: null,
    rpm_limit: 60,
    price_multiplier: '1',
    cost_limit_rules: [],
    created_at_ms: 1_730_000_000_000,
    updated_at_ms: 1_730_000_000_000,
    expired: false,
    last_request_at_ms: null,
  }
}

function baseHome(options: {
  nowMS: number
  empty?: boolean
  principalType: 'admin' | 'access_key'
}) {
  const empty = options.empty === true
  const isAccessKey = options.principalType === 'access_key'
  return {
    server_now_ms: options.nowMS,
    started_at_ms: options.nowMS - 7_200_000,
    version: 'v1.2.3',
    inventory: {
      group_count: empty ? 0 : 2,
      credential_count: empty ? 0 : 5,
      available_credential_count: empty ? 0 : 4,
      model_count: empty ? 0 : 3,
    },
    access_keys: empty ? [] : isAccessKey ? [scopedKey] : [prodKey, devKey],
    current_access_key: isAccessKey ? currentAccessKey() : null,
  }
}

function healthPayload(nowMS: number, attention: boolean) {
  return {
    observed_at_ms: nowMS,
    version: 'v1.2.3',
    uptime_seconds: 7200,
    snapshot_revision: 1,
    stats_window_seconds: 300,
    counts: { credentials: 5, available: 3, cooldown: 1, blacklisted: 1 },
    groups: [
      {
        id: 1,
        name: 'prod',
        enabled: true,
        counts: { credentials: 5, available: 3, cooldown: 1, blacklisted: 1 },
      },
    ],
    cooldown_credentials: attention
      ? [
          {
            credential_id: 8,
            group_id: 1,
            group_name: 'prod',
            cooldown_until_ms: nowMS + 3_600_000,
            failure_count: 4,
            recent_success_count: 0,
            recent_problem_count: 4,
            consecutive_problem_count: 4,
            recovery: {
              automatic: true,
              mode: 'cooldown_expiry',
              at_ms: nowMS + 3_600_000,
            },
            identity: 'sk-bill',
            last_failure_category: 'billing',
            last_status_code: 402,
          },
        ]
      : [],
    blacklisted_credentials: attention
      ? [
          {
            credential_id: 9,
            group_id: 1,
            group_name: 'prod',
            cooldown_until_ms: null,
            failure_count: 7,
            recent_success_count: 0,
            recent_problem_count: 7,
            consecutive_problem_count: 7,
            recovery: {
              automatic: true,
              mode: 'scheduled_release',
              at_ms: nowMS + 86_400_000,
            },
            identity: 'sk-dead',
            last_failure_category: 'invalid_key',
            last_status_code: 401,
          },
        ]
      : [],
    low_quota_credentials: [],
    expiring_reset_credits: [],
    blocked_access_keys: [],
    request_log: {
      enqueued_total: 0,
      persisted_total: 0,
      dropped_not_running_total: 0,
      dropped_queue_full_total: 0,
      dropped_stopping_total: 0,
      dropped_persist_failed_total: 0,
      dropped_shutdown_total: 0,
      dropped_total: 0,
      write_failure_total: 0,
      access_quota_checkpoint_write_failure_total: 0,
      access_quota_checkpoint_degraded: false,
      retention_delete_failure_total: 0,
      queue_depth: 0,
      queue_capacity: 0,
      last_write_failure_at_ms: null,
      last_access_quota_checkpoint_write_failure_at_ms: null,
      last_retention_failure_at_ms: null,
    },
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

function statisticsPayload(nowMS: number, range: '24h' | '30d') {
  const bucketMS = range === '30d' ? dayMilliseconds : 3_600_000
  const bucketCount = range === '30d' ? 30 : 24
  const toMS = (Math.floor(nowMS / bucketMS) + 1) * bucketMS
  const fromMS = toMS - bucketMS * bucketCount
  return {
    range,
    granularity: range === '30d' ? 'day' : 'hour',
    from_ms: fromMS,
    to_ms: toMS,
    observed_at_ms: nowMS,
    summary: {
      request_count: 3,
      success_count: 2,
      failure_count: 1,
      total_tokens: 150,
      input_tokens: 100,
      cache_read_tokens: 0,
      cache_write_unknown_tokens: 0,
      estimated_cost_nano_usd: '15000000',
      usage_missing_count: 0,
      partial_count: 0,
      unpriced_request_count: 0,
      pricing_partial_count: 0,
    },
    series: Array.from({ length: bucketCount }, (_, index) => ({
      bucket_start_ms: fromMS + index * bucketMS,
      bucket_end_ms: fromMS + (index + 1) * bucketMS,
      request_count: index === 0 ? 3 : 0,
      failure_count: index === 0 ? 1 : 0,
    })),
    rankings: {
      models: [
        {
          model: 'gpt-4o',
          request_count: 3,
          total_tokens: 150,
          estimated_cost_nano_usd: '15000000',
        },
      ],
      groups: [
        {
          group: { id: 1, name: 'prod', deleted: false },
          request_count: 3,
          total_tokens: 150,
          estimated_cost_nano_usd: '15000000',
        },
      ],
      access_keys: [
        {
          access_key: { id: 1, name: 'prod-key', deleted: false },
          request_count: 3,
          total_tokens: 150,
          estimated_cost_nano_usd: '15000000',
        },
      ],
    },
  }
}

function subscriptionAccountsPayload(nowMS: number) {
  return {
    observed_at_ms: nowMS,
    items: [
      {
        channel_id: 'openai',
        channel_name: 'OpenAI',
        channel_mark: 'O',
        channel_icon: 'openai',
        capabilities: {
          model_discovery: true,
          quota_observation: true,
          credential_actions: ['reset_credit'],
          outbound_proxy: false,
        },
        group_count: 1,
        available_group_count: 1,
        credential: {
          credential_id: 11,
          connection_type: 'subscription',
          secret_version: 1,
          mask: '••••sub',
          account: { email: 'sub@example.com' },
          auth_state: 'ready',
          observation: {
            state: 'fresh',
            snapshot: {
              plan_summary: { name: 'Pro', level: 'premium' },
              quota_windows: [
                {
                  id: 'w1',
                  label: 'Weekly',
                  label_key: 'weekly',
                  scope: 'model',
                  unit: 'requests',
                  used: 20,
                  limit: 100,
                  remaining: 80,
                  utilization: 0.2,
                  reset_at_ms: nowMS + 3 * dayMilliseconds,
                  window_seconds: 604_800,
                  state: 'available',
                  is_primary: true,
                },
              ],
              reset_credits_available: 0,
              reset_credits: [],
            },
            observation_version: 1,
            observed_at_ms: nowMS,
            last_attempt_at_ms: nowMS,
          },
          effective_status: 'available',
          recent_success_count: 5,
          recent_failure_count: 0,
          consecutive_failure_count: 0,
          last_failure_category: 'ok',
          last_status_code: null,
          cooldown_until_ms: null,
          recovery: { mode: 'none', automatic: false, at_ms: null },
        },
      },
    ],
  }
}

interface HomeRequests {
  paths: string[]
  reveals: number[]
}

interface MockHomeOptions {
  principalType?: 'admin' | 'access_key'
  empty?: boolean
  attention?: boolean
  subscription?: boolean
  baseStatus?: number
}

async function mockHome(
  page: Page,
  options: MockHomeOptions = {},
): Promise<HomeRequests> {
  const principalType = options.principalType ?? 'admin'
  const requests: HomeRequests = { paths: [], reveals: [] }
  const baseStatus = options.baseStatus ?? 200
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route: Route) => {
    const url = new URL(route.request().url())
    const path = url.pathname
    const nowMS = Date.now()
    requests.paths.push(`${route.request().method()} ${path}`)
    const fulfill = (data: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data }),
      })
    const fail = () =>
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ code: 500, message: 'boom', data: null }),
      })
    if (path === '/api/auth/session') {
      await fulfill({ authenticated: true, principal_type: principalType })
      return
    }
    if (path === '/api/home') {
      if (baseStatus !== 200) {
        await fail()
        return
      }
      await fulfill(
        baseHome({ nowMS, empty: options.empty === true, principalType }),
      )
      return
    }
    if (path === '/api/home/statistics') {
      const range = url.searchParams.get('range') === '24h' ? '24h' : '30d'
      await fulfill(statisticsPayload(nowMS, range))
      return
    }
    if (path === '/api/home/subscription-accounts') {
      await fulfill(
        options.subscription === false
          ? { observed_at_ms: nowMS, items: [] }
          : subscriptionAccountsPayload(nowMS),
      )
      return
    }
    if (path === '/api/health') {
      await fulfill(healthPayload(nowMS, options.attention === true))
      return
    }
    if (path === '/api/system/update') {
      await fulfill({ update: null })
      return
    }
    if (path === '/api/access-keys/1/reveal' || path === '/api/access-keys/2/reveal') {
      const id = path.endsWith('/1/reveal') ? 1 : 2
      requests.reveals.push(id)
      await fulfill({ id, key: `sk-e2e-revealed-${id}`, revealed_at_ms: nowMS })
      return
    }
    await fulfill({})
  })
  return requests
}

async function expectAstryxDocument(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible()
  await expect(page.locator('[data-testid="desktop-nav"]')).toBeVisible()
}

test('renders the admin ledger and canonicalizes the home query', async ({
  page,
}) => {
  await mockHome(page, { attention: true })
  await page.goto('/?access_key_id=abc&client=bogus&junk=1', {
    waitUntil: 'load',
  })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/$/)

  // Summary facts + version stamp.
  const title = page.getByRole('heading', {
    name: /2 Groups.*4\/5 credentials available.*3 models/,
  })
  await expect(title).toBeVisible()
  await expect(page.getByText('v1.2.3')).toBeVisible()

  // Attention rows: billing cooldown first, then blacklisted — both link into
  // the classic-owned group credentials view with the right status filter.
  const billing = page.getByRole('link', {
    name: /prod has 1 credentials with insufficient balance/,
  })
  await expect(billing).toBeVisible()
  await expect(billing).toHaveAttribute(
    'href',
    '/groups/1?tab=credentials&credential_status=cooldown',
  )
  const blacklisted = page.getByRole('link', {
    name: /prod has 1 blacklisted credentials/,
  })
  await expect(blacklisted).toHaveAttribute(
    'href',
    '/groups/1?tab=credentials&credential_status=blacklisted',
  )

  // Subscription accounts card.
  await expect(
    page.getByRole('heading', { name: 'Recently used' }),
  ).toBeVisible()
  await expect(
    page.getByRole('article', { name: 'sub@example.com · Available' }),
  ).toBeVisible()

  // Gateway connection + spend.
  await expect(
    page.getByRole('heading', { name: 'Connect to the gateway' }),
  ).toBeVisible()
  await expect(
    page.getByRole('heading', { name: 'Estimated over 30 days' }),
  ).toBeVisible()
  await expect(
    page.getByRole('link', { name: 'View usage details for gpt-4o' }),
  ).toHaveAttribute('href', '/monitor?tab=usage&range=30d&upstream_model=gpt-4o')
})

test('shows the welcome state for an empty admin home', async ({ page }) => {
  await mockHome(page, { empty: true })
  await page.goto('/', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(
    page.getByRole('heading', { name: 'Welcome to GPT-Load' }),
  ).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Import channel credentials' }),
  ).toBeVisible()
  // The empty ledger never mounts gateway or spend sections.
  await expect(
    page.getByRole('heading', { name: 'Connect to the gateway' }),
  ).toHaveCount(0)
})

test('shows the error state and recovers through retry', async ({ page }) => {
  const requests = await mockHome(page, { baseStatus: 500 })
  await page.goto('/', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(
    page.getByText(
      'Unable to load the Home inventory, so the configuration state is unknown.',
    ),
  ).toBeVisible()

  // Recovering the endpoint lets Retry rebuild the page.
  await page.unroute('**/api/home')
  await page.route('**/api/home', async (route: Route) => {
    requests.paths.push('GET /api/home')
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        code: 0,
        message: 'ok',
        data: baseHome({ nowMS: Date.now(), principalType: 'admin' }),
      }),
    })
  })
  await page.getByRole('button', { name: 'Retry' }).click()
  await expect(
    page.getByRole('heading', {
      name: /2 Groups.*4\/5 credentials available.*3 models/,
    }),
  ).toBeVisible()
})

test('keeps stale content with a warning banner when a refresh fails', async ({
  page,
}) => {
  await mockHome(page)
  await page.goto('/', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  const facts = page.getByRole('heading', {
    name: /2 Groups.*4\/5 credentials available.*3 models/,
  })
  await expect(facts).toBeVisible()

  // Fail the base endpoint, then leave and return — `refetchOnMount: always`
  // makes the remount revalidate and surface the stale-data banner.
  await page.unroute('**/api/home')
  await page.route('**/api/home', async (route: Route) => {
    await route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ code: 500, message: 'boom', data: null }),
    })
  })
  await page.getByRole('link', { name: 'Models', exact: true }).click()
  await expect(page).toHaveURL(/\/models/)
  await page.getByRole('link', { name: 'Home', exact: true }).click()
  await expect(page).toHaveURL(/\/$/)

  await expect(facts).toBeVisible()
  await expect(
    page
      .getByRole('status')
      .filter({
        hasText:
          'Unable to load the Home inventory, so the configuration state is unknown.',
      }),
  ).toBeVisible()
})

test('scopes home to the access-key session', async ({ page }) => {
  const requests = await mockHome(page, { principalType: 'access_key' })
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto('/?access_key_id=2&client=codex', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  // access_key_id is admin-only query state — it drops out of the URL while
  // the valid client choice stays.
  await expect(page).toHaveURL(/\/\?client=codex$/)

  // Identity card + read-only boundary copy.
  await expect(
    page.getByRole('heading', { name: 'my-key' }),
  ).toBeVisible()
  await expect(
    page.getByText('Current sign-in identity'),
  ).toBeVisible()
  await expect(
    page.getByText(/access key read-only view/),
  ).toBeVisible()

  // The access-key selector is disabled and the admin-only queries never run.
  await expect(
    page.getByRole('combobox', { name: 'Access key' }),
  ).toBeDisabled()
  const requested = requests.paths
  expect(requested.some((path) => path.includes('/api/health'))).toBe(false)
  expect(
    requested.some((path) => path.includes('subscription-accounts')),
  ).toBe(false)
  expect(requested.some((path) => path.includes('/api/system/update'))).toBe(
    false,
  )

  // Copying uses the session credential directly — no reveal request.
  await page
    .getByRole('button', { name: 'Copy access key' })
    .click()
  await expect(
    page.getByRole('status').filter({ hasText: 'Access key copied' }),
  ).toBeVisible()
  expect(requests.reveals).toHaveLength(0)
  await expect
    .poll(async () =>
      page.evaluate(() => navigator.clipboard.readText()),
    )
    .toBe('e2e-auth-key')
})

test('copies the access key through the admin reveal flow', async ({
  page,
}) => {
  const requests = await mockHome(page)
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto('/', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  await page
    .getByRole('button', { name: 'Copy access key' })
    .click()
  await expect(
    page.getByRole('status').filter({ hasText: 'Access key copied' }),
  ).toBeVisible()
  expect(requests.reveals).toEqual([1])
  await expect
    .poll(async () =>
      page.evaluate(() => navigator.clipboard.readText()),
    )
    .toBe('sk-e2e-revealed-1')
})

test('shows the manual copy dialog when the clipboard is unavailable', async ({
  page,
}) => {
  const requests = await mockHome(page)
  await page.addInitScript(() => {
    Object.defineProperty(window.navigator, 'clipboard', {
      value: { writeText: () => Promise.reject(new Error('denied')) },
      configurable: true,
    })
    // execCommand fallback fails too — the dialog must surface the secret.
    Document.prototype.execCommand = () => false
  })
  await page.goto('/', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  await page
    .getByRole('button', { name: 'Copy access key' })
    .click()
  const dialog = page.getByRole('dialog', { name: 'Copy content' })
  await expect(dialog).toBeVisible()
  await expect(dialog.locator('input')).toHaveValue('sk-e2e-revealed-1')
  expect(requests.reveals).toEqual([1])
})

test('selects gateway clients through the canonical query and flags incompatible ones', async ({
  page,
}) => {
  await mockHome(page)
  await page.goto('/', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  // Codex needs openai-responses: fine for prod-key.
  await page.getByRole('button', { name: 'Select a client' }).click()
  const picker = page.getByRole('dialog', { name: 'Select a client' })
  await expect(picker).toBeVisible()
  await picker.getByRole('button', { name: 'Codex', exact: true }).click()
  await expect(page).toHaveURL(/[?&]client=codex/)
  await expect(
    page.getByRole('heading', { name: 'Connect to the gateway' }),
  ).toBeVisible()
  await expect(page.getByText('wire_api = "responses"')).toBeVisible()

  // Switch to the completions-only key: codex lands in the unsupported group
  // and the panel reports the missing protocol.
  await page.getByRole('combobox', { name: 'Access key' }).click()
  await page
    .getByRole('option', { name: /dev-key/ })
    .click()
  await expect(page).toHaveURL(/access_key_id=2/)
  await expect(
    page
      .getByRole('status')
      .filter({
        hasText: 'Codex requires openai-responses for this access key',
      }),
  ).toBeVisible()
})

test('confirms quick import before requesting the custom scheme', async ({
  page,
}) => {
  const requests = await mockHome(page)
  await page.goto('/', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  // Default client cc-switch, default target Claude Code (no model needed).
  const importButton = page.getByRole('button', {
    name: 'Import and enable',
    exact: true,
  })
  await expect(importButton).toBeEnabled()
  await importButton.click()

  const confirm = page.getByRole('alertdialog', {
    name: 'Open CC Switch · Claude Code',
  })
  await expect(confirm).toBeVisible()
  await confirm
    .getByRole('button', { name: 'Import and enable' })
    .click()

  // The reveal ran and the popup was asked to navigate to ccswitch:// — the
  // page reports success even though no real handler exists in the browser.
  await expect.poll(() => requests.reveals.length).toBe(1)
  await expect(
    page
      .getByRole('status')
      .filter({
        hasText:
          'Requested CC Switch · Claude Code to open. Confirm the import in the app',
      }),
  ).toBeVisible()
})
