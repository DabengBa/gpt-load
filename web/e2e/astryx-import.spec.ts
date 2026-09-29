import { expect, test, type Page, type Route } from '@playwright/test'

// Phase 4: the React /import surface — mode switching, group_id deep links,
// stable idempotent submission, subscription staging polling, and the
// sessionStorage re-auth recovery chain (401 -> capture -> login -> restore).

// The first navigation in a cold run pays vite's on-demand compile of the
// astryx import graph; keep generous headroom over the usual 90s budget.
test.setTimeout(120_000)
const FIRST_PAINT = 90_000

const AUTH_SESSION = { authenticated: true, principal_type: 'admin' }

const CHANNELS = [
  {
    channel_id: 'openai',
    name: 'OpenAI',
    mark: 'OA',
    icon: 'openai',
    search_terms: ['openai'],
    description: '',
    default_base_url: 'https://api.openai.com',
    notices: [],
    param_fields: [],
    credential_fields: [],
    connection: {
      type: 'api_key',
      credential_input: 'batch_text',
      authorization_methods: [],
    },
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
  },
  {
    channel_id: 'claude_sub',
    name: 'Claude Subscription',
    mark: 'CS',
    icon: 'claude',
    search_terms: ['claude'],
    description: '',
    default_base_url: 'https://api.anthropic.com',
    notices: [],
    param_fields: [],
    credential_fields: [],
    connection: {
      type: 'subscription',
      credential_input: 'authorization',
      authorization_methods: ['browser_oauth'],
    },
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
  },
]

const GROUP_OPTIONS = [
  {
    id: 7,
    name: 'Alpha Group',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    enabled: true,
    models: ['gpt-4o'],
  },
  {
    id: 9,
    name: 'Beta Subscription',
    channel_id: 'claude_sub',
    connection_type: 'subscription',
    params: {},
    provider_url: null,
    enabled: true,
    models: [],
  },
]

const RUNTIME = {
  first_byte_timeout: 30,
  request_timeout: 60,
  stream_idle_timeout: 30,
  blacklist_threshold: 3,
  header_rules: { set: {}, remove: [] },
  affinity_enabled: false,
  responses_websocket_enabled: false,
  responses_reasoning_status_filter_enabled: false,
}

const GROUP_42 = {
  summary: {
    id: 42,
    name: 'Imported Group',
    price_multiplier: '1',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    service_status: 'available',
    service_status_reason: null,
    credential_count: 2,
    model_count: 0,
  },
  settings: {
    name: 'Imported Group',
    price_multiplier: '1',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    enabled: true,
    overrides: { ...RUNTIME, parameter_overrides: [] },
    effective: RUNTIME,
    proxy: {
      configured_mode: 'inherit',
      effective_mode: 'direct',
      effective_source: 'default',
      has_auth: false,
    },
  },
  models: { items: [], total: 0, pending: 0 },
}

interface ImportRequests {
  createBodies: Record<string, unknown>[]
  createIdempotencyKeys: string[]
  credentialImports: { groupID: string; body: Record<string, unknown>; key: string }[]
  stageAuthorizations: Record<string, unknown>[]
  stagePolls: string[]
}

async function mockImportApi(
  page: Page,
  options: { createStatus?: number; pollSequence?: Record<string, unknown>[] } = {},
): Promise<ImportRequests> {
  const requests: ImportRequests = {
    createBodies: [],
    createIdempotencyKeys: [],
    credentialImports: [],
    stageAuthorizations: [],
    stagePolls: [],
  }
  const ok = (data: unknown) => ({ code: 0, message: 'ok', data })
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
    // OAuth popup stand-in: the stager needs `location.replace` and `close`.
    window.open = () =>
      ({ closed: false, close() {}, location: { replace() {} } }) as unknown as Window
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route: Route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const fulfill = (data: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(ok(data)),
      })
    if (path === '/api/auth/session') return fulfill(AUTH_SESSION)
    if (path === '/api/channels') return fulfill({ items: CHANNELS, total: CHANNELS.length })
    if (path === '/api/groups/options') return fulfill(GROUP_OPTIONS)
    if (path === '/api/groups' && request.method() === 'POST') {
      requests.createBodies.push((request.postDataJSON() ?? {}) as Record<string, unknown>)
      requests.createIdempotencyKeys.push(request.headers()['idempotency-key'] ?? '')
      if (options.createStatus === 401) {
        return route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({ code: 401, message: 'unauthorized', data: null }),
        })
      }
      return fulfill({
        group_id: 42,
        group_name: 'Imported Group',
        credentials_added: 2,
        credentials_duplicated: 0,
      })
    }
    const credentialImport = path.match(/^\/api\/groups\/(\d+)\/credentials\/import$/)
    if (credentialImport && request.method() === 'POST') {
      requests.credentialImports.push({
        groupID: credentialImport[1]!,
        body: (request.postDataJSON() ?? {}) as Record<string, unknown>,
        key: request.headers()['idempotency-key'] ?? '',
      })
      return fulfill({ credentials_added: 2, credentials_duplicated: 0 })
    }
    if (path === '/api/credential-stages/authorizations' && request.method() === 'POST') {
      requests.stageAuthorizations.push(
        (request.postDataJSON() ?? {}) as Record<string, unknown>,
      )
      return fulfill({
        stage_id: 'stage_abc',
        status: 'pending_authorization',
        authorization_method: 'browser_oauth',
        authorization_url: 'https://auth.example.com/oauth',
        redirect_uri: 'http://127.0.0.1/oauth/callback',
        expires_at_ms: Date.now() + 600_000,
        account: {},
      })
    }
    const stageGet = path.match(/^\/api\/credential-stages\/([a-zA-Z0-9_-]+)$/)
    if (stageGet && request.method() === 'GET') {
      requests.stagePolls.push(stageGet[1]!)
      const next = options.pollSequence?.[requests.stagePolls.length - 1]
      return fulfill(
        next ?? {
          stage_id: stageGet[1],
          status: 'ready',
          authorization_method: 'browser_oauth',
          expires_at_ms: Date.now() + 600_000,
          account: { email_mask: 't***@example.com' },
        },
      )
    }
    if (path === '/api/groups/42') return fulfill(GROUP_42.summary)
    if (path === '/api/groups/42/settings') return fulfill(GROUP_42.settings)
    if (path === '/api/groups/42/models') return fulfill(GROUP_42.models)
    return fulfill({})
  })
  return requests
}

test('mode switch swaps views and canonicalizes the URL', async ({ page }) => {
  await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await expect(
    page.getByRole('radio', { name: 'New Group' }),
  ).toHaveAttribute('aria-checked', 'true')

  await page.getByRole('radio', { name: 'Existing Group' }).click()
  await expect(page).toHaveURL(/\/import\?.*mode=existing/)
  await expect(page.getByRole('combobox', { name: 'Group' })).toBeVisible()
})

test('group_id deep link lands in existing mode with the group preselected', async ({
  page,
}) => {
  await mockImportApi(page)
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })

  await expect(
    page.getByRole('radio', { name: 'Existing Group' }),
  ).toHaveAttribute('aria-checked', 'true', { timeout: FIRST_PAINT })
  await expect(page.getByRole('combobox', { name: 'Group' })).toContainText('Alpha Group')
})

test('existing-group import posts credentials with a stable idempotency key', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })

  const submit = page.getByRole('button', { name: 'Add credentials' })
  await expect(submit).toBeVisible({ timeout: FIRST_PAINT })
  await page.locator('#channel-credentials').fill('sk-alpha-1\nsk-alpha-2')
  await expect(submit).toBeEnabled()
  await submit.click()

  await expect.poll(() => requests.credentialImports.length).toBe(1)
  expect(requests.credentialImports[0]!.groupID).toBe('7')
  expect(requests.credentialImports[0]!.key.length).toBeGreaterThan(0)
  expect(String(requests.credentialImports[0]!.body.credentials)).toContain('sk-alpha-1')
})

test('new-group create posts once with an idempotency key and navigates to the group', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(
    page.getByRole('heading', { name: 'Import channel credentials' }),
  ).toBeVisible({ timeout: FIRST_PAINT })
  // First channel is auto-adopted; credentials make the form submittable.
  await page.locator('#channel-credentials').fill('sk-live-1\nsk-live-2')
  await page.getByRole('button', { name: 'Create Group' }).click()

  await expect.poll(() => requests.createBodies.length).toBe(1)
  const body = requests.createBodies[0]!
  expect(body.channel_id).toBe('openai')
  expect(String(body.credentials)).toContain('sk-live-1')
  expect(requests.createIdempotencyKeys[0]!.length).toBeGreaterThan(0)

  await page.waitForURL(/\/groups\/42/, { timeout: 15_000 })
})

test('a 401 captures the draft to sessionStorage and re-login restores it', async ({
  page,
  context,
}) => {
  await mockImportApi(page, { createStatus: 401 })
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(
    page.getByRole('heading', { name: 'Import channel credentials' }),
  ).toBeVisible({ timeout: FIRST_PAINT })
  await page.locator('#channel-credentials').fill('sk-rescue-1\nsk-rescue-2')
  await page.getByRole('button', { name: 'Create Group' }).click()

  // The unauthorized handler captures the draft, clears the session, and
  // bounces to /login?redirect=...
  await page.waitForURL(/\/login\?.*redirect=/, { timeout: 15_000 })
  const stored = await page.evaluate(() =>
    window.sessionStorage.getItem('gpt-load.import-reauth-draft'),
  )
  expect(stored).not.toBeNull()
  const draft = JSON.parse(stored!) as { draft: { mode: string; credentials: string } }
  expect(draft.draft.mode).toBe('new')
  expect(draft.draft.credentials).toContain('sk-rescue-1')

  // Re-auth: the session endpoint still answers admin, so signing in lands
  // back on /import and the consumed draft repopulates the form.
  await page.evaluate(() => window.localStorage.setItem('gpt-load.auth-key', 'e2e-auth-key'))
  await page.getByLabel('Sign-in key', { exact: true }).fill('e2e-auth-key')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await page.waitForURL(/\/import/, { timeout: 15_000 })
  await expect(page.locator('#channel-credentials')).toHaveValue(/sk-rescue-1/, {
    timeout: 15_000,
  })
  // The draft is single-consumption: sessionStorage no longer holds it.
  await expect
    .poll(() =>
      page.evaluate(() => window.sessionStorage.getItem('gpt-load.import-reauth-draft')),
    )
    .toBeNull()
  void context
})

test('subscription channel stages an account through authorization polling', async ({
  page,
}) => {
  const requests = await mockImportApi(page, {
    pollSequence: [
      {
        stage_id: 'stage_abc',
        status: 'pending_authorization',
        authorization_method: 'browser_oauth',
        authorization_url: 'https://auth.example.com/oauth',
        redirect_uri: 'http://127.0.0.1/oauth/callback',
        expires_at_ms: Date.now() + 600_000,
        account: {},
      },
    ],
  })
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(
    page.getByRole('heading', { name: 'Import channel credentials' }),
  ).toBeVisible({ timeout: FIRST_PAINT })
  await page.getByRole('radio', { name: 'Subscription account' }).click()
  await page
    .getByRole('button', { name: 'Sign in with Claude Subscription' })
    .click()

  await expect.poll(() => requests.stageAuthorizations.length).toBe(1)
  expect(requests.stageAuthorizations[0]!.channel_id).toBe('claude_sub')
  await expect.poll(() => requests.stagePolls.length, { timeout: 10_000 }).toBeGreaterThan(0)
  await expect(page.getByText('t***@example.com')).toBeVisible({ timeout: 15_000 })
})
