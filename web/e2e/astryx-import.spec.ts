import { expect, test, type Page, type Route } from '@playwright/test'

// Phase 4: the React /import surface — mode switching, group_id deep links,
// stable idempotent submission and the
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

    param_fields: [],
    credential_fields: [],
    connection: {
      type: 'api_key',
      credential_input: 'batch_text',
    },
    capabilities: {
      model_discovery: false,

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
    channel_id: 'cliproxyapi',
    name: 'CLIProxyAPI',
    mark: 'CPA',
    icon: 'cpa',
    search_terms: ['cpa'],
    description: '',
    default_base_url: '',

    param_fields: [],
    credential_fields: [],
    connection: {
      type: 'api_key',
      credential_input: 'batch_text',
    },
    capabilities: {
      model_discovery: false,

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

const PICKER_CHANNELS = [
  ['openai', 'OpenAI'],
  ['anthropic', 'Anthropic'],
  ['gemini', 'Gemini'],
  ['openai_compatible', 'OpenAI Compatible'],
  ['cliproxyapi', 'CLIProxyAPI'],
  ['deepseek', 'DeepSeek'],
  ['groq', 'Groq'],
  ['mistral', 'Mistral'],
  ['extra', 'Extra Channel'],
].map(([channel_id, name]) => ({
  ...CHANNELS[0],
  channel_id,
  name,
  icon: channel_id,
  param_fields: [
    {
      key: 'base_url',
      label: 'Base URL',
      input_kind: 'url',
      required: channel_id === 'openai_compatible',
      sensitive: false,
      default_value: null,
    },
  ],
  credential_fields: [
    {
      key: 'api_key',
      label: 'API Key',
      input_kind: 'secret',
      required: true,
      sensitive: true,
      default_value: null,
    },
  ],
}))

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
    name: 'Beta Gateway',
    channel_id: 'cliproxyapi',
    connection_type: 'api_key',
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

    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    service_status: 'available',
    service_status_reason: null,
    credential_configured: true,
    credential_status: 'available',
    model_count: 0,
  },
  settings: {
    name: 'Imported Group',

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
}

async function mockImportApi(
  page: Page,
  options: {
    createStatus?: number
    credentialCount?: number
  } = {},
): Promise<ImportRequests> {
  const requests: ImportRequests = {
    createBodies: [],
    createIdempotencyKeys: [],
    credentialImports: [],
  }
  const ok = (data: unknown) => ({ code: 0, message: 'ok', data })
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
        body: JSON.stringify(ok(data)),
      })
    if (path === '/api/auth/session') return fulfill(AUTH_SESSION)
    if (path === '/api/channels') return fulfill({ items: CHANNELS, total: CHANNELS.length })
    if (path === '/api/groups/options') return fulfill(GROUP_OPTIONS)
    if (path === '/api/groups/7')
      return fulfill({
        ...GROUP_42.summary,
        id: 7,
        name: 'Alpha Group',
        credential_configured: (options.credentialCount ?? 0) > 0,
        credential_status: (options.credentialCount ?? 0) > 0 ? 'available' : null,
      })
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
        credential_id: 9,
      })
    }
    const credentialImport = path.match(/^\/api\/groups\/(\d+)\/credential$/)
    if (credentialImport && request.method() === 'POST') {
      requests.credentialImports.push({
        groupID: credentialImport[1]!,
        body: (request.postDataJSON() ?? {}) as Record<string, unknown>,
        key: request.headers()['idempotency-key'] ?? '',
      })
      return fulfill({
        group_id: Number(credentialImport[1]),
        credential_id: 9,
      })
    }
    if (path === '/api/groups/42') return fulfill(GROUP_42.summary)
    if (path === '/api/groups/42/settings') return fulfill(GROUP_42.settings)
    if (path === '/api/groups/42/models') return fulfill(GROUP_42.models)
    return fulfill({})
  })
  return requests
}

test('API-only picker retains external CPA without subscription authorization controls', async ({
  page,
}) => {
  await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })
  const externalGateway = page.getByRole('button', { name: 'CLIProxyAPI', exact: true })
  await expect(externalGateway).toBeVisible({ timeout: FIRST_PAINT })
  await externalGateway.click()
  await expect(externalGateway).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('radio', { name: 'Subscription account' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Sign in with/ })).toHaveCount(0)
})

test('visible group name label focuses its input', async ({ page }) => {
  await mockImportApi(page)
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  const label = page
    .locator('label')
    .filter({ hasText: /^Group name/ })
    .filter({ visible: true })
  await expect(label).toBeVisible({ timeout: FIRST_PAINT })
  await label.click()
  await expect(page.locator(`input[id="${await label.getAttribute('for')}"]`)).toBeFocused()
})

test('channel search exposes its list and selects an option with the keyboard', async ({
  page,
}) => {
  await mockImportApi(page)
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: {
          items: [
            ...CHANNELS,
            ...['anthropic', 'gemini', 'openai_compatible', 'extra'].map((channel_id) => ({
              ...CHANNELS[0],
              channel_id,
              name: channel_id,
            })),
          ],
          total: 6,
        },
      },
    }),
  )
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await page
    .getByRole('button', { name: 'Other channels', exact: true })
    .click({ timeout: FIRST_PAINT })
  const search = page.getByRole('combobox', { name: /Search channels/ })
  await expect(search).toHaveAttribute('aria-expanded', 'true')
  const list = page.getByRole('listbox', { name: 'Other channels' })
  await expect(list).toBeVisible()
  await expect(search).toHaveAttribute('aria-controls', (await list.getAttribute('id')) as string)
  await search.fill('extra')
  await expect(search).toHaveAttribute('aria-activedescendant', /channel-extra$/)
  await search.press('Enter')
  await expect(page.getByRole('button', { name: 'extra', exact: true })).toBeVisible()
  await expect(list).not.toBeVisible()
})

for (const width of [390, 320]) {
  test(`channel picker contents fit the popup and viewport at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 })
    await mockImportApi(page)
    await page.route('**/api/channels', (route) =>
      route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: { items: PICKER_CHANNELS, total: PICKER_CHANNELS.length },
        },
      }),
    )
    await page.goto('/import?mode=new', { waitUntil: 'commit' })
    await page.getByRole('button', { name: 'Other channels', exact: true }).click({
      timeout: FIRST_PAINT,
    })
    const popup = page.getByRole('dialog', { name: 'Other channels', exact: true })
    const search = popup.getByRole('combobox', { name: /Search channels/ })
    await expect(search).toBeFocused()
    await expect(popup.getByRole('option')).toHaveCount(5)
    // Check inner controls and icons, not just the outer surface: a fitting
    // popover can still clip an oversized inner panel or scroll it sideways.
    await expect
      .poll(() =>
        popup.evaluate((surface) => {
          const layer = surface.closest('[popover]')!
          const outer = layer.getBoundingClientRect()
          const left = Math.max(0, outer.left)
          const right = Math.min(window.innerWidth, outer.right)
          const top = Math.max(0, outer.top)
          const bottom = Math.min(window.innerHeight, outer.bottom)
          return Array.from(
            surface.querySelectorAll(
              '[role="combobox"], [role="listbox"], [role="option"], .channel-icon, svg',
            ),
          )
            .filter((element) => {
              const rect = element.getBoundingClientRect()
              return (
                rect.left < left - 1 ||
                rect.right > right + 1 ||
                rect.top < top - 1 ||
                rect.bottom > bottom + 1
              )
            })
            .map((element) => element.getAttribute('role') ?? element.className.toString())
        }),
      )
      .toEqual([])
    await search.fill('extra')
    await search.press('Enter')
    await expect(page.getByRole('button', { name: 'Extra Channel', exact: true })).toBeVisible()
    await expect(popup).not.toBeVisible()
  })
}

test('unmatched channel search explains the compatible alternative and selects it by keyboard', async ({
  page,
}) => {
  await mockImportApi(page)
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: { items: PICKER_CHANNELS, total: PICKER_CHANNELS.length },
      },
    }),
  )
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await page.getByRole('button', { name: 'Other channels', exact: true }).click({
    timeout: FIRST_PAINT,
  })
  const popup = page.getByRole('dialog', { name: 'Other channels', exact: true })
  const search = popup.getByRole('combobox', { name: /Search channels/ })
  await search.fill('zzzznonexistent')
  await expect(popup.getByText('No channels match', { exact: true })).toBeVisible()
  await expect(
    popup.getByText('Use this option only if the provider supports the OpenAI API.', {
      exact: true,
    }),
  ).toBeVisible()
  const alternative = popup.getByRole('option', {
    name: 'Use OpenAI-compatible channel',
    exact: true,
  })
  await expect(alternative).toBeVisible()
  await expect(search).toHaveAttribute(
    'aria-activedescendant',
    (await alternative.getAttribute('id'))!,
  )
  await search.press('Enter')
  await expect(popup).not.toBeVisible()
  await expect(
    page.getByRole('button', { name: 'OpenAI Compatible', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true')
})

test('adds the first manual model from the empty import form', async ({ page }) => {
  await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })
  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await page.getByRole('button', { name: 'Add model', exact: true }).click()
  await expect(page.locator('[data-model-id-index="0"]')).toBeVisible()
  await page.locator('[data-model-id-index="0"]').fill('ux-first-model')
  await page.getByRole('button', { name: 'Add model', exact: true }).click()
  await expect(page.locator('[data-model-id-index="1"]')).toBeVisible()
})

test('single credential rejects multiple lines including duplicate keys', async ({ page }) => {
  const requests = await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })
  const input = page.locator('#channel-credentials')
  await expect(input).toBeVisible({ timeout: FIRST_PAINT })
  await input.fill('sk-first\nsk-second')
  await expect(input).toHaveAttribute('aria-invalid', 'true')
  await expect(page.getByRole('button', { name: 'Create Group' })).toBeDisabled()
  expect(requests.createBodies).toHaveLength(0)
  await input.fill('sk-first\nsk-first')
  await expect(page.getByRole('button', { name: 'Create Group' })).toBeDisabled()
  await input.fill('sk-first')
  await expect(page.getByRole('button', { name: 'Create Group' })).toBeEnabled()
})

test('one formatted OpenAI JSON credential can be submitted unchanged', async ({ page }) => {
  const requests = await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })
  const input = page.locator('#channel-credentials')
  await expect(input).toBeVisible({ timeout: FIRST_PAINT })
  const credential = '{\n  "api_key": "test-json-placeholder"\n}'
  await input.fill(credential)
  await expect(input).not.toHaveAttribute('aria-invalid', 'true')
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect.poll(() => requests.createBodies.length).toBe(1)
  expect(requests.createBodies[0]!.credential).toBe(credential)
})

test('existing Group summary failure is visible and retry recovers the first import', async ({
  page,
}) => {
  await mockImportApi(page)
  let fail = true
  await page.route('**/api/groups/7', async (route) => {
    await route.fulfill({
      status: fail ? 500 : 200,
      contentType: 'application/json',
      body: JSON.stringify(
        fail
          ? { code: 'INTERNAL_SERVER_ERROR', message: 'failed', data: null }
          : {
              code: 0,
              message: 'ok',
              data: {
                ...GROUP_42.summary,
                id: 7,
                name: 'Alpha Group',
                credential_configured: false,
                credential_status: null,
              },
            },
      ),
    })
  })
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })
  await expect(page.getByText('Unable to load Group details.', { exact: true })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await expect(page.locator('#channel-credentials')).toBeDisabled()
  fail = false
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.locator('#channel-credentials')).toBeEnabled()
  await page.locator('#channel-credentials').fill('sk-first\nsk-second')
  await expect(page.getByRole('button', { name: 'Import credential', exact: true })).toBeDisabled()
  await page.locator('#channel-credentials').fill('sk-first')
  await expect(page.getByRole('button', { name: 'Import credential', exact: true })).toBeEnabled()
})

test('same-target conflict offers management, never append', async ({ page }) => {
  const requests = await mockImportApi(page)
  await page.route('**/api/groups', async (route) => {
    await route.fulfill({
      status: 409,
      contentType: 'application/json',
      body: JSON.stringify({
        code: 'CHANNEL_TARGET_CONFLICT',
        message: 'conflict',
        data: { groups: [{ id: 7, name: 'Alpha Group' }] },
      }),
    })
  })
  await page.goto('/import', { waitUntil: 'commit' })
  await expect(page.locator('#channel-credentials')).toBeVisible({ timeout: FIRST_PAINT })
  await page.locator('#channel-credentials').fill('sk-first')
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Manage Group', exact: true })).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Import credentials here', exact: true }),
  ).toHaveCount(0)
  await page.getByRole('button', { name: 'Manage Group', exact: true }).click()
  await page.getByRole('button', { name: 'Discard changes', exact: true }).click()
  await expect(page).toHaveURL(/\/groups\/7$/)
  expect(requests.credentialImports).toHaveLength(0)
})

test('single-credential server rejection explains how to correct the draft', async ({ page }) => {
  await mockImportApi(page)
  await page.route('**/api/groups', async (route) => {
    await route.fulfill({
      status: 409,
      contentType: 'application/json',
      body: JSON.stringify({ code: 'SINGLE_CREDENTIAL_REQUIRED', message: 'only one', data: null }),
    })
  })
  await page.goto('/import', { waitUntil: 'commit' })
  await expect(page.locator('#channel-credentials')).toBeVisible({ timeout: FIRST_PAINT })
  await page.locator('#channel-credentials').fill('sk-first')
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect(
    page.getByText('Keep only one credential, even if the lines are identical.', { exact: true }),
  ).toBeVisible()
  await expect(page.getByText('Unable to create the Group', { exact: true })).toHaveCount(0)
})

test('mode switch swaps views and canonicalizes the URL', async ({ page }) => {
  await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await expect(page.getByRole('radio', { name: 'New Group' })).toHaveAttribute(
    'aria-checked',
    'true',
  )

  await page.getByRole('radio', { name: 'Existing Group' }).click()
  await expect(page).toHaveURL(/\/import\?.*mode=existing/)
  await expect(page.getByRole('combobox', { name: 'Group' })).toBeVisible()
})

test('group_id deep link lands in existing mode with the group preselected', async ({ page }) => {
  await mockImportApi(page)
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })

  await expect(page.getByRole('radio', { name: 'Existing Group' })).toHaveAttribute(
    'aria-checked',
    'true',
    { timeout: FIRST_PAINT },
  )
  await expect(page.getByRole('combobox', { name: 'Group' })).toContainText('Alpha Group')
})

test('empty existing-group import posts one credential with a stable idempotency key', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })

  const submit = page.getByRole('button', { name: 'Import credential', exact: true })
  await expect(submit).toBeVisible({ timeout: FIRST_PAINT })
  await page.locator('#channel-credentials').fill('sk-alpha-1')
  await expect(submit).toBeEnabled()
  await submit.click()

  await expect.poll(() => requests.credentialImports.length).toBe(1)
  expect(requests.credentialImports[0]!.groupID).toBe('7')
  expect(requests.credentialImports[0]!.key.length).toBeGreaterThan(0)
  expect(String(requests.credentialImports[0]!.body.credential)).toContain('sk-alpha-1')
})

test('populated existing Group guides management instead of importing or replacing', async ({
  page,
}) => {
  const requests = await mockImportApi(page, { credentialCount: 1 })
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })
  const manage = page.getByRole('link', { name: 'Manage credential', exact: true })
  await expect(manage).toBeVisible({ timeout: FIRST_PAINT })
  await expect(manage).toHaveAttribute('href', '/groups/7')
  await expect(page.locator('#channel-credentials')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Import credential', exact: true })).toHaveCount(0)
  await manage.click()
  await expect(page).toHaveURL(/\/groups\/7$/)
  expect(requests.credentialImports).toHaveLength(0)
})

test('new-group create posts one credential with an idempotency key and navigates to the group', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  // First channel is auto-adopted; credentials make the form submittable.
  await page.locator('#channel-credentials').fill('sk-live-1')
  await page.getByRole('button', { name: 'Create Group' }).click()

  await expect.poll(() => requests.createBodies.length).toBe(1)
  const body = requests.createBodies[0]!
  expect(body.channel_id).toBe('openai')
  expect(body.credential).toBe('sk-live-1')
  expect(body).not.toHaveProperty('credentials')
  expect(requests.createIdempotencyKeys[0]!.length).toBeGreaterThan(0)

  await page.waitForURL(/\/groups\/42/, { timeout: 15_000 })
})

test('a 401 captures the draft to sessionStorage and re-login restores it', async ({
  page,
  context,
}) => {
  await mockImportApi(page, { createStatus: 401 })
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await page.locator('#channel-credentials').fill('sk-rescue-1')
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
  await page.getByRole('textbox', { name: 'Sign-in key', exact: true }).fill('e2e-auth-key')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await page.waitForURL(/\/import/, { timeout: 15_000 })
  await expect(page.locator('#channel-credentials')).toHaveValue(/sk-rescue-1/, {
    timeout: 15_000,
  })
  // The draft is single-consumption: sessionStorage no longer holds it.
  await expect
    .poll(() => page.evaluate(() => window.sessionStorage.getItem('gpt-load.import-reauth-draft')))
    .toBeNull()
  void context
})

test('import discovery runs against the entered connection and appends candidates', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  const discoveries: unknown[] = []
  // The default OpenAI fixture advertises no discovery capability; enable it
  // so the publish step exposes the discovery entry point.
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: {
          items: [
            {
              ...CHANNELS[0]!,
              capabilities: { ...CHANNELS[0]!.capabilities, model_discovery: true },
            },
            ...CHANNELS.slice(1),
          ],
          total: CHANNELS.length,
        },
      },
    }),
  )
  await page.route('**/api/models/discover', async (route) => {
    discoveries.push(route.request().postDataJSON())
    return route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: {
          models: [
            {
              id: 'gpt-discovered',
              name: 'GPT Discovered',
              sources: ['live'],
              pricing_status: 'pending',
              pricing_source: null,
            },
            {
              id: 'gpt-priced',
              name: 'GPT Priced',
              sources: ['catalog'],
              pricing_status: 'configured',
              pricing_source: 'OpenAI',
            },
          ],
        },
      },
    })
  })
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await page.locator('#channel-credentials').fill('sk-discover')

  const discover = page.getByRole('button', { name: 'Get upstream models' })
  await expect(discover).toBeEnabled()
  await discover.click()

  const dialog = page.getByRole('dialog', { name: 'Get upstream models' })
  await expect(dialog).toBeVisible()

  // Runtime evidence: the discovery request carries the entered credential.
  await expect.poll(() => discoveries.length).toBe(1)
  expect(discoveries[0]).toMatchObject({
    channel_id: 'openai',
    connection_type: 'api_key',
    credentials: 'sk-discover',
  })

  // Evidence legend stays readable without tooltips.
  const legend = dialog.getByLabel('Evidence legend')
  await expect(legend).toBeVisible()
  await expect(legend.getByText('Live discovery', { exact: true })).toBeVisible()

  await expect(dialog.getByText('0 of 2 available models selected')).toBeVisible()
  await dialog
    .getByRole('checkbox', { name: /^GPT Discovered GPT Discovered gpt-discovered/ })
    .click()
  await dialog.getByRole('checkbox', { name: /^GPT Priced GPT Priced gpt-priced/ }).click()
  await expect(dialog.getByText('2 of 2 available models selected')).toBeVisible()

  await dialog.getByRole('button', { name: 'Add to current list' }).click()
  await expect(dialog).not.toBeVisible()
  const modelTable = page.getByRole('table', { name: 'Models to publish' })
  await expect(modelTable.getByRole('code')).toHaveText(['gpt-discovered', 'gpt-priced'])
  expect(requests.createBodies).toHaveLength(0)
})
