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
  stageAuthorizations: Record<string, unknown>[]
  stagePolls: string[]
}

async function mockImportApi(
  page: Page,
  options: {
    createStatus?: number
    credentialCount?: number
    pollSequence?: Record<string, unknown>[]
  } = {},
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
    if (path === '/api/credential-stages/authorizations' && request.method() === 'POST') {
      requests.stageAuthorizations.push((request.postDataJSON() ?? {}) as Record<string, unknown>)
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

for (const authState of [
  'reauthorization_required',
  'outcome_unknown',
  'available',
  'empty',
  'delete',
  'unauthorized',
  'mismatch',
]) {
  test(`existing subscription ${authState} configures only one stage`, async ({ page }) => {
    const requests = await mockImportApi(page)
    const connections: { body: unknown; key: string; expectedID: string }[] = []
    let configured = authState !== 'empty'
    let deletes = 0
    await page.route('**/api/groups/9', (route) =>
      route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: {
            ...GROUP_42.summary,
            id: 9,
            name: 'Beta Subscription',
            channel_id: 'claude_sub',
            connection_type: 'subscription',
            credential_configured: configured,
            credential_status: !configured
              ? null
              : ['available', 'delete'].includes(authState)
                ? 'available'
                : 'disabled',
          },
        },
      }),
    )
    await page.route('**/api/groups/9/credential', async (route) => {
      if (route.request().method() === 'DELETE') {
        expect(route.request().headers()['x-credential-id']).toBe('99')
        configured = false
        deletes += 1
      }
      return route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: {
            credential: !configured
              ? null
              : {
                  credential_id: 99,
                  connection_type: 'subscription',
                  auth_state: ['available', 'delete'].includes(authState)
                    ? 'ready'
                    : ['unauthorized', 'mismatch'].includes(authState)
                      ? 'reauthorization_required'
                      : authState,
                  secret_version: 1,
                  mask: '***',
                  account: { email_mask: 't***@example.com' },
                  effective_status: ['available', 'delete'].includes(authState)
                    ? 'available'
                    : 'disabled',
                  recent_success_count: 0,
                  recent_failure_count: 0,
                  consecutive_failure_count: 0,
                  last_failure_category: 'ok',
                  last_status_code: null,
                  cooldown_until_ms: null,
                  recovery: { mode: 'none', automatic: false, at_ms: null },
                },
            observation: null,
          },
        },
      })
    })
    await page.route('**/api/groups/9/settings', (route) =>
      route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: {
            ...GROUP_42.settings,
            name: 'Beta Subscription',
            channel_id: 'claude_sub',
            connection_type: 'subscription',
          },
        },
      }),
    )
    await page.route('**/api/groups/9/models', (route) =>
      route.fulfill({ json: { code: 0, message: 'ok', data: GROUP_42.models } }),
    )
    await page.route('**/api/groups/9/credential/connect', async (route) => {
      const expectedID = route.request().headers()['x-credential-id'] ?? ''
      expect(expectedID).toBe(['empty', 'delete'].includes(authState) ? '0' : '99')
      connections.push({
        body: route.request().postDataJSON(),
        key: route.request().headers()['idempotency-key'] ?? '',
        expectedID,
      })
      if (authState === 'mismatch')
        return route.fulfill({
          status: 409,
          json: {
            code: 'SINGLE_CREDENTIAL_REQUIRED',
            message: 'same account required',
            data: null,
          },
        })
      if (authState === 'unauthorized' && connections.length === 1)
        return route.fulfill({
          status: 401,
          json: { code: 401, message: 'unauthorized', data: null },
        })
      if (connections.length === 1)
        return route.fulfill({
          status: 503,
          json: {
            code: 'TEMPORARILY_UNAVAILABLE',
            message: 'retry',
            data: null,
          },
        })
      return route.fulfill({
        json: { code: 0, message: 'ok', data: { group_id: 9, credential_id: 99 } },
      })
    })
    if (authState === 'delete') {
      await page.goto('/groups/9?tab=credentials', { waitUntil: 'commit' })
      await expect(page.getByRole('button', { name: 'More actions' })).toBeVisible({
        timeout: FIRST_PAINT,
      })
      await page.getByRole('button', { name: 'More actions' }).click()
      await page.getByRole('button', { name: 'Delete', exact: true }).click()
      await page
        .getByRole('alertdialog')
        .getByRole('button', { name: 'Delete', exact: true })
        .click()
      await expect.poll(() => deletes).toBe(1)
      await expect(page.getByText('No keys yet', { exact: true })).toBeVisible()
      await page.getByRole('button', { name: 'Add key', exact: true }).click()
    } else await page.goto('/import?mode=existing&group_id=9', { waitUntil: 'commit' })
    if (authState === 'available') {
      await expect(
        page.getByText('This group already has a credential', { exact: false }),
      ).toBeVisible({ timeout: FIRST_PAINT })
      await expect(
        page.getByRole('button', { name: 'Sign in with Claude Subscription' }),
      ).toHaveCount(0)
      expect(connections).toHaveLength(0)
      return
    }
    const authorize = page.getByRole('button', { name: 'Sign in with Claude Subscription' })
    await expect(authorize).toBeVisible({ timeout: FIRST_PAINT })
    await authorize.click()
    await expect.poll(() => requests.stageAuthorizations.length).toBe(1)
    expect(requests.stageAuthorizations[0]).toEqual({ channel_id: 'claude_sub', group_id: 9 })
    const submit = page.getByRole('button', { name: 'Import credential', exact: true })
    await expect(submit).toBeEnabled()
    await submit.click()
    await expect.poll(() => connections.length).toBe(1)
    if (authState === 'mismatch') {
      await expect(
        page.getByText('This Group already has a credential.', { exact: false }),
      ).toBeVisible()
      expect(connections[0]?.body).toEqual({ staged_credential_id: 'stage_abc' })
      expect(deletes).toBe(0)
      await expect(page).toHaveURL(/\/import/)
      return
    }
    if (authState === 'unauthorized') {
      await page.waitForURL(/\/login\?.*redirect=/)
      const stored = await page.evaluate(() =>
        JSON.parse(sessionStorage.getItem('gpt-load.import-reauth-draft') ?? 'null'),
      )
      expect(stored.draft.mode).toBe('existing')
      expect(stored.draft.group_id).toBe(9)
      expect(stored.draft.staged_credential.stage_id).toBe('stage_abc')
      expect(stored.draft.staged_credential.authorization_method).toBe('browser_oauth')
      await page.getByRole('textbox', { name: 'Sign-in key', exact: true }).fill('e2e-auth-key')
      await page.getByRole('button', { name: 'Sign in', exact: true }).click()
      await page.waitForURL(/\/import/)
      await expect(page.getByText('t***@example.com', { exact: true })).toBeVisible()
      await expect(submit).toBeEnabled()
      await submit.click()
      await expect.poll(() => connections.length).toBe(2)
      expect(connections[1]?.body).toEqual({ staged_credential_id: 'stage_abc' })
      expect(requests.stageAuthorizations).toHaveLength(1)
      await expect
        .poll(() => page.evaluate(() => sessionStorage.getItem('gpt-load.import-reauth-draft')))
        .toBeNull()
      return
    }
    await page.getByRole('button', { name: 'Check result', exact: true }).click()
    await expect.poll(() => connections.length).toBe(2)
    expect(connections[0]?.body).toEqual({ staged_credential_id: 'stage_abc' })
    expect(connections[0]?.key).not.toBe('')
    expect(connections[1]?.key).toBe(connections[0]?.key)
    expect(connections[1]?.expectedID).toBe(connections[0]?.expectedID)
  })
}
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

for (const status of ['pending_authorization', 'expired', 'failed']) {
  test(`subscription ${status} can retry without connecting a second account`, async ({ page }) => {
    const requests = await mockImportApi(page)
    let authorizations = 0
    const stage = () => ({
      stage_id: authorizations === 1 ? 'stage_old' : 'stage_retried',
      status: authorizations === 1 ? status : 'ready',
      authorization_method: 'browser_oauth',
      authorization_url: 'https://auth.example.com/oauth',
      redirect_uri: 'http://127.0.0.1/oauth/callback',
      expires_at_ms: Date.now() + 600_000,
      account: {},
    })
    await page.route('**/api/credential-stages/**', async (route) => {
      if (route.request().url().endsWith('/authorizations')) authorizations += 1
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data: stage() }),
      })
    })
    await page.goto('/import', { waitUntil: 'commit' })
    await expect(page.locator('#channel-credentials')).toBeVisible({ timeout: FIRST_PAINT })
    await page.getByRole('radio', { name: 'Subscription account' }).click()
    await page.getByRole('button', { name: 'Sign in with Claude Subscription' }).click()
    const add = page.getByRole('button', { name: 'Connect another account', exact: true })
    await expect(add).toHaveCount(0)
    const retry = page.getByRole('button', { name: 'Start a new authorization', exact: true })
    await expect(retry).toBeEnabled()
    await retry.click()
    await expect.poll(() => authorizations).toBe(2)
    await expect(add).toHaveCount(0)
    await page.getByRole('button', { name: 'Create Group', exact: true }).click()
    await expect.poll(() => requests.createBodies.length).toBe(1)
    expect(requests.createBodies[0]!.staged_credential_id).toBe('stage_retried')
  })
}

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
    page.getByText(
      'A Group allows only one distinct credential. Keep one credential and remove the others.',
      { exact: true },
    ),
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

test('subscription channel stages an account through authorization polling', async ({ page }) => {
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

  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await page.getByRole('radio', { name: 'Subscription account' }).click()
  await page.getByRole('button', { name: 'Sign in with Claude Subscription' }).click()

  await expect.poll(() => requests.stageAuthorizations.length).toBe(1)
  expect(requests.stageAuthorizations[0]!.channel_id).toBe('claude_sub')
  await expect.poll(() => requests.stagePolls.length, { timeout: 10_000 }).toBeGreaterThan(0)
  await expect(page.getByText('t***@example.com')).toBeVisible({ timeout: 15_000 })
})
