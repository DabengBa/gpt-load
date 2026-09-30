import { expect, test, type Page, type Route } from '@playwright/test'

// Phase 3 access-keys-domain coverage for the Astryx entry: collection with
// status summary + canonical route query, the create/edit drawer (unsaved
// guard), row actions (status toggle, delete with typed confirmation,
// cost-limit reset), the rotate flow, and reveal-on-copy.
//
// The mock mirrors `projectAccessKeyCollection` invariants: summary totals
// must equal active+disabled, pagination must match total_pages derivation,
// and no collection field may look secret-like (masked_key only).

const now = 1730000000000

function accessKey(
  id: number,
  name: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    id,
    name,
    masked_key: `sk-00000000****${String(id).padStart(4, '0')}`,
    status: 'active',
    filters: { groups: [], protocols: [], models: [], allowed_cidrs: [] },
    expires_at_ms: null,
    rpm_limit: 0,
    price_multiplier: '1',
    cost_limit_rules: [],
    created_at_ms: now - 86_400_000,
    updated_at_ms: now - 3_600_000,
    expired: false,
    last_request_at_ms: now - 60_000,
    ...overrides,
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

// Mutation responses project only AccessKeyMetadata — collection-item fields
// (`expired`, `last_request_at_ms`) are rejected by the strict projector, so
// the mock strips them before fulfilling POST/PUT/rotate.
function metadataOnly(item: Record<string, unknown>): Record<string, unknown> {
  const rest = { ...item }
  delete rest.expired
  delete rest.last_request_at_ms
  return rest
}

interface AccessKeysRequests {
  collectionQueries: URLSearchParams[]
  posts: Record<string, unknown>[]
  puts: { id: number; body: Record<string, unknown> }[]
  deletes: number[]
  reveals: number[]
  rotates: number[]
  resets: { id: number; body: Record<string, unknown> }[]
}

async function mockAccessKeys(
  page: Page,
  options: { principalType?: 'admin' | 'access_key'; empty?: boolean } = {},
): Promise<AccessKeysRequests> {
  const principalType = options.principalType ?? 'admin'
  const requests: AccessKeysRequests = {
    collectionQueries: [],
    posts: [],
    puts: [],
    deletes: [],
    reveals: [],
    rotates: [],
    resets: [],
  }
  let items: Record<string, unknown>[] = options.empty
    ? []
    : [
        accessKey(1, 'prod key'),
        accessKey(2, 'dev key', {
          status: 'disabled',
          rpm_limit: 600,
          cost_limit_rules: [
            { id: 11, kind: 'total', limit_usd: '10.0', period_seconds: 0 },
            { id: 12, kind: 'periodic', limit_usd: '1.5', period_seconds: 86_400 },
          ],
        }),
      ]
  let nextID = 3

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
    if (path === '/api/access-keys' && request.method() === 'GET') {
      requests.collectionQueries.push(url.searchParams)
      const page_ = Number(url.searchParams.get('page') ?? '1')
      const totalItems = items.length
      const totalPages = totalItems === 0 ? 0 : Math.ceil(totalItems / 20)
      await fulfill({
        summary: {
          total: totalItems,
          active: items.filter((item) => item.status === 'active').length,
          disabled: items.filter((item) => item.status === 'disabled').length,
        },
        items: items.slice((page_ - 1) * 20, page_ * 20),
        pagination: {
          page: page_,
          page_size: 20,
          total_items: totalItems,
          total_pages: totalPages,
        },
      })
      return
    }
    if (path === '/api/access-keys' && request.method() === 'POST') {
      const body = request.postDataJSON() as Record<string, unknown>
      requests.posts.push(body)
      const created = accessKey(nextID++, String(body.name ?? 'unnamed'), {
        filters: body.filters ?? { groups: [], protocols: [], models: [], allowed_cidrs: [] },
        rpm_limit: body.rpm_limit ?? 0,
      })
      items = [created, ...items]
      await fulfill({ ...metadataOnly(created), key: `sk-e2e-new-${created.id}`, replayed: false })
      return
    }
    const itemMatch = /^\/api\/access-keys\/(\d+)(?:\/(reveal|rotate|cost-limits\/reset))?$/.exec(
      path,
    )
    if (itemMatch) {
      const id = Number(itemMatch[1])
      const action = itemMatch[2]
      const target = items.find((item) => item.id === id)
      if (action === 'reveal') {
        requests.reveals.push(id)
        await fulfill({ id, key: `sk-e2e-revealed-${id}`, revealed_at_ms: now })
        return
      }
      if (action === 'rotate' && request.method() === 'POST') {
        requests.rotates.push(id)
        if (target) Object.assign(target, { updated_at_ms: now })
        await fulfill({
          ...metadataOnly(target),
          key: `sk-e2e-rotated-${id}`,
          replayed: false,
          updated_at_ms: now,
        })
        return
      }
      if (action === 'cost-limits/reset' && request.method() === 'POST') {
        requests.resets.push({ id, body: request.postDataJSON() as Record<string, unknown> })
        await fulfill({ reset: true })
        return
      }
      if (request.method() === 'PUT' && target) {
        const body = request.postDataJSON() as Record<string, unknown>
        requests.puts.push({ id, body })
        Object.assign(target, body, { updated_at_ms: now })
        await fulfill(metadataOnly(target))
        return
      }
      if (request.method() === 'DELETE' && target) {
        requests.deletes.push(id)
        items = items.filter((item) => item.id !== id)
        await fulfill({ deleted: true })
        return
      }
      await fulfill(target ? metadataOnly(target) : {}, target ? 200 : 404)
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
    await fulfill({})
  })
  return requests
}

test.setTimeout(90_000)

async function expectAstryxDocument(page: Page): Promise<void> {
  // The first astryx navigation in a run eats the cold vite transform of the
  // whole module graph; give the shell assertion room while `goto` uses
  // `commit` so the response itself never blocks on it.
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({
    timeout: 60_000,
  })
  await expect(page.locator('[data-testid="desktop-nav"]')).toBeVisible()
}

async function openEditDrawer(page: Page, name: string) {
  await page.getByRole('button', { name: `Open details for access key “${name}”` }).click()
  const dialog = page.getByRole('dialog', { name: 'Edit access key' })
  await expect(dialog).toBeVisible()
  return dialog
}

test('renders the collection and canonicalizes invalid route query params', async ({ page }) => {
  await mockAccessKeys(page)
  await page.goto('/access-keys?status=junk&page=0&action=bogus', { waitUntil: 'commit' })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/access-keys$/)

  await expect(page.getByRole('heading', { name: 'Access keys', exact: true })).toBeVisible()
  const table = page.getByRole('table', { name: 'Access key list' })
  await expect(table).toBeVisible()
  await expect(table.getByText('prod key', { exact: true }).first()).toBeVisible()
  await expect(table.getByText('dev key', { exact: true }).first()).toBeVisible()
  // Masked only — no collection DTO ever carries plaintext.
  await expect(table.getByText(/sk-00000000\*\*\*\*0001/).first()).toBeVisible()
})

test('applies search and status filters through the route query', async ({ page }) => {
  const requests = await mockAccessKeys(page)
  await page.goto('/access-keys', { waitUntil: 'commit' })
  await expectAstryxDocument(page)
  await expect(page.getByRole('table', { name: 'Access key list' })).toBeVisible()

  await page.getByRole('textbox', { name: 'Search' }).fill('prod')
  await expect(page).toHaveURL(/[?&]q=prod/)

  await page.getByRole('combobox', { name: 'Access key status overview' }).click()
  await page.getByRole('option', { name: /Disabled/ }).click()
  await expect(page).toHaveURL(/[?&]status=disabled/)

  const last = requests.collectionQueries.at(-1)
  expect(last?.get('q')).toBe('prod')
  expect(last?.get('status')).toBe('disabled')
})

test('creates an access key through the drawer', async ({ page }) => {
  const requests = await mockAccessKeys(page)
  await page.goto('/access-keys', { waitUntil: 'commit' })
  await expectAstryxDocument(page)

  await page.getByRole('button', { name: 'Create access key' }).first().click()
  const drawer = page.getByRole('dialog', { name: 'Create access key' })
  await expect(drawer).toBeVisible()
  await expect(page).toHaveURL(/[?&]action=create/)

  await drawer.getByRole('textbox', { name: 'Name' }).fill('e2e created')
  await drawer.getByRole('button', { name: 'Create key' }).click()

  await expect.poll(() => requests.posts.length).toBe(1)
  expect(requests.posts[0]?.name).toBe('e2e created')
  // Drawer closes and the collection refetches.
  await expect(drawer).not.toBeVisible()
  await expect.poll(() => requests.collectionQueries.length).toBeGreaterThan(1)
})

test('blocks route navigation while the drawer has unsaved edits', async ({ page }) => {
  await mockAccessKeys(page)
  await page.goto('/access-keys', { waitUntil: 'commit' })
  await expectAstryxDocument(page)

  const drawer = await openEditDrawer(page, 'prod key')
  await expect(page).toHaveURL(/[?&]action=edit&access_key_id=1/)
  await drawer.getByRole('textbox', { name: 'Requests per minute' }).fill('30')

  await page.goBack()
  const blocker = page.getByRole('alertdialog')
  await expect(blocker).toBeVisible()
  await blocker.getByRole('button', { name: 'Continue editing' }).click()
  await expect(drawer).toBeVisible()
  await expect(page).toHaveURL(/access_key_id=1/)

  await page.goBack()
  const discard = page.getByRole('alertdialog')
  await expect(discard).toBeVisible()
  await discard.getByRole('button', { name: 'Discard changes' }).click()
  await expect(page).not.toHaveURL(/access_key_id=/)
})

test('edits an access key and toggles its status', async ({ page }) => {
  const requests = await mockAccessKeys(page)
  await page.goto('/access-keys', { waitUntil: 'commit' })
  await expectAstryxDocument(page)

  const drawer = await openEditDrawer(page, 'dev key')
  await drawer.getByRole('textbox', { name: 'Requests per minute' }).fill('1200')
  await drawer.getByRole('button', { name: 'Save changes' }).click()

  await expect.poll(() => requests.puts.length).toBe(1)
  expect(requests.puts[0]?.id).toBe(2)
  expect(requests.puts[0]?.body?.rpm_limit).toBe(1200)
  await expect(drawer).not.toBeVisible()

  const table = page.getByRole('table', { name: 'Access key list' })
  await table.getByRole('button', { name: 'Enable' }).first().click()
  await expect.poll(() => requests.puts.length).toBe(2)
  expect(requests.puts[1]?.body?.status).toBe('active')
})

test('deletes an access key with typed confirmation', async ({ page }) => {
  const requests = await mockAccessKeys(page)
  await page.goto('/access-keys', { waitUntil: 'commit' })
  await expectAstryxDocument(page)

  const table = page.getByRole('table', { name: 'Access key list' })
  const prodRow = table.getByRole('row').filter({ hasText: 'prod key' })
  await prodRow.getByRole('button', { name: 'Delete' }).click()

  const dialog = page.getByRole('dialog', { name: 'Delete this access key?' })
  await expect(dialog).toBeVisible()
  const confirm = dialog.getByRole('button', { name: 'Delete access key' })
  await expect(confirm).toBeDisabled()

  await dialog.getByRole('textbox', { name: /Type prod key to confirm/ }).fill('prod key')
  await expect(confirm).toBeEnabled()
  await confirm.click()

  await expect.poll(() => requests.deletes).toEqual([1])
  await expect(dialog).not.toBeVisible()
  await expect(table.getByRole('cell', { name: 'prod key', exact: true })).toHaveCount(0)
})

test('resets cost-limit rules from the row action', async ({ page }) => {
  const requests = await mockAccessKeys(page)
  await page.goto('/access-keys', { waitUntil: 'commit' })
  await expectAstryxDocument(page)

  const table = page.getByRole('table', { name: 'Access key list' })
  const devRow = table.getByRole('row').filter({ hasText: 'dev key' })
  await devRow.getByRole('button', { name: 'Reset allowances' }).click()

  const dialog = page.getByRole('dialog', { name: 'Reset cost allowances?' })
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: 'Reset 2 rules' }).click()

  await expect.poll(() => requests.resets.length).toBe(1)
  expect(requests.resets[0]?.id).toBe(2)
})

test('rotates an access key inside the edit drawer', async ({ page }) => {
  const requests = await mockAccessKeys(page)
  await page.goto('/access-keys', { waitUntil: 'commit' })
  await expectAstryxDocument(page)

  const drawer = await openEditDrawer(page, 'prod key')
  await drawer.getByRole('button', { name: 'Rotate key' }).click()

  const dialog = page.getByRole('dialog', { name: 'Rotate this access key now?' })
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: 'Rotate now' }).click()

  await expect.poll(() => requests.rotates).toEqual([1])
  await expect(dialog.getByText('New access key', { exact: true })).toBeVisible()
  await expect(dialog.getByText('sk-e2e-rotated-1')).toBeVisible()
})

test('reveals the access key on copy only', async ({ page }) => {
  const requests = await mockAccessKeys(page)
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto('/access-keys', { waitUntil: 'commit' })
  await expectAstryxDocument(page)

  const table = page.getByRole('table', { name: 'Access key list' })
  await expect(table.getByText(/sk-00000000\*\*\*\*0001/).first()).toBeVisible()
  expect(requests.reveals).toHaveLength(0)

  await table.getByRole('button', { name: 'Copy access key' }).first().click()
  await expect.poll(() => requests.reveals).toEqual([1])
  await expect
    .poll(async () => page.evaluate(() => navigator.clipboard.readText()))
    .toBe('sk-e2e-revealed-1')
})

test('access_key principals are redirected away from the admin-only page', async ({ page }) => {
  await mockAccessKeys(page, { principalType: 'access_key' })
  await page.goto('/access-keys', { waitUntil: 'commit' })
  // Boot → session validation → AuthGate redirect outruns the default expect
  // window under cold module transforms; give the URL assertion headroom.
  await expect(page).not.toHaveURL(/\/access-keys/, { timeout: 60_000 })
})
