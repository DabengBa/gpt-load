import { expect, test, type Page, type Route } from '@playwright/test'

// Phase 2 models-domain coverage for the Astryx entry: collection filters with
// canonical route query, the upstream detail drawer, embedded price editing
// (save/unpriced/reset), unsaved-change interception, and the access-key
// read-only mode.
//
// The mock mirrors `projectModelCollection`/`projectUpstreamModelDetail`
// invariants: configured+user_set prices carry no matched provider, referenced
// prices must keep reference counts consistent with route/affected groups, and
// detail associations must line up with client/group counts.

const allProtocols = [
  'openai-completions',
  'openai-responses',
  'openai-images',
  'openai-embeddings',
  'rerank',
  'anthropic',
  'gemini',
]

function routeGroup(id: number, name: string, enabled = true) {
  return {
    id,
    name,
    channel_id: 'openai',
    params: {},
    enabled,
    client_protocols: allProtocols,
  }
}

const prodGroup = routeGroup(1, 'prod')

function basePrice(overrides: Record<string, unknown> = {}) {
  return {
    id: 7,
    channel_id: 'openai',
    channel_name: 'OpenAI',
    channel_mark: 'O',
    channel_icon: 'openai',
    model_id: 'gpt-4o-upstream',
    prices: { input: '0.0025', output: '0.01', cache_read: '0.00125', cache_write: null },
    mode_schedules: {},
    pricing_status: 'configured',
    method: 'user_set',
    matched_provider_id: null,
    match_source: null,
    referenced: true,
    reference_count: 1,
    reference_group_count: 1,
    context_tiers: [],
    updated_at_ms: 1730000000000,
    can_reset: true,
    can_delete: false,
    ...overrides,
  }
}

function catalogReference(modelId: string) {
  return {
    source: 'actual_provider',
    provider_id: 'openai',
    provider_name: 'OpenAI',
    model: {
      id: modelId,
      name: 'GPT-4o Upstream',
      description: '',
      family: 'gpt-4o',
      modalities: { input: ['text'], output: ['text'] },
      limits: { context: 128000, input: null, output: 16384 },
      capabilities: {
        attachment: true,
        reasoning: false,
        tool_call: true,
        structured_output: true,
        temperature: true,
      },
      release_date: '',
      last_updated: '',
      knowledge: '',
      open_weights: false,
      status: '',
    },
  }
}

function baseCollection(isAccessKey = false) {
  const price = basePrice()
  return {
    summary: {
      client_model_count: 1,
      upstream_model_count: 1,
      price_count: 1,
      pending_price_count: 0,
      unreferenced_price_count: 0,
    },
    catalog: {
      available: true,
      checked_at_ms: 1730000000000,
      successful_fetch_at_ms: 1730000000000,
      error_code: '',
    },
    items: [
      {
        client_model: 'gpt-4o',
        protocols: allProtocols,
        upstream_models: [
          {
            model_id: 'gpt-4o-upstream',
            alias_applied: true,
            price,
            route_groups: isAccessKey ? [] : [prodGroup],
            affected_groups: isAccessKey ? [] : [prodGroup],
            catalog_reference: catalogReference('gpt-4o-upstream'),
          },
        ],
      },
    ],
    pagination: { page: 1, page_size: 10, total_items: 1, total_pages: 1 },
  }
}

function baseDetail() {
  return {
    model_id: 'gpt-4o-upstream',
    price: basePrice(),
    catalog_reference: catalogReference('gpt-4o-upstream'),
    associations: [
      { client_model: 'gpt-4o', alias_applied: true, group: prodGroup },
    ],
    client_model_count: 1,
    group_count: 1,
  }
}

interface ModelsRequests {
  collectionQueries: URLSearchParams[]
  puts: Record<string, unknown>[]
  resets: number[]
}

async function mockModels(
  page: Page,
  options: { principalType?: 'admin' | 'access_key' } = {},
): Promise<ModelsRequests> {
  const principalType = options.principalType ?? 'admin'
  const requests: ModelsRequests = { collectionQueries: [], puts: [], resets: [] }
  let price = basePrice()
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route: Route) => {
    const url = new URL(route.request().url())
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
    if (path === '/api/models') {
      requests.collectionQueries.push(url.searchParams)
      await fulfill(baseCollection(principalType === 'access_key'))
      return
    }
    if (path === '/api/model-prices/sync') {
      await fulfill({
        trigger: 'manual',
        checked_at_ms: 1730000000000,
        successful_fetch_at_ms: 1730000000000,
        not_modified: false,
        skipped: false,
        error_code: null,
      })
      return
    }
    if (path === '/api/model-prices/7') {
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON() as Record<string, unknown>
        requests.puts.push(body)
        const prices = body as { input?: string | null }
        price = basePrice(prices.input !== undefined ? { prices: { ...basePrice().prices, input: prices.input } } : {})
        await fulfill(price)
        return
      }
      await fulfill({ ...baseDetail(), price })
      return
    }
    if (path === '/api/model-prices/7/reset') {
      requests.resets.push(7)
      await fulfill(price)
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

async function openDrawer(page: Page) {
  await page
    .getByRole('button', {
      name: 'View details and pricing for upstream model gpt-4o-upstream',
    })
    .first()
    .click()
  const dialog = page.getByRole('dialog', { name: 'gpt-4o-upstream' })
  await expect(dialog).toBeVisible()
  return dialog
}

test('renders the collection and canonicalizes invalid route query params', async ({
  page,
}) => {
  await mockModels(page)
  await page.goto('/models?group_status=junk&pricing_status=nope&page=0&selected_price_id=abc', {
    waitUntil: 'load',
  })
  await expectAstryxDocument(page)
  await expect(page).toHaveURL(/\/models$/)

  await expect(
    page.getByRole('heading', { name: 'Models', exact: true }),
  ).toBeVisible()
  const tree = page.getByRole('table', { name: 'Client models and upstream prices' })
  await expect(tree).toBeVisible()
  await expect(tree.getByText('gpt-4o', { exact: true })).toBeVisible()
  await expect(tree.getByText('gpt-4o-upstream', { exact: true }).first()).toBeVisible()
  await expect(tree.getByText('prod')).toBeVisible()
})

test('applies search and status filters through the route query', async ({
  page,
}) => {
  const requests = await mockModels(page)
  await page.goto('/models', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  await expect(
    page.getByRole('table', { name: 'Client models and upstream prices' }),
  ).toBeVisible()

  await page
    .getByRole('textbox', { name: 'Search' })
    .fill('claude')
  await expect(page).toHaveURL(/[?&]q=claude/)

  await page.getByRole('combobox', { name: 'Pricing status' }).click()
  await page.getByRole('option', { name: 'Pending' }).click()
  await expect(page).toHaveURL(/[?&]pricing_status=pending/)

  await page.getByRole('combobox', { name: 'Group status' }).click()
  await page.getByRole('option', { name: 'All Groups' }).click()
  await expect(page).toHaveURL(/[?&]group_status=all/)

  const last = requests.collectionQueries.at(-1)
  expect(last?.get('q')).toBe('claude')
  expect(last?.get('pricing_status')).toBe('pending')
  expect(last?.get('group_status')).toBe('all')
})

test('opens the upstream drawer and saves a price edit', async ({ page }) => {
  const requests = await mockModels(page)
  await page.goto('/models', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  const dialog = await openDrawer(page)
  await expect(page).toHaveURL(/[?&]selected_price_id=7/)
  await expect(dialog.getByText('Pricing channel')).toBeVisible()
  await expect(dialog.getByText('Specifications')).toBeVisible()
  await expect(dialog.getByText('GPT-4o Upstream')).toBeVisible()
  await expect(dialog.getByText('Relationships')).toBeVisible()

  const input = dialog.getByRole('textbox', { name: 'Input' }).first()
  await input.fill('0.005')
  const save = dialog.getByRole('button', { name: 'Save', exact: true })
  await expect(save).toBeEnabled()
  await save.click()

  await expect.poll(() => requests.puts.length).toBe(1)
  expect(requests.puts[0]?.input).toBe('0.005')
  expect(requests.puts[0]?.confirm_unpriced).toBe(false)

  // The mutation invalidation refetches the collection.
  await expect.poll(() => requests.collectionQueries.length).toBeGreaterThan(1)
})

test('asks for confirmation before marking a price unpriced', async ({ page }) => {
  const requests = await mockModels(page)
  await page.goto('/models', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  const dialog = await openDrawer(page)
  for (const name of ['Input', 'Output', 'Cache read']) {
    await dialog.getByRole('textbox', { name }).first().fill('')
  }
  await dialog.getByRole('button', { name: 'Save', exact: true }).click()

  const confirm = page.getByRole('dialog', { name: 'Mark this model unpriced?' })
  await expect(confirm).toBeVisible()
  await confirm.getByRole('button', { name: 'Confirm unpriced' }).click()

  await expect.poll(() => requests.puts.length).toBe(1)
  expect(requests.puts[0]?.confirm_unpriced).toBe(true)
})

test('blocks route navigation while the drawer has unsaved price edits', async ({
  page,
}) => {
  await mockModels(page)
  await page.goto('/models', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  const dialog = await openDrawer(page)
  await expect(page).toHaveURL(/[?&]selected_price_id=7/)
  await dialog.getByRole('textbox', { name: 'Input' }).first().fill('0.009')

  // Back navigation removes `selected_price_id`; the unsaved guard must
  // intercept it. The modal drawer also covers the shell nav, so history
  // traversal is the reachable navigation path — same as classic.
  await page.goBack()

  const blocker = page.getByRole('alertdialog')
  await expect(blocker).toBeVisible()
  await blocker.getByRole('button', { name: 'Continue editing' }).click()
  await expect(dialog).toBeVisible()
  await expect(page).toHaveURL(/[?&]selected_price_id=7/)

  await page.goBack()
  const discard = page.getByRole('alertdialog')
  await expect(discard).toBeVisible()
  await discard.getByRole('button', { name: 'Discard changes' }).click()
  await expect(page).not.toHaveURL(/selected_price_id=/)
})

test('resets a configured price through the drawer', async ({ page }) => {
  const requests = await mockModels(page)
  await page.goto('/models', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  const dialog = await openDrawer(page)
  await dialog.getByRole('button', { name: 'Reset prices' }).click()
  const confirm = page.getByRole('dialog', { name: 'Reset prices?' })
  await expect(confirm).toBeVisible()
  await confirm.getByRole('button', { name: 'Reset prices' }).click()

  await expect.poll(() => requests.resets.length).toBe(1)
})

test('access-key sessions get a read-only collection', async ({ page }) => {
  await mockModels(page, { principalType: 'access_key' })
  await page.goto('/models', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  const tree = page.getByRole('table', { name: 'Client models and upstream prices' })
  await expect(tree).toBeVisible()
  await expect(tree.getByText('gpt-4o-upstream', { exact: true }).first()).toBeVisible()

  await expect(
    page.getByRole('button', { name: 'Sync catalog and automatic prices' }),
  ).toHaveCount(0)
  await expect(
    page.getByRole('button', {
      name: 'View details and pricing for upstream model gpt-4o-upstream',
    }),
  ).toHaveCount(0)
})
