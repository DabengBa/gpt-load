import { expect, test, type Page, type Route } from '@playwright/test'
import type { ModelPriceUpdateRequest } from '../src/shared/control/resources/model-prices'

// U003 unified-schedule price journey: the price drawer opens from
// `/schedule?selected_price_id`, keeps base/tier price editing, reset,
// mapping, and the shared-impact warning, no longer renders catalog
// specifications, and exposes the global catalog sync as a secondary action
// with success / failure / retry feedback.
//
// The mocks mirror `projectModelCollection`/`projectUpstreamModelDetail`
// invariants: `reference_count` equals the association count,
// `reference_group_count` equals the distinct group count, and each
// association group belongs to the price channel.

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
const stagingGroup = routeGroup(2, 'staging')

function basePrice(overrides: Record<string, unknown> = {}) {
  return {
    id: 7,
    channel_id: 'openai',
    channel_name: 'OpenAI',
    channel_mark: 'O',
    channel_icon: 'openai',
    model_id: 'gpt-4o-upstream',
    prices: { input: '0.0025', output: '0.01', cache_read: '0.00125', cache_write: null },
    mode_schedules: {
      fast: {
        prices: { input: '0.005', output: '0.02', cache_read: '0.0025', cache_write: null },
        context_tiers: [],
      },
    },
    pricing_status: 'configured',
    method: 'user_set',
    matched_provider_id: null,
    match_source: null,
    referenced: true,
    reference_count: 2,
    reference_group_count: 2,
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

function baseCollection() {
  const price = basePrice()
  return {
    summary: {
      client_model_count: 2,
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
            route_groups: [prodGroup, stagingGroup],
            affected_groups: [prodGroup, stagingGroup],
            catalog_reference: catalogReference('gpt-4o-upstream'),
          },
        ],
      },
      {
        client_model: 'gpt-4o-mini',
        protocols: allProtocols,
        upstream_models: [
          {
            model_id: 'gpt-4o-upstream',
            alias_applied: true,
            price,
            route_groups: [prodGroup, stagingGroup],
            affected_groups: [prodGroup, stagingGroup],
            catalog_reference: catalogReference('gpt-4o-upstream'),
          },
        ],
      },
    ],
    pagination: { page: 1, page_size: 10, total_items: 2, total_pages: 1 },
  }
}

function baseDetail() {
  return {
    model_id: 'gpt-4o-upstream',
    price: basePrice(),
    catalog_reference: catalogReference('gpt-4o-upstream'),
    associations: [
      { client_model: 'gpt-4o', alias_applied: true, group: prodGroup },
      { client_model: 'gpt-4o-mini', alias_applied: true, group: stagingGroup },
    ],
    client_model_count: 2,
    group_count: 2,
  }
}

interface PriceJourneyRequests {
  puts: Record<string, unknown>[]
  resets: number[]
  syncs: number
  syncOutcome: 'ok' | 'fail'
}

async function mockPriceJourney(page: Page): Promise<PriceJourneyRequests> {
  const requests: PriceJourneyRequests = {
    puts: [],
    resets: [],
    syncs: 0,
    syncOutcome: 'ok',
  }
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
      await fulfill({ authenticated: true, principal_type: 'admin' })
      return
    }
    if (path === '/api/models') {
      await fulfill(baseCollection())
      return
    }
    if (path === '/api/model-prices/sync') {
      requests.syncs += 1
      if (requests.syncOutcome === 'fail') {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ code: 1, message: 'sync failed', data: null }),
        })
        return
      }
      await fulfill({
        trigger: 'manual',
        checked_at_ms: 1730000000000,
        successful_fetch_at_ms: 1730000000000,
        not_modified: false,
        skipped: false,
      })
      return
    }
    if (path === '/api/model-prices/7') {
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON() as ModelPriceUpdateRequest
        requests.puts.push(body)
        price = basePrice({
          prices: {
            input: body.input,
            output: body.output,
            cache_read: body.cache_read,
            cache_write: body.cache_write,
          },
          context_tiers: body.context_tiers.map(({ threshold_tokens, ...prices }) => ({
            threshold_tokens,
            prices,
          })),
          mode_schedules: Object.fromEntries(
            Object.entries(body.mode_schedules).map(([mode, schedule]) => [
              mode,
              {
                ...schedule,
                context_tiers: schedule.context_tiers.map(({ threshold_tokens, ...prices }) => ({
                  threshold_tokens,
                  prices,
                })),
              },
            ]),
          ),
        })
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
    if (path === '/api/model-route/schedule' && route.request().method() === 'GET') {
      await fulfill({ items: [] })
      return
    }
    await fulfill({})
  })
  return requests
}

async function expectAstryxDocument(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({ timeout: 60_000 })
}

async function openPriceDrawer(page: Page) {
  const dialog = page.getByRole('dialog', { name: 'gpt-4o-upstream' })
  // Final contract: the schedule page owns the admin price drawer through
  // `selected_price_id`. Until U002 mounts ModelUpstreamDrawer this cannot
  // open yet — the red run records exactly that integration gap.
  await page.goto('/schedule?selected_price_id=7', { waitUntil: 'commit' })
  await expectAstryxDocument(page)
  await dialog.waitFor({ state: 'visible', timeout: 30_000 })
  return dialog
}

test.setTimeout(90_000)

test('390px price matrix shows base and tier field labels without desktop duplicates', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await mockPriceJourney(page)
  const dialog = await openPriceDrawer(page)
  const fields = [
    ['input', 'Input'],
    ['output', 'Output'],
    ['cache_read', 'Cache read'],
    ['cache_write', 'Cache write'],
  ] as const

  await dialog.getByRole('button', { name: 'Add tier' }).click()
  await dialog.getByRole('textbox', { name: 'Tier', exact: true }).fill('128000')
  for (const [field, label] of fields) {
    const baseLabel = dialog.locator(`label[for="model-price-base-${field}"]`).filter({
      hasText: label,
    })
    const tierLabel = dialog.locator(`label[for^="model-price-tier-"][for$="-${field}"]`).filter({
      hasText: label,
    })
    await expect(
      baseLabel.locator('span').filter({ hasText: new RegExp(`^${label}$`) }),
    ).toBeVisible()
    await expect(
      tierLabel.locator('span').filter({ hasText: new RegExp(`^${label}$`) }),
    ).toBeVisible()
  }
  await page.screenshot({ path: testInfo.outputPath('prices-390px.png'), fullPage: true })

  await page.setViewportSize({ width: 1280, height: 900 })
  for (const [field, label] of fields) {
    await expect(dialog.locator(`label[for="model-price-base-${field}"] span`)).toBeHidden()
    await expect(
      dialog.locator(`label[for^="model-price-tier-"][for$="-${field}"] span`),
    ).toBeHidden()
    await expect(
      dialog.locator('div[aria-hidden="true"]').getByText(label, { exact: true }),
    ).toBeVisible()
  }
  await page.screenshot({ path: testInfo.outputPath('prices-desktop.png'), fullPage: true })
})

test('price drawer drops specifications but keeps editing, mapping, and impact', async ({
  page,
}) => {
  const requests = await mockPriceJourney(page)
  const dialog = await openPriceDrawer(page)

  // Specifications are gone: neither the section heading nor the catalog
  // metadata that only the spec sheet rendered.
  await expect(dialog.getByRole('heading', { name: 'Specifications' })).toHaveCount(0)
  await expect(dialog.getByText('GPT-4o Upstream')).toHaveCount(0)
  await expect(dialog.getByText('No catalog metadata')).toHaveCount(0)

  // Identity, relationships, and the alias mapping stay.
  await expect(dialog.getByText('Pricing channel')).toBeVisible()
  await expect(dialog.getByText('Upstream model')).toBeVisible()
  await expect(dialog.getByRole('heading', { name: 'Relationships' })).toBeVisible()
  await expect(
    dialog.getByLabel('Client model gpt-4o maps to upstream model gpt-4o-upstream'),
  ).toBeVisible()

  // Shared impact comes from the full detail counts.
  await expect(
    dialog.getByText(
      /This pricing identity has 2 model references across 2 client models and 2 Groups/,
    ),
  ).toBeVisible()

  // Base price edit + save keeps working.
  const input = dialog.getByRole('textbox', { name: 'Input' }).first()
  await input.fill('0.005')
  const save = dialog.getByRole('button', { name: 'Save', exact: true })
  await expect(save).toBeEnabled()
  await save.click()
  await expect.poll(() => requests.puts.length).toBe(1)
  expect(requests.puts[0]?.input).toBe('0.005')

  // Tier editing stays available.
  await expect(dialog.getByRole('button', { name: 'Add tier' })).toBeVisible()

  // Reset stays available in the drawer footer.
  await dialog.getByRole('button', { name: 'Reset prices' }).click()
  const confirm = page.getByRole('dialog', { name: 'Reset prices?' })
  await expect(confirm).toBeVisible()
  await confirm.getByRole('button', { name: 'Reset prices' }).click()
  await expect.poll(() => requests.resets.length).toBe(1)
})

test('drawer exposes the global catalog sync as a secondary action', async ({ page }) => {
  const requests = await mockPriceJourney(page)
  const dialog = await openPriceDrawer(page)

  // The sync control lives inside the drawer and states its global scope.
  const sync = dialog.getByRole('button', { name: 'Sync catalog and automatic prices' })
  await expect(sync).toBeVisible()
  await expect(dialog.getByText(/Global action/)).toBeVisible()
  await sync.click()
  await expect.poll(() => requests.syncs).toBe(1)
  await expect(
    dialog
      .getByRole('status')
      .filter({ hasText: 'The model catalog and automatic prices are synced' }),
  ).toBeVisible()
})

test('drawer saves actual tier and Fast prices', async ({ page }) => {
  const requests = await mockPriceJourney(page)
  const dialog = await openPriceDrawer(page)
  await dialog.getByRole('button', { name: 'Add tier' }).click()
  await dialog.getByRole('textbox', { name: 'Tier', exact: true }).fill('128000')
  await dialog.getByRole('textbox', { name: 'Input', exact: true }).nth(1).fill('0.008')
  await dialog.getByRole('textbox', { name: 'Input', exact: true }).last().fill('0.009')
  await dialog.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(() => requests.puts.length).toBe(1)
  expect(requests.puts[0]?.context_tiers).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        threshold_tokens: 128000,
        input: '0.008',
      }),
    ]),
  )
  expect(requests.puts[0]?.mode_schedules).toEqual(
    expect.objectContaining({
      fast: expect.objectContaining({ prices: expect.objectContaining({ input: '0.009' }) }),
    }),
  )
  await page.reload()
  await expect(dialog.getByRole('textbox', { name: 'Input', exact: true }).last()).toHaveValue(
    '0.009',
  )
})

test('drawer protects unsaved prices when closing', async ({ page }) => {
  const requests = await mockPriceJourney(page)
  const dialog = await openPriceDrawer(page)
  await dialog.getByRole('textbox', { name: 'Input', exact: true }).first().fill('0.007')
  await dialog.getByRole('button', { name: 'Close', exact: true }).click()
  const confirm = page.getByRole('alertdialog', { name: 'Discard unsaved changes?' })
  await expect(confirm).toBeVisible()
  await confirm.getByRole('button', { name: 'Continue editing', exact: true }).click()
  await expect(dialog.getByRole('textbox', { name: 'Input', exact: true }).first()).toHaveValue(
    '0.007',
  )
  await dialog.getByRole('button', { name: 'Close', exact: true }).click()
  await confirm.getByRole('button', { name: 'Discard changes', exact: true }).click()
  await expect(page.getByRole('dialog', { name: 'gpt-4o-upstream' })).not.toBeVisible()
  expect(requests.puts).toHaveLength(0)
})

test('drawer sync failure surfaces an alert with retry', async ({ page }) => {
  const requests = await mockPriceJourney(page)
  requests.syncOutcome = 'fail'
  const dialog = await openPriceDrawer(page)

  await dialog.getByRole('button', { name: 'Sync catalog and automatic prices' }).click()
  await expect.poll(() => requests.syncs).toBe(1)
  const alert = dialog
    .getByRole('alert')
    .filter({ hasText: 'Unable to sync the model catalog and prices' })
  await expect(alert).toBeVisible()
  await expect(alert.getByRole('button', { name: 'Retry' })).toBeVisible()

  await dialog.getByRole('textbox', { name: 'Input' }).first().fill('0.005')
  await expect(alert.getByRole('button', { name: 'Retry' })).toBeDisabled()
  await expect(
    dialog.getByRole('button', { name: 'Sync catalog and automatic prices' }),
  ).toBeDisabled()
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect(alert.getByRole('button', { name: 'Retry' })).toBeEnabled()

  requests.syncOutcome = 'ok'
  await alert.getByRole('button', { name: 'Retry' }).click()
  await expect.poll(() => requests.syncs).toBe(2)
  await expect(
    dialog
      .getByRole('status')
      .filter({ hasText: 'The model catalog and automatic prices are synced' }),
  ).toBeVisible()
  await expect(alert).toHaveCount(0)
})
