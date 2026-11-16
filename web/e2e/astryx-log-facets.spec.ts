import { expect, test } from '@playwright/test'

import {
  installRequestLogDisplayRoutes,
  installRequestLogTableRoutes,
} from './fixtures/request-log-display'

test('only route-valid historical model facets can be selected', async ({ page }) => {
  const { logRequests } = await installRequestLogTableRoutes(page)
  const maxBytes = `${'é'.repeat(127)}a`
  await page.route('**/api/logs/facets?*', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'OK',
        data: {
          groups: [{ id: 99, name: 'historical-alpha' }],
          client_models: [
            'broken\nmodel',
            'broken\u0085model',
            ' leading',
            'trailing ',
            ' ',
            'x'.repeat(256),
            'é'.repeat(128),
            'valid model',
            maxBytes,
          ].map((value) => ({ value })),
        },
      },
    }),
  )
  await page.goto('/logs?group_id=99&status=error', { waitUntil: 'commit' })
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4, { timeout: 60_000 })
  await page.getByRole('button', { name: 'Client model' }).click()
  await expect(page.getByRole('option', { name: /broken/ })).toHaveCount(0)
  await expect(page.getByRole('option')).toHaveCount(3)
  await expect(page.getByRole('option', { name: maxBytes, exact: true })).toBeVisible()
  await page.getByRole('option', { name: 'valid model', exact: true }).click()
  await expect.poll(() => logRequests.at(-1)?.searchParams.get('client_model')).toBe('valid model')
  await expect(page).toHaveURL(/group_id=99/)
  await expect(page).toHaveURL(/status=error/)
  expect(logRequests.at(-1)?.searchParams.get('group_id')).toBe('99')
  expect(logRequests.at(-1)?.searchParams.get('status')).toBe('error')
  await page.getByRole('button', { name: 'Client model' }).click()
  await page.getByRole('option', { name: maxBytes, exact: true }).click()
  await expect.poll(() => logRequests.at(-1)?.searchParams.get('client_model')).toBe(maxBytes)
  expect(logRequests.at(-1)?.searchParams.get('group_id')).toBe('99')
  expect(logRequests.at(-1)?.searchParams.get('status')).toBe('error')
})

test('historical facets apply immediately without losing other drafts across URL updates', async ({
  page,
}) => {
  const { logRequests } = await installRequestLogTableRoutes(page)
  const facetRequests: URL[] = []
  await page.route('**/api/logs/facets?*', async (route) => {
    facetRequests.push(new URL(route.request().url()))
    await route.fulfill({
      json: {
        code: 0,
        message: 'OK',
        data: {
          groups: [{ id: 99, name: 'historical-alpha' }],
          client_models: [{ value: 'historic-model' }],
        },
      },
    })
  })
  await page.goto('/logs?limit=20', { waitUntil: 'commit' })
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4, { timeout: 60_000 })
  await page.getByRole('button', { name: 'Next' }).click()
  await expect(page).toHaveURL(/log_cursors=/)
  await page.getByRole('combobox', { name: 'Status', exact: true }).click()
  await page.getByRole('option', { name: 'Error', exact: true }).click()
  await page.getByRole('button', { name: 'Any-attempt Group' }).click()
  await page.getByRole('combobox', { name: 'Search options' }).fill('historical')
  await page.getByRole('option', { name: 'historical-alpha' }).click()
  await expect(page).toHaveURL(/group_id=99/)
  await expect(page).not.toHaveURL(/log_cursors=/)
  await expect(page).not.toHaveURL(/status=/)
  await expect(page.getByRole('combobox', { name: 'Status', exact: true })).toContainText('Error')
  await expect.poll(() => logRequests.at(-1)?.searchParams.get('group_id')).toBe('99')
  await expect.poll(() => facetRequests.at(-1)?.searchParams.get('group_id')).toBe('99')
  expect(facetRequests.at(-1)?.searchParams.has('status')).toBe(false)
  expect(facetRequests.at(-1)?.searchParams.has('from_ms')).toBe(true)
  expect(facetRequests.at(-1)?.searchParams.has('to_ms')).toBe(true)
  expect(facetRequests.at(-1)?.searchParams.has('cursor')).toBe(false)
  await page.getByRole('button', { name: 'Client model' }).click()
  await page.getByRole('option', { name: 'historic-model' }).click()
  await expect(page).toHaveURL(/client_model=historic-model/)
  await expect(page).not.toHaveURL(/status=/)
  await expect(page.getByRole('combobox', { name: 'Status', exact: true })).toContainText('Error')
  await page.getByRole('button', { name: 'Client model' }).click()
  await page.getByRole('option', { name: 'Any client model' }).click()
  await expect(page).not.toHaveURL(/client_model=/)
  await expect(page.getByRole('combobox', { name: 'Status', exact: true })).toContainText('Error')
  await page.getByRole('button', { name: 'Any-attempt Group' }).click()
  await page.getByRole('option', { name: 'All groups (any attempt)' }).click()
  await expect(page).not.toHaveURL(/group_id=/)
  await expect(page.getByRole('combobox', { name: 'Status', exact: true })).toContainText('Error')
})

test('selected historical values remain available when facets are empty', async ({ page }) => {
  await installRequestLogDisplayRoutes(page)
  await page.route('**/api/logs/facets?*', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'OK',
        data: { groups: [], client_models: [] },
      },
    }),
  )
  await page.goto('/logs?group_id=99&client_model=gone-model', { waitUntil: 'commit' })
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4, { timeout: 60_000 })
  await expect(page.getByRole('button', { name: 'Any-attempt Group' })).toContainText('#99')
  await expect(page.getByRole('button', { name: 'Client model' })).toContainText('gone-model')
  await page.getByRole('button', { name: 'Any-attempt Group' }).click()
  await expect(page.getByRole('option', { name: '#99' })).toBeVisible()
})

test('facet errors are visible without using configuration as history', async ({ page }) => {
  await installRequestLogDisplayRoutes(page)
  await page.route('**/api/logs/facets?*', (route) =>
    route.fulfill({
      status: 500,
      json: {
        code: 1,
        message: 'failure',
        data: null,
      },
    }),
  )
  await page.goto('/logs', { waitUntil: 'commit' })
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4, { timeout: 60_000 })
  await expect(page.getByRole('button', { name: 'Any-attempt Group' })).toBeDisabled({
    timeout: 15_000,
  })
  await expect(page.getByRole('button', { name: 'Client model' })).toBeDisabled()
  await expect(page.getByText(/some filter options/i)).toBeVisible()
})

test('access key sessions never request admin facets', async ({ page }) => {
  await installRequestLogDisplayRoutes(page, undefined, null, 'access_key')
  const facetRequests: string[] = []
  page.on('request', (request) => {
    if (new URL(request.url()).pathname === '/api/logs/facets') facetRequests.push(request.url())
  })
  await page.goto('/logs?from_ms=1700000000000&to_ms=1700003600000', { waitUntil: 'commit' })
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4, { timeout: 60_000 })
  await expect(page.getByRole('button', { name: 'Any-attempt Group' })).toHaveCount(0)
  await expect(page.getByRole('textbox', { name: 'Client model' })).toBeVisible()
  await Promise.all([
    page.waitForResponse((response) => new URL(response.url()).pathname === '/api/logs'),
    page.getByRole('button', { name: 'Apply', exact: true }).first().click(),
  ])
  expect(facetRequests).toEqual([])
})

for (const facet of [
  { query: 'group_id=99', button: 'Any-attempt Group', clear: 'All groups (any attempt)' },
  { query: 'client_model=old-model', button: 'Client model', clear: 'Any client model' },
]) {
  test(`invalid affinity key blocks clearing selected ${facet.button}`, async ({ page }) => {
    const { logRequests } = await installRequestLogTableRoutes(page)
    const facetRequests: string[] = []
    page.on('request', (request) => {
      if (new URL(request.url()).pathname === '/api/logs/facets') facetRequests.push(request.url())
    })
    await page.goto(`/logs?affinity_key=not-a-canonical-key&${facet.query}`, {
      waitUntil: 'commit',
    })
    await expect(page.locator('#logs-filter-error')).toContainText('canonical 16', {
      timeout: 60_000,
    })
    await page.getByRole('button', { name: facet.button }).click()
    await page.getByRole('option', { name: facet.clear }).click()
    await page.waitForTimeout(300)
    await expect(page).toHaveURL(/affinity_key=not-a-canonical-key/)
    await expect(page).toHaveURL(new RegExp(facet.query))
    await page.getByRole('button', { name: 'Apply', exact: true }).first().click()
    expect(logRequests).toHaveLength(0)
    expect(facetRequests).toHaveLength(0)
  })
}

test('refreshing the list also refreshes historical facets under the same filters', async ({
  page,
}) => {
  const { logRequests } = await installRequestLogTableRoutes(page)
  const facetRequests: URL[] = []
  await page.route('**/api/logs/facets?*', async (route) => {
    facetRequests.push(new URL(route.request().url()))
    await route.fulfill({
      json: {
        code: 0,
        message: 'OK',
        data: {
          groups: [
            {
              id: facetRequests.length === 1 ? 1 : 2,
              name: facetRequests.length === 1 ? 'old-history' : 'new-history',
            },
          ],
          client_models: [],
        },
      },
    })
  })
  await page.goto('/logs?from_ms=1700000000000&to_ms=1700003600000&status=success&limit=20', {
    waitUntil: 'commit',
  })
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4, { timeout: 60_000 })
  await page.getByRole('button', { name: 'Any-attempt Group' }).click()
  await expect(page.getByRole('option', { name: 'old-history' })).toBeVisible()
  await page.getByRole('button', { name: 'Any-attempt Group' }).click()
  const initialListCount = logRequests.length
  await page.getByRole('button', { name: 'Apply', exact: true }).first().click()
  await expect.poll(() => logRequests.length).toBe(initialListCount + 1)
  await expect.poll(() => facetRequests.length).toBe(2)
  expect(facetRequests[1]?.search).toBe(facetRequests[0]?.search)
  await page.getByRole('button', { name: 'Any-attempt Group' }).click()
  await expect(page.getByRole('option', { name: 'new-history' })).toBeVisible()
  await expect(page.getByRole('option', { name: 'old-history' })).toHaveCount(0)
})

test('retrying a failed list load refreshes historical facets too', async ({ page }) => {
  const routes = await installRequestLogTableRoutes(page)
  routes.failNextList()
  let facetCount = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname === '/api/logs/facets') facetCount += 1
  })
  await page.goto('/logs', { waitUntil: 'commit' })
  await expect(page.getByText('Unable to load request logs.')).toBeVisible({ timeout: 60_000 })
  await expect.poll(() => facetCount).toBe(1)
  await page.getByRole('button', { name: 'Retry' }).click()
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4)
  await expect.poll(() => facetCount).toBe(2)
})
