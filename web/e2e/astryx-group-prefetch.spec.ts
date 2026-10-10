import { expect, test, type Page, type Route } from '@playwright/test'

test.setTimeout(90_000)

async function mockGroupPrefetch(
  page: Page,
  principalType: 'admin' | 'access_key',
  summaryReady = Promise.resolve(),
) {
  const requests: string[] = []
  await page.addInitScript(
    (key) => window.localStorage.setItem('gpt-load.auth-key', key),
    'e2e-auth-key',
  )
  await page.route('**/api/**', async (route: Route) => {
    const path = new URL(route.request().url()).pathname
    requests.push(path)
    if (path === '/api/auth/session') {
      await route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: { authenticated: true, principal_type: principalType },
        },
      })
      return
    }
    if (path === '/api/groups') {
      await route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: {
            observed_at_ms: 1,
            summary: { total: 1, available: 1, unavailable: 0, disabled: 0 },
            items: [
              {
                id: 1,
                name: 'Group 0001',
                channel_id: 'openai',
                connection_type: 'api_key',
                params: {},
                provider_url: null,
                status: 'available',
                model_count: 1,
                client_model_count: 1,
                credential_configured: true,
                credential_status: 'available',
              },
            ],
            pagination: { page: 1, page_size: 100, total_items: 1, total_pages: 1 },
          },
        },
      })
      return
    }
    if (path === '/api/groups/1') {
      await summaryReady
      await route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: {
            id: 1,
            name: 'Group 0001',
            channel_id: 'openai',
            connection_type: 'api_key',
            params: {},
            provider_url: null,
            service_status: 'available',
            service_status_reason: null,
            credential_configured: true,
            credential_status: 'available',
            model_count: 1,
          },
        },
      })
      return
    }
    await route.fulfill({ json: { code: 0, message: 'ok', data: { items: [], total: 0 } } })
  })
  return requests
}

for (const intent of ['hover', 'focus'] as const) {
  test(`admin ${intent} prefetches only summary and deduplicates intent and click`, async ({
    page,
  }) => {
    let releaseSummary = () => {}
    const summaryReady = new Promise<void>((resolve) => {
      releaseSummary = resolve
    })
    const requests = await mockGroupPrefetch(page, 'admin', summaryReady)
    await page.goto('/groups', { waitUntil: 'load' })
    const link = page.getByRole('link', { name: 'View details for Group 0001' }).first()
    await link[intent]()
    await expect.poll(() => requests.filter((path) => path === '/api/groups/1').length).toBe(1)
    await link.dispatchEvent('pointerenter')
    await link.focus()
    await page.evaluate(() => new Promise(requestAnimationFrame))
    expect(requests.filter((path) => path === '/api/groups/1').length).toBe(1)
    const summaryResponse = page.waitForResponse(
      (response) => new URL(response.url()).pathname === '/api/groups/1',
    )
    releaseSummary()
    await summaryResponse
    await page.evaluate(() => new Promise(requestAnimationFrame))
    expect(requests).not.toContain('/api/groups/1/credential')
    expect(requests).not.toContain('/api/groups/1/models')
    expect(requests).not.toContain('/api/groups/1/settings')
    await link.focus()
    await link.click()
    await expect(page.getByRole('heading', { name: 'Group 0001' })).toBeVisible()
    expect(requests.filter((path) => path === '/api/groups/1').length).toBe(1)
  })
}

test('access-key intent does not prefetch a group summary', async ({ page }) => {
  const requests = await mockGroupPrefetch(page, 'access_key')
  await page.goto('/groups', { waitUntil: 'load' })
  await expect(page).toHaveURL(/\/$/)
  await expect(page.getByRole('table', { name: 'Group list' })).toHaveCount(0)
  expect(requests).not.toContain('/api/groups/1')
})

test('anonymous navigation retains login gate without group requests', async ({ page }) => {
  const requests: string[] = []
  page.on('request', (request) => {
    if (request.url().includes('/api/groups')) requests.push(request.url())
  })
  await page.goto('/groups', { waitUntil: 'load' })
  await expect(page).toHaveURL(/\/login\?redirect=/)
  expect(requests).toEqual([])
})
