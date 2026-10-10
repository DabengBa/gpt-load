import { expect, test } from '@playwright/test'

test.setTimeout(90_000)

test('cold login does not load business page modules', async ({ page }) => {
  const business = /features\/(?:home|groups|import|logs|monitor|settings)\/.*\.tsx?(?:\?|$)/
  const requests: string[] = []
  page.on('request', (request) => requests.push(request.url()))
  await page.goto('/login', { waitUntil: 'commit' })
  await expect(page.getByRole('heading', { name: 'Sign in to GPT-Load' })).toBeVisible({
    timeout: 60_000,
  })
  expect(requests.some((url) => business.test(url))).toBe(false)
})

test('intent preloads a guarded route without navigating or changing auth', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'e2e-admin-key'))
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        path === '/api/auth/session'
          ? { code: 0, message: 'ok', data: { authenticated: true, principal_type: 'admin' } }
          : { code: 0, message: 'ok', data: {} },
      ),
    })
  })
  await page.goto('/groups', { waitUntil: 'commit' })
  await expect(page.getByTestId('astryx-shell')).toBeVisible({ timeout: 60_000 })
  const before = page.url()
  const warmed = page.waitForRequest((request) =>
    request.url().includes('/features/home/HomeView.tsx'),
  )
  await page.getByTestId('desktop-nav').getByRole('link', { name: 'Home' }).hover()
  await warmed
  expect(page.url()).toBe(before)
  expect(await page.evaluate(() => localStorage.getItem('gpt-load.auth-key'))).toBe('e2e-admin-key')
  await page.getByRole('link', { name: 'Import credentials', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible()
})

test('failed business chunk recovers by authenticated retry', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'e2e-admin-key'))
  await page.route('**/api/**', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: { authenticated: true, principal_type: 'admin' },
      },
    }),
  )
  await page.route('**/features/groups/GroupsView.tsx*', (route) => route.abort())
  await page.goto('/groups', { waitUntil: 'commit' })
  await expect(page.getByRole('button', { name: 'Retry', exact: true })).toBeVisible({
    timeout: 60_000,
  })
  await page.unroute('**/features/groups/GroupsView.tsx*')
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Groups', exact: true })).toBeVisible({
    timeout: 60_000,
  })
})

test('direct deep route loads its business view', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'e2e-admin-key'))
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        path === '/api/auth/session'
          ? { code: 0, message: 'ok', data: { authenticated: true, principal_type: 'admin' } }
          : { code: 0, message: 'ok', data: {} },
      ),
    })
  })
  await page.goto('/groups', { waitUntil: 'commit' })
  await expect(page.getByRole('heading', { name: 'Groups', exact: true })).toBeVisible({
    timeout: 60_000,
  })
})
