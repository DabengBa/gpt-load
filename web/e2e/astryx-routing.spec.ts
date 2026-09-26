import { expect, test } from '@playwright/test'

// B7: TanStack Router assembly on the Astryx entry. Runs in the `astryx`
// project (preference cookie seeded). Guard behavior mirrors classic:
// requiresAuth redirects to /login?redirect=..., adminOnly routes send
// access_key principals home, trailing-slash and case mismatches hit
// not-found.

const FLAGGED_PATH = '/settings'

test('unauthenticated access to a guarded route redirects to login with redirect', async ({
  page,
  context,
}) => {
  await context.clearCookies()
  await context.addCookies([
    { name: 'gpt-load.frontend', value: 'astryx', domain: '127.0.0.1', path: '/' },
  ])
  await page.goto(FLAGGED_PATH, { waitUntil: 'load' })
  await page.waitForURL(/\/login\?.*redirect=/, { timeout: 10_000 })
  const url = new URL(page.url())
  expect(url.pathname).toBe('/login')
  expect(url.searchParams.get('redirect')).toBe(FLAGGED_PATH)
  await expect(
    page.getByRole('heading', { name: 'Sign in to GPT-Load' }),
  ).toBeVisible()
  await expect(page.getByLabel('Sign-in key', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible()
})

test('group detail param routes match and render the stub', async ({
  page,
  context,
}) => {
  // An anonymous user gets bounced to login before the group stub renders —
  // but only after the route matched, so /login carries the full redirect.
  await context.clearCookies()
  await context.addCookies([
    { name: 'gpt-load.frontend', value: 'astryx', domain: '127.0.0.1', path: '/' },
  ])
  await page.goto('/groups/42', { waitUntil: 'load' })
  await page.waitForURL(/\/login/, { timeout: 10_000 })
  const url = new URL(page.url())
  expect(url.searchParams.get('redirect')).toBe('/groups/42')
})

test('a trailing slash misses the manifest route and renders not-found', async ({
  page,
  context,
}) => {
  await context.clearCookies()
  await context.addCookies([
    { name: 'gpt-load.frontend', value: 'astryx', domain: '127.0.0.1', path: '/' },
  ])
  // '/settings/' is not a manifest path, so the server still serves the
  // Astryx document (cookie-only fallback) and the client router 404s it.
  await page.goto('/settings/', { waitUntil: 'load' })
  await expect(
    page.getByRole('heading', { name: 'This page does not exist' }),
  ).toBeVisible()
})

test('route matching is case-sensitive like the classic router', async ({
  page,
  context,
}) => {
  await context.clearCookies()
  await context.addCookies([
    { name: 'gpt-load.frontend', value: 'astryx', domain: '127.0.0.1', path: '/' },
  ])
  await page.goto('/SETTINGS', { waitUntil: 'load' })
  await expect(
    page.getByRole('heading', { name: 'This page does not exist' }),
  ).toBeVisible()
})

// In-app navigation to a classic-owned path must hand off with a document
// navigation: /monitor is unflagged, so only a full reload lets the server
// select the classic document — an SPA nav would strand the user on a stub.
test('nav to a classic-owned route leaves the Astryx document', async ({
  page,
  context,
}) => {
  await context.clearCookies()
  await context.addCookies([
    { name: 'gpt-load.frontend', value: 'astryx', domain: '127.0.0.1', path: '/' },
  ])
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        path === '/api/auth/session'
          ? {
              code: 0,
              message: 'ok',
              data: { authenticated: true, principal_type: 'admin' },
            }
          : { code: 0, message: 'ok', data: {} },
      ),
    })
  })

  await page.goto('/groups', { waitUntil: 'load' })
  await expect(page.getByTestId('astryx-shell')).toBeVisible()

  await Promise.all([
    page.waitForURL(/\/monitor$/, { timeout: 10_000 }),
    page.locator('[data-testid="desktop-nav"]').getByRole('link', { name: 'Monitor' }).click(),
  ])

  // The classic document now owns the page: its shell carries the desktop-nav
  // class, and the astryx shell marker is gone.
  await expect(page.locator('nav.desktop-nav')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId('astryx-shell')).toHaveCount(0)
})

// Unflagged paths keep serving the classic document even when the
// preference cookie opts into Astryx; /login is flagged, so it serves the
// Astryx document under the same cookie.
test('document selection follows the manifest astryx flag', async ({ page }) => {
  const classic = await page.request.get('/monitor', {
    headers: { accept: 'text/html' },
  })
  expect(await classic.text()).toContain('/src/main.ts')

  const astryx = await page.request.get('/login', {
    headers: { accept: 'text/html' },
  })
  expect(await astryx.text()).toContain('/src/frontends/astryx/main.tsx')
})
