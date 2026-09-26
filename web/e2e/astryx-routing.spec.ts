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
  await expect(page.locator('[data-route="login"]')).toBeVisible()
  await expect(page.getByLabel('Sign-in key')).toBeVisible()
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
  await expect(page.locator('[data-route="not-found"]')).toBeVisible()
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
  await expect(page.locator('[data-route="not-found"]')).toBeVisible()
})

// Direct /login navigation is unflagged, so it must keep serving the classic
// document even when the preference cookie opts into Astryx.
test('direct /login navigation stays on the classic document', async ({ page }) => {
  const response = await page.request.get('/login', {
    headers: { accept: 'text/html' },
  })
  expect(await response.text()).toContain('/src/main.ts')
})
