import { expect, test } from '@playwright/test'

// B7: TanStack Router assembly on the single document entry. Guard behavior
// mirrors the retired classic frontend:
// requiresAuth redirects to /login?redirect=..., adminOnly routes send
// access_key principals home, trailing-slash and case mismatches hit
// not-found.

const FLAGGED_PATH = '/settings'

// `commit` waits only for response headers — the first navigation in a cold
// run eats the vite transform of the astryx graph, which can exceed 30s.
test.setTimeout(90_000)

test('unauthenticated access to a guarded route redirects to login with redirect', async ({
  page,
}) => {
  await page.goto(FLAGGED_PATH, { waitUntil: 'commit' })
  await page.waitForURL(/\/login\?.*redirect=/, { timeout: 60_000 })
  const url = new URL(page.url())
  expect(url.pathname).toBe('/login')
  expect(url.searchParams.get('redirect')).toBe(FLAGGED_PATH)
  await expect(page.getByRole('heading', { name: 'Sign in to GPT-Load' })).toBeVisible({
    timeout: 60_000,
  })
  await expect(page.getByLabel('Sign-in key', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible()
})

test('group detail param routes match and render the stub', async ({ page }) => {
  // An anonymous user gets bounced to login before the group stub renders —
  // but only after the route matched, so /login carries the full redirect.
  await page.goto('/groups/42', { waitUntil: 'commit' })
  await page.waitForURL(/\/login/, { timeout: 10_000 })
  const url = new URL(page.url())
  expect(url.searchParams.get('redirect')).toBe('/groups/42')
})

test('a trailing slash misses the manifest route and renders not-found', async ({ page }) => {
  // '/settings/' is not a manifest path: the server falls back to the
  // single document and the client router 404s it.
  await page.goto('/settings/', { waitUntil: 'commit' })
  await expect(page.getByRole('heading', { name: 'This page does not exist' })).toBeVisible({
    timeout: 60_000,
  })
})

test('route matching is case-sensitive like the classic router', async ({ page }) => {
  await page.goto('/SETTINGS', { waitUntil: 'commit' })
  await expect(page.getByRole('heading', { name: 'This page does not exist' })).toBeVisible({
    timeout: 60_000,
  })
})

// In-app navigation to an Astryx-owned path stays inside the document:
// /import is flagged since Phase 4, so the shell link performs an SPA
// navigation (canonicalizing to ?mode=new) without a document handoff.
test('nav to a flagged route stays inside the Astryx document', async ({ page }) => {
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

  await page.goto('/groups', { waitUntil: 'commit' })
  await expect(page.getByTestId('astryx-shell')).toBeVisible({ timeout: 60_000 })

  await page.getByRole('link', { name: 'Import credentials' }).click()
  await expect(page).toHaveURL(/\/import\?.*mode=new/, { timeout: 10_000 })

  // Still the Astryx document: the shell marker survives and the import view
  // rendered in place.
  await expect(page.getByTestId('astryx-shell')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: 10_000,
  })
})

// Every page route is served by the single Astryx document; a leftover
// gpt-load.frontend cookie no longer influences document selection.
test('document selection ignores the retired frontend cookie', async ({ page, context }) => {
  const document = await page.request.get('/import', {
    headers: { accept: 'text/html' },
  })
  expect(await document.text()).toContain('/src/frontends/astryx/main.tsx')

  await context.clearCookies()
  await context.addCookies([
    { name: 'gpt-load.frontend', value: 'classic', domain: '127.0.0.1', path: '/' },
  ])
  const withClassicCookie = await page.request.get('/import', {
    headers: { accept: 'text/html' },
  })
  expect(await withClassicCookie.text()).toContain('/src/frontends/astryx/main.tsx')
})
