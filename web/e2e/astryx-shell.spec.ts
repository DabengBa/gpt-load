import { expect, test, type Page } from '@playwright/test'

// B9: shell parity on the Astryx entry. Covers the AuthGate-validated
// topbar shell (admin vs access_key principal visibility), the
// login → shell → logout round trip, the not-found view, the preferences
// panel (theme/locale/frontend switch), and compact-density measurements
// against the classic token values (gate #6 input: ±1px).

// Cold Vite transforms of the full route graph exceed the default test
// timeout; keep the same headroom as astryx-groups.spec.ts.
test.setTimeout(90_000)

type Principal = 'admin' | 'access_key'

async function seedSession(page: Page, principal: Principal): Promise<void> {
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/auth/session') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          code: 0,
          message: 'ok',
          data: { authenticated: true, principal_type: principal },
        }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: {} }),
    })
  })
}

const adminNav = [
  'Home',
  'Groups',
  'Models',
  'Access keys',
  'Monitor',
  'Dispatch center',
  'Request logs',
  'Settings',
] as const
const accessKeyNav = ['Home', 'Models', 'Monitor', 'Request logs'] as const

test('login → authed shell → sign out round trip', async ({ page }) => {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/auth/session') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          code: 0,
          message: 'ok',
          data: { authenticated: true, principal_type: 'admin' },
        }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: {} }),
    })
  })

  await page.goto('/login', { waitUntil: 'load' })
  const input = page.getByLabel('Sign-in key', { exact: true })
  await expect(input).toBeVisible()
  await expect(input).toBeFocused()

  await input.fill('e2e-auth-key')
  await page.getByRole('button', { name: 'Sign in', exact: true }).click()
  await page.waitForURL(/\/$/, { timeout: 10_000 })

  // Authenticated shell: topbar nav + import action for an admin principal.
  const nav = page.locator('[data-testid="desktop-nav"]')
  await expect(nav).toBeVisible()
  for (const label of adminNav) {
    await expect(nav.getByRole('link', { name: label })).toBeVisible()
  }
  await expect(
    page.getByRole('link', { name: 'Import credentials' }),
  ).toBeVisible()
  expect(
    await page.evaluate(() => window.localStorage.getItem('gpt-load.auth-key')),
  ).toBe('e2e-auth-key')

  // Sign out from the preferences popover: back on /login, key cleared.
  await page
    .getByRole('button', { name: 'Menu and preferences' })
    .click()
  await page.getByRole('button', { name: 'Sign out' }).click()
  await page.waitForURL(/\/login/, { timeout: 10_000 })
  expect(
    await page.evaluate(() => window.localStorage.getItem('gpt-load.auth-key')),
  ).toBeNull()
  await expect(page.getByLabel('Sign-in key', { exact: true })).toBeVisible()
})

// Entry goes through /settings: it is the flagged document path, and for an
// access_key principal the adminOnly guard bounces to home — so the reduced
// nav assertions run on the Astryx shell after the redirect.
test('access_key principal sees reduced nav, read-only badge, no import', async ({
  page,
}) => {
  await seedSession(page, 'access_key')
  await page.goto('/settings', { waitUntil: 'load' })
  await page.waitForURL(/\/$/, { timeout: 10_000 })

  const nav = page.locator('[data-testid="desktop-nav"]')
  await expect(nav).toBeVisible()
  for (const label of accessKeyNav) {
    await expect(nav.getByRole('link', { name: label })).toBeVisible()
  }
  for (const label of ['Groups', 'Access keys', 'Dispatch center', 'Settings']) {
    await expect(nav.getByRole('link', { name: label })).toHaveCount(0)
  }
  await expect(
    page.getByRole('link', { name: 'Import credentials' }),
  ).toHaveCount(0)
  await expect(page.getByText('Access key · Read-only')).toBeVisible()
})

test('not-found shows the requested path and routes back home', async ({
  page,
}) => {
  await seedSession(page, 'admin')
  await page.goto('/no-such-console-page', { waitUntil: 'load' })
  await expect(
    page.getByRole('heading', { name: 'This page does not exist' }),
  ).toBeVisible()
  await expect(page.locator('code', { hasText: '/no-such-console-page' })).toBeVisible()
  await page.getByRole('link', { name: 'Back to Home' }).click()
  await page.waitForURL(/\/$/, { timeout: 10_000 })
})

test('preferences panel switches theme and locale', async ({ page }) => {
  await seedSession(page, 'admin')
  await page.goto('/settings', { waitUntil: 'load' })
  await expect(page.locator('[data-testid="desktop-nav"]')).toBeVisible()

  await page.getByRole('button', { name: 'Menu and preferences' }).click()

  // Theme segment: Dark writes data-theme="dark" on <html>.
  await page.getByRole('radio', { name: 'Dark' }).click()
  await expect
    .poll(() => page.evaluate(() => document.documentElement.dataset.theme))
    .toBe('dark')

  // Locale segment: 中文 re-renders shell text and syncs <html lang>.
  await page.getByRole('radio', { name: '中文' }).click()
  await expect
    .poll(() => page.evaluate(() => document.documentElement.lang))
    .toBe('zh-CN')
  await expect(
    page.getByRole('button', { name: '菜单与偏好设置' }),
  ).toBeVisible()
})

// Density parity measurements (gate #6 input). Values come from
// astryx/theme/tokens.css, which the Astryx entry imports at the
// tokens layer; each rendered metric must land within ±1px.
const densityExpectations = [
  { metric: 'topbar height', expected: 54 },
  { metric: 'topbar padding-inline', expected: 30 },
  { metric: 'import action height', expected: 30 },
  { metric: 'shell font-size', expected: 13.5 },
] as const

test('compact density matches classic tokens within ±1px', async ({ page }) => {
  await seedSession(page, 'admin')
  await page.goto('/settings', { waitUntil: 'load' })
  await page.locator('[data-testid="desktop-nav"]').waitFor()

  const measurements = await page.evaluate(() => {
    const topbar = document.querySelector('header')
    const shell = document.querySelector('[data-testid="astryx-shell"]')
    const importAction = document.querySelector(
      '[aria-label="Import credentials"]',
    )
    const navLink = document.querySelector('[data-testid="desktop-nav"] a')
    if (!topbar || !shell || !importAction || !navLink) {
      throw new Error('DENSITY_TARGETS_MISSING')
    }
    const topbarStyle = getComputedStyle(topbar)
    return {
      'topbar height': topbar.getBoundingClientRect().height,
      'topbar padding-inline': parseFloat(topbarStyle.paddingLeft),
      'import action height': importAction.getBoundingClientRect().height,
      'shell font-size': parseFloat(getComputedStyle(shell).fontSize),
      'nav link font-size': parseFloat(getComputedStyle(navLink).fontSize),
    }
  })

  for (const { metric, expected } of densityExpectations) {
    expect(
      Math.abs(measurements[metric] - expected),
      `${metric}: ${measurements[metric]}px vs classic ${expected}px`,
    ).toBeLessThanOrEqual(1)
  }
  // Nav-link type is a fixed 13px value in both shells (not a token).
  expect(measurements['nav link font-size']).toBe(13)
})
