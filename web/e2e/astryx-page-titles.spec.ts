import { expect, test } from '@playwright/test'

import { installRequestLogTableRoutes } from './fixtures/request-log-display'

test.setTimeout(90_000)

test('page headings stay accessible while matching the shell breakpoint', async ({ page }) => {
  await installRequestLogTableRoutes(page)
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/auth/session' || path.startsWith('/api/logs')) return route.fallback()
    await route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ code: 'UNAVAILABLE', message: 'fixture', data: {} }),
    })
  })

  for (const [path, id, name] of [
    ['/', 'home-title', 'Home'],
    ['/groups', 'groups-title', 'Groups'],
    ['/announcements', 'announcements-title', 'Announcements'],
    ['/schedule', 'schedule-title', 'Dispatch center'],
    ['/monitor', 'monitor-title', 'Monitor'],
    ['/settings', 'settings-title', 'Settings'],
    ['/logs', 'logs-title', 'Request logs'],
  ] as const) {
    await page.setViewportSize({ width: 861, height: 900 })
    await page.goto(path)
    const heading = page.locator(`h1#${id}`)
    await expect(heading, path).toHaveText(name, { timeout: 30_000 })
    await expect(page.getByRole('heading', { name, level: 1, exact: true })).toHaveCount(1)
    await expect(page.locator(`[aria-labelledby="${id}"]`).first()).toHaveAccessibleName(name)
    const topName = page.locator('header').first().getByText(name, { exact: true }).first()
    await expect(topName).toBeVisible()
    const desktop = await heading.evaluate((element) => ({
      position: getComputedStyle(element).position,
      width: element.getBoundingClientRect().width,
    }))
    expect(desktop, `${path}: visually hidden without removing its semantics`).toEqual({
      position: 'absolute',
      width: 1,
    })
    for (const width of [860, 390]) {
      await page.setViewportSize({ width, height: 900 })
      await expect(topName).toBeHidden()
      await expect(heading).toBeVisible()
      expect(await heading.evaluate((element) => getComputedStyle(element).position)).not.toBe(
        'absolute',
      )
    }
  }

  // These routes use the parent navigation label, not their own page title.
  await page.setViewportSize({ width: 1200, height: 900 })
  for (const [path, id] of [
    ['/import', 'import-page-title'],
    ['/groups/invalid', 'group-detail-title'],
  ] as const) {
    await page.goto(path)
    const heading = page.locator(`h1#${id}`)
    await expect(heading).toBeVisible()
    expect(await heading.evaluate((element) => getComputedStyle(element).position)).not.toBe(
      'absolute',
    )
  }
})
