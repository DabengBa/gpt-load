import { expect, test } from '@playwright/test'

import {
  ACCESS_KEY,
  ADMIN_KEY,
  AFFINITY_KEY,
  DIFFERENT_AFFINITY_KEY,
  EMPTY_AFFINITY_KEY,
  installRequestLogAffinityRoutes,
  openRequestLogs,
} from './fixtures/request-log-affinity.ts'

const preservedQuery =
  '&from_ms=1700000000000&to_ms=1700003600000&status=success&client_model=gpt-4o&limit=50&selected_request_id=11111111-1111-4111-8111-111111111111&log_cursors=%5B%22old-cursor%22%5D'

function latestLogRequest(routes: { logRequests: URL[] }): URL {
  const request = routes.logRequests.at(-1)
  expect(request).toBeDefined()
  return request as URL
}

test.describe('request log affinity filter', () => {
  test('direct logs route loads and is available in the primary navigation', async ({ page }) => {
    for (const principal of ['admin', 'access_key'] as const) {
      const routes = await installRequestLogAffinityRoutes(page, principal)
      await openRequestLogs(page, routes)

      await expect(page).toHaveURL(/\/logs$/u)
      await expect(page.getByRole('link', { name: 'Request logs' }).first()).toBeVisible()
      await expect(page.getByRole('link', { name: 'Request logs' }).first()).toHaveAttribute(
        'aria-current',
        'page',
      )
    }
  })

  test('legacy monitor logs query normalizes to the principal default without log filters', async ({
    page,
  }) => {
    const logQuery =
      'tab=logs&from_ms=1700000000000&to_ms=1700003600000&selected_request_id=11111111-1111-4111-8111-111111111111&log_cursors=%5B%22old-cursor%22%5D&affinity_key=0123456789abcdef%2A%2A%2A%2Afedcba9876543210&status=success&client_model=gpt-4o&limit=50'

    for (const [principal, expectedTab] of [
      ['admin', 'health'],
      ['access_key', 'usage'],
    ] as const) {
      const routes = await installRequestLogAffinityRoutes(page, principal)
      await page.goto(`/monitor?${logQuery}`)
      await expect(page).toHaveURL(/\/monitor(?:\?|$)/u)
      await expect.poll(() => new URL(page.url()).searchParams.get('tab')).toBe(expectedTab)

      const normalizedURL = new URL(page.url())
      expect(normalizedURL.pathname).toBe('/monitor')
      expect(normalizedURL.searchParams.get('tab')).toBe(expectedTab)
      for (const key of [
        'from_ms',
        'to_ms',
        'selected_request_id',
        'log_cursors',
        'affinity_key',
        'status',
        'client_model',
        'limit',
      ]) {
        expect(normalizedURL.searchParams.has(key)).toBe(false)
      }
      await expect(page.getByRole('link', { name: 'Request logs' }).first()).toBeVisible()
      await expect(page.locator('.logs-tab')).toHaveCount(0)
      expect(routes.logRequests).toHaveLength(0)
    }
  })

  test('admin key supports click, Enter, Space, exact first page, and preserved filters', async ({
    page,
  }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, preservedQuery)

    const keyButtons = page.getByTestId('logs-affinity-key-filter')
    await expect(keyButtons).toHaveCount(3)
    await expect(page.locator('.logs-list__affinity-key-cell code')).toHaveCount(1)
    const timeColumnStyle = await page
      .locator('.logs-list__time')
      .first()
      .evaluate((element) => {
        const style = getComputedStyle(element)
        return { fontFamily: style.fontFamily, fontSize: style.fontSize }
      })
    expect(timeColumnStyle.fontFamily).toContain('ui-monospace')
    expect(timeColumnStyle.fontSize).toBe('10.5px')
    await test.info().attach('time-column-computed-style.json', {
      body: JSON.stringify(timeColumnStyle, null, 2),
      contentType: 'application/json',
    })
    await page.keyboard.press('Escape')
    await expect(page.locator('.app-drawer__overlay')).toBeHidden()

    await keyButtons.first().click()
    const clickedURL = new URL(page.url())
    expect(clickedURL.searchParams.get('affinity_key')).toBe(AFFINITY_KEY)
    expect(clickedURL.searchParams.get('from_ms')).toBe('1700000000000')
    expect(clickedURL.searchParams.get('to_ms')).toBe('1700003600000')
    expect(clickedURL.searchParams.get('status')).toBe('success')
    expect(clickedURL.searchParams.get('client_model')).toBe('gpt-4o')
    expect(clickedURL.searchParams.get('limit')).toBe('50')
    expect(clickedURL.searchParams.has('selected_request_id')).toBe(false)
    expect(clickedURL.searchParams.has('log_cursors')).toBe(false)
    const filteredRequest = latestLogRequest(routes)
    expect(filteredRequest.searchParams.get('affinity_key')).toBe(AFFINITY_KEY)
    expect(filteredRequest.searchParams.has('cursor')).toBe(false)
    await expect(page.getByText(AFFINITY_KEY).first()).toBeVisible()

    await page.getByRole('button', { name: /Remove filter.*Affinity scope key/u }).click()
    await expect(page).not.toHaveURL(/affinity_key=/u)
    expect(latestLogRequest(routes).searchParams.has('affinity_key')).toBe(false)

    await openRequestLogs(page, routes, preservedQuery)
    await page.keyboard.press('Escape')
    await expect(page.locator('.app-drawer__overlay')).toBeHidden()
    await expect(keyButtons.first()).toBeVisible()
    await keyButtons.first().press('Enter')
    await expect(page).toHaveURL(/affinity_key=/u)
    expect(latestLogRequest(routes).searchParams.get('affinity_key')).toBe(AFFINITY_KEY)

    await openRequestLogs(page, routes, preservedQuery)
    await page.keyboard.press('Escape')
    await expect(page.locator('.app-drawer__overlay')).toBeHidden()
    await keyButtons.first().press('Space')
    await expect(page).toHaveURL(/affinity_key=/u)
    expect(latestLogRequest(routes).searchParams.get('affinity_key')).toBe(AFFINITY_KEY)

    await openRequestLogs(
      page,
      routes,
      `&affinity_key=${encodeURIComponent(DIFFERENT_AFFINITY_KEY)}`,
    )
    await expect(page.getByText(DIFFERENT_AFFINITY_KEY).first()).toBeVisible()
  })

  test('reapplying unchanged filters refreshes the current log page', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(
      page,
      routes,
      '&from_ms=1700000000000&to_ms=1700003600000&status=success&client_model=gpt-4o&limit=50',
    )
    const initialRequestCount = routes.logRequests.length

    await page.getByRole('button', { name: 'Apply' }).first().click()
    await expect.poll(() => routes.logRequests.length).toBe(initialRequestCount + 1)
  })

  test('empty filtered pages show the filtered empty state', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, `&affinity_key=${encodeURIComponent(EMPTY_AFFINITY_KEY)}`)

    await expect(page.getByText('No logs match the current filters')).toBeVisible()
    expect(latestLogRequest(routes).searchParams.get('affinity_key')).toBe(EMPTY_AFFINITY_KEY)
  })

  test('invalid draft and URL filters do not issue an unfiltered request', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, '')
    const initialRequestCount = routes.logRequests.length

    await page.getByRole('button', { name: /More filters/u }).click()
    const affinityInput = page.locator('#logs-affinity-key')
    await affinityInput.fill('not-a-canonical-key')
    await page.getByRole('button', { name: 'Apply' }).last().click()
    await expect(affinityInput).toHaveAttribute('aria-invalid', 'true')
    await expect(page.getByRole('alert')).toContainText('canonical 16')
    expect(routes.logRequests).toHaveLength(initialRequestCount)

    await page.goto('/logs?affinity_key=not-a-canonical-key')
    await expect(page.getByRole('alert')).toContainText('canonical 16')
    expect(routes.logRequests).toHaveLength(initialRequestCount)
  })

  test('invalid URL filters remain visible and block both apply paths', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')

    for (const affinityKey of ['', 'not-a-canonical-key']) {
      await openRequestLogs(page, routes, `&affinity_key=${encodeURIComponent(affinityKey)}`)
      await expect(page.getByRole('alert')).toContainText('canonical 16')
      const initialRequestCount = routes.logRequests.length

      await page.getByRole('button', { name: 'Apply' }).first().click()
      await expect(page.getByRole('alert')).toContainText('canonical 16')
      expect(routes.logRequests).toHaveLength(initialRequestCount)

      await page.getByRole('button', { name: /More filters/u }).click()
      await page.getByRole('button', { name: 'Apply' }).last().click()
      await expect(page.getByRole('alert')).toContainText('canonical 16')
      expect(routes.logRequests).toHaveLength(initialRequestCount)
    }
  })
  test('access-key principals cannot see or send affinity filters', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'access_key')
    await openRequestLogs(
      page,
      routes,
      `&affinity_key=${encodeURIComponent(AFFINITY_KEY)}&from_ms=1700000000000&to_ms=1700003600000`,
    )

    await expect(page.getByRole('columnheader', { name: 'Affinity scope key' })).toHaveCount(0)
    await expect(page.getByTestId('logs-affinity-key-filter')).toHaveCount(0)
    await expect(page.locator('#logs-affinity-key')).toHaveCount(0)
    await expect(page.getByRole('button', { name: /Affinity scope key/u })).toHaveCount(0)
    expect(latestLogRequest(routes).searchParams.has('affinity_key')).toBe(false)

    await page.getByRole('button', { name: /More filters/u }).click()
    await expect(page.locator('#logs-affinity-key')).toHaveCount(0)
    expect(page.url()).not.toContain('affinity_key=')
  })

  test('fixture credentials are deterministic and scoped', async ({ page }) => {
    expect(ADMIN_KEY).toBe('e2e-admin-key')
    expect(ACCESS_KEY).toBe('e2e-access-key')
    await installRequestLogAffinityRoutes(page, 'admin')
  })
})
