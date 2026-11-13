import { expect, test } from '@playwright/test'

test('announcement loading, error recovery, long content and keyboard controls', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.addInitScript(() => {
    localStorage.setItem('gpt-load.auth-key', 'e2e-announcements')
    localStorage.setItem('gpt-load.locale', 'en-US')
  })
  let reads = 0
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/announcements') {
      reads++
      if (reads === 1) {
        await new Promise((resolve) => setTimeout(resolve, 500))
        return route.fulfill({ status: 500, json: { code: 500, message: 'error' } })
      }
      return route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: {
            sites: [],
            items: [
              {
                id: 'one',
                site_root_url: 'https://example.com',
                site_name: 'Example',
                content: 'long'.repeat(300),
                extra: 'Extra',
                level: '',
                source: 'notice',
                event_at_ms: 1700000000000,
                fetched_at_ms: 1700000000000,
                groups: [],
              },
            ],
          },
        },
      })
    }
    return route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: path === '/api/auth/session' ? { authenticated: true, principal_type: 'admin' } : {},
      },
    })
  })
  await page.goto('/announcements')
  await expect(page.getByRole('status', { name: 'Loading this section…' })).toBeVisible()
  await expect(page.getByText('Failed to load announcements')).toBeVisible()
  const retry = page.getByRole('button', { name: 'Retry' })
  await retry.focus()
  await page.keyboard.press('Enter')
  await expect(page.getByText('Extra')).toBeVisible()
  if (process.env.UI_SCREENSHOTS)
    await page.screenshot({
      path: `${process.env.UI_SCREENSHOTS}/announcements-mobile.png`,
      fullPage: true,
    })
  const panel = page.locator('[data-content-panel]')
  await expect(panel).toHaveCount(1)
  await expect(panel.getByRole('textbox', { name: 'Search' })).toBeVisible()
  await expect(panel.locator('article')).toHaveCount(1)
  await expect(panel.locator('article').locator('[data-content-panel]')).toHaveCount(0)
  await expect(page.getByRole('link', { name: 'Example' })).toHaveAttribute(
    'rel',
    'noopener noreferrer',
  )
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.getByRole('textbox', { name: 'Search' }).focus()
  await page.keyboard.type('test')
  await expect(page.getByRole('textbox', { name: 'Search' })).toHaveValue('test')
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.getByRole('textbox', { name: 'Search' }).clear()
  await expect(page.getByText('Extra')).toBeVisible()
  if (process.env.UI_SCREENSHOTS)
    await page.screenshot({
      path: `${process.env.UI_SCREENSHOTS}/announcements-desktop.png`,
      fullPage: true,
    })
  await page.evaluate(() => localStorage.setItem('gpt-load.theme', 'dark'))
  await page.reload()
  await expect(page.getByText('Extra')).toBeVisible()
  await expect(panel).toHaveCSS('background-color', 'rgb(23, 27, 32)')
  if (process.env.UI_SCREENSHOTS)
    await page.screenshot({
      path: `${process.env.UI_SCREENSHOTS}/announcements-dark.png`,
      fullPage: true,
    })
})
