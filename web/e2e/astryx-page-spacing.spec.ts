import { expect, test } from '@playwright/test'

import { installRequestLogTableRoutes } from './fixtures/request-log-display'

test.setTimeout(90_000)

const pages = [
  '/',
  '/groups',
  '/groups/1',
  '/monitor',
  '/logs',
  '/settings',
  '/import',
  '/schedule',
]

for (const width of [390, 1440, 1920]) {
  test(`management page frames keep responsive gutters at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 })
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await installRequestLogTableRoutes(page)
    await page.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname
      if (path === '/api/auth/session' || path.startsWith('/api/logs')) return route.fallback()
      let data: unknown = {}
      if (path === '/api/channels') data = { items: [], total: 0 }
      if (path === '/api/groups/options' || path === '/api/access-keys/options') data = []
      if (path === '/api/groups') {
        data = {
          observed_at_ms: 1_730_000_000_000,
          summary: { total: 0, available: 0, unavailable: 0, disabled: 0 },
          items: [],
          pagination: { page: 1, page_size: 100, total_items: 0, total_pages: 0 },
        }
      }
      if (path === '/api/groups/1') {
        data = {
          id: 1,
          name: 'Spacing group',
          channel_id: 'openai',
          connection_type: 'api_key',
          params: {},
          provider_url: null,
          service_status: 'available',
          service_status_reason: null,
          credential_configured: false,
          credential_status: null,
          model_count: 0,
        }
      }
      if (path === '/api/groups/1/credential') data = { credential: null, observation: null }
      if (path === '/api/groups/1/models') data = { items: [], total: 0, pending: 0 }
      // Error states retain the same page frame; full content is exercised by domain specs.
      const unavailable = [
        '/api/home',
        '/api/settings',
        '/api/usage',
        '/api/groups/1/settings',
      ].includes(path)
      await route.fulfill({
        status: unavailable ? 503 : 200,
        contentType: 'application/json',
        body: JSON.stringify({ code: unavailable ? 'UNAVAILABLE' : 0, message: 'fixture', data }),
      })
    })

    for (const path of pages) {
      await page.goto(path)
      const root = page.locator('#main-content > :first-child')
      await expect(root, `page root for ${path} at ${width}px`).toBeVisible({ timeout: 60_000 })
      const geometry = await root.evaluate((element) => {
        const style = getComputedStyle(element)
        const rect = element.getBoundingClientRect()
        const main = document.querySelector('#main-content')!
        const mainStyle = getComputedStyle(main)
        return {
          top: parseFloat(style.paddingTop),
          bottom: parseFloat(style.paddingBottom),
          inline: parseFloat(style.paddingLeft),
          innerWidth: rect.width - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
          innerLeft: rect.left + parseFloat(style.paddingLeft),
          mainPadding: mainStyle.padding,
          overflow: Math.max(
            document.documentElement.scrollWidth,
            document.body.scrollWidth,
            main.scrollWidth,
          ),
        }
      })
      expect.soft(geometry.top, `${path} top`).toBe(width === 390 ? 16 : 26)
      expect.soft(geometry.bottom, `${path} bottom`).toBe(width === 390 ? 40 : 60)
      expect.soft(geometry.inline, `${path} gutter`).toBe(width === 390 ? 12 : 24)
      expect.soft(geometry.mainPadding, `${path} shell padding`).toBe('0px')
      expect.soft(geometry.overflow, `${path} horizontal overflow`).toBeLessThanOrEqual(width)
      if (path === '/monitor' || path === '/logs' || path === '/groups/1') {
        const expectedWidth = Math.min(width - (width === 390 ? 24 : 48), 1240)
        expect.soft(geometry.innerWidth, `${path} content width`).toBe(expectedWidth)
        expect.soft(geometry.innerLeft, `${path} alignment`).toBe((width - expectedWidth) / 2)
      } else {
        const content = await root.locator(':scope > :first-child').boundingBox()
        expect(content, `${path} content frame`).not.toBeNull()
        const expectedWidth = Math.min(
          width - (width === 390 ? 24 : 48),
          path === '/schedule' ? 1440 : 1240,
        )
        expect.soft(content!.width, `${path} content width`).toBe(expectedWidth)
        expect.soft(content!.x, `${path} alignment`).toBe((width - expectedWidth) / 2)
      }
      await page.screenshot({
        path: testInfo.outputPath(`${path.replaceAll('/', '_') || 'home'}.png`),
        fullPage: true,
      })
      if (path === '/logs') {
        const values = page.getByTestId('logs-list__token-values').first()
        await expect(values).toBeVisible()
        const separated = await values.evaluate((element) => {
          const line = element.firstElementChild!
          const hint = element.querySelector('[data-testid="logs-list__cache-rate"]')!
          const range = document.createRange()
          range.selectNodeContents(line)
          const text = range.getBoundingClientRect()
          const button = hint.getBoundingClientRect()
          return text.right <= button.left || text.bottom <= button.top || button.bottom <= text.top
        })
        expect(separated, 'token text and cache-rate control do not overlap').toBe(true)
      }
    }
    expect(errors).toEqual([])
  })
}
