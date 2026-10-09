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

for (const width of [390, 900, 1440, 1920]) {
  test(`management pages use compact gutters and available content width at ${width}px`, async ({
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
          summary: { total: 1, available: 1, unavailable: 0, disabled: 0 },
          items: [
            {
              id: 1,
              name: 'Spacing group',
              channel_id: 'openai',
              connection_type: 'api_key',
              params: {},
              provider_url: null,
              status: 'available',
              model_count: 4,
              client_model_count: 2,
              credential_configured: false,
              credential_status: null,
            },
          ],
          pagination: { page: 1, page_size: 100, total_items: 1, total_pages: 1 },
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
      if (path === '/api/model-route/schedule')
        data = {
          items: [
            {
              external_model: 'worker',
              protocol: 'openai-completions',
              operation: 'chat_completion',
              candidate_count: 1,
              group_count: 1,
              cooled_candidates: 0,
              blacklisted_candidates: 0,
            },
          ],
        }
      if (path === '/api/model-route/schedule/detail')
        data = {
          observed_at_ms: 1_700_000_000_000,
          snapshot_revision: 11,
          external_model: 'worker',
          protocol: 'openai-completions',
          operation: 'chat_completion',
          route_requirement: 'any',
          access_key: { id: 7, name: 'e2e access key', status: 'active' },
          routable: true,
          reason_code: null,
          groups: [
            {
              group_id: 1,
              group_name: 'Spacing group',
              channel_id: 'openai',
              enabled: true,
              request_count: 10,
              success_rate: 1,
              entries: [
                {
                  entry_id: 'entry-1',
                  model_id: 'model-a',
                  alias: '',
                  weight: 50,
                  priority: 1,
                  enabled: true,
                  circuit_breaker: {
                    configured: { blacklist_threshold: null, cooldown_seconds: null },
                    effective: { blacklist_threshold: 3, cooldown_seconds: 60 },
                    sources: { blacklist_threshold: 'default', cooldown_seconds: 'default' },
                  },
                  reasoning: { configured: null, effective: null, source: 'provider_default' },
                  runtime: {
                    state: 'available',
                    cooldown_until_ms: null,
                    blacklist_release_at_ms: null,
                    failure_count: 0,
                    failure_version: 0,
                  },
                  included: true,
                  routable: true,
                  reason_code: null,
                  configured_share: 1,
                  effective_share: 1,
                  credentials: [],
                },
              ],
            },
          ],
        }
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
      await page.goto(path === '/schedule' ? '/schedule?schedule_model=worker' : path)
      if (path === '/groups')
        await expect(
          page
            .getByRole('table', { name: 'Group list' })
            .getByText('Spacing group', { exact: true }),
        ).toBeVisible()
      if (path === '/schedule')
        await expect(page.getByRole('table', { name: 'Schedule detail' })).toBeVisible()
      const root = page.locator('#main-content > :first-child')
      await expect(root, `page root for ${path} at ${width}px`).toBeVisible({ timeout: 60_000 })
      const geometry = await root.evaluate((element) => {
        const style = getComputedStyle(element)
        const rect = element.getBoundingClientRect()
        const main = document.querySelector('#main-content')!
        const mainStyle = getComputedStyle(main)
        const content = Array.from(element.children)
          .filter((child) => {
            const childStyle = getComputedStyle(child)
            return childStyle.position !== 'absolute' && childStyle.position !== 'fixed'
          })
          .map((child) => child.getBoundingClientRect())
          .filter((child) => child.width > 0 && child.height > 0)
        return {
          top: parseFloat(style.paddingTop),
          bottom: parseFloat(style.paddingBottom),
          leftPadding: parseFloat(style.paddingLeft),
          rightPadding: parseFloat(style.paddingRight),
          innerWidth: rect.width - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
          innerLeft: rect.left + parseFloat(style.paddingLeft),
          mainPadding: mainStyle.padding,
          mainLeft: main.getBoundingClientRect().left,
          mainWidth: main.getBoundingClientRect().width,
          content: content.map((child) => ({ left: child.left, width: child.width })),
          leadingSpace: Math.min(...content.map((child) => child.top)) - rect.top,
          trailingSpace: rect.bottom - Math.max(...content.map((child) => child.bottom)),
          overflow: Math.max(
            document.documentElement.scrollWidth,
            document.body.scrollWidth,
            main.scrollWidth,
          ),
        }
      })
      // Dense table/report pages use 16px; the remaining forms retain 24px.
      // Bound vertical whitespace too: full-width wrappers must not hide an
      // obsolete minimum height or a large empty area before/after content.
      const gutter = width === 390 ? 12 : ['/', '/monitor', '/schedule'].includes(path) ? 16 : 24
      expect.soft(geometry.leftPadding, `${path} left gutter`).toBe(gutter)
      expect.soft(geometry.rightPadding, `${path} right gutter`).toBe(gutter)
      for (const [name, value, max] of [
        ['top', geometry.top, 26],
        ['bottom', geometry.bottom, 60],
      ] as const) {
        expect.soft(value, `${path} ${name}: sufficient spacing`).toBeGreaterThanOrEqual(12)
        expect.soft(value, `${path} ${name}: no excessive whitespace`).toBeLessThanOrEqual(max)
      }
      // Empty portal targets may leave one grid gap. Keep this separate from
      // page padding so a large minimum-height spacer still fails.
      for (const [name, extra] of [
        ['leading', geometry.leadingSpace - geometry.top],
        ['trailing', geometry.trailingSpace - geometry.bottom],
      ] as const) {
        expect
          .soft(extra, `${path} ${name} content stays inside padding`)
          .toBeGreaterThanOrEqual(-1)
        expect.soft(extra, `${path} ${name} extra whitespace`).toBeLessThanOrEqual(16)
      }
      expect.soft(geometry.mainPadding, `${path} shell padding`).toBe('0px')
      expect.soft(geometry.overflow, `${path} horizontal overflow`).toBeLessThanOrEqual(width)
      const expectedWidth = geometry.mainWidth - gutter * 2
      expect.soft(geometry.innerWidth, `${path} available width`).toBeCloseTo(expectedWidth, 0)
      expect
        .soft(geometry.innerLeft, `${path} alignment`)
        .toBeCloseTo(geometry.mainLeft + gutter, 0)
      expect(geometry.content.length, `${path} visible content`).toBeGreaterThan(0)
      for (const [index, content] of geometry.content.entries()) {
        expect
          .soft(content.width, `${path} content ${index} occupies available width`)
          .toBeCloseTo(expectedWidth, 0)
        expect
          .soft(content.left, `${path} content ${index} alignment`)
          .toBeCloseTo(geometry.mainLeft + gutter, 0)
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
