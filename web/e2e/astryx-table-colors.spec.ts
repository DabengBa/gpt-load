import { expect, test, type Locator } from '@playwright/test'

import { installRequestLogTableRoutes } from './fixtures/request-log-display'

test.setTimeout(90_000)
// Read-only demo lane: no seeding, writes, or paid upstream calls.
test.skip(!process.env.GPT_LOAD_DEMO_AUTH_KEY, 'Requires populated local demo API and its auth key')

async function presentation(headers: Locator) {
  return headers.evaluateAll((cells) =>
    cells.map((cell) => {
      const s = getComputedStyle(cell)
      return {
        background: s.backgroundColor,
        color: s.color,
        font: s.fontFamily,
        size: s.fontSize,
        weight: s.fontWeight,
        spacing: s.letterSpacing,
        transform: s.textTransform,
        border: s.borderBottomColor,
      }
    }),
  )
}

for (const theme of ['light', 'dark']) {
  for (const width of [390, 1440]) {
    test(`opaque unified table chrome ${theme} ${width}`, async ({ page }, info) => {
      await page.setViewportSize({ width, height: 1000 })
      await page.addInitScript(
        ({ theme, key }) => {
          localStorage.setItem('gpt-load.theme', theme)
          localStorage.setItem('gpt-load.auth-key', key)
          localStorage.setItem('gpt-load.locale', 'en-US')
        },
        { theme, key: process.env.GPT_LOAD_DEMO_AUTH_KEY ?? '' },
      )
      await page.goto('/groups')
      const table = page.locator('table').first()
      await expect(table.locator('tbody tr').first()).toBeVisible()
      await page.mouse.move(0, 0)
      const headers = await presentation(table.locator('th'))
      await info.attach('groups-headers', {
        body: JSON.stringify(headers),
        contentType: 'application/json',
      })
      expect(new Set(headers.map((h) => JSON.stringify(h))).size).toBe(1)
      expect(headers[0].background).toBe(
        theme === 'light' ? 'rgb(245, 244, 241)' : 'rgb(18, 21, 26)',
      )
      const row = table.locator('tbody tr').first()
      const colors = await row
        .locator('td')
        .evaluateAll((cells) => cells.map((cell) => getComputedStyle(cell).backgroundColor))
      expect(new Set(colors).size).toBe(1)
      expect(colors[0]).toBe(theme === 'light' ? 'rgb(255, 255, 255)' : 'rgb(23, 27, 32)')
      if (width > 860) {
        expect(
          await row
            .locator('td')
            .evaluateAll(
              (cells) =>
                cells.filter((cell) => getComputedStyle(cell).position === 'sticky').length,
            ),
        ).toBeGreaterThanOrEqual(2)
        await table.evaluate((element) => {
          const scroller = element.parentElement
          if (scroller) scroller.scrollLeft = scroller.scrollWidth
        })
        expect(
          await row
            .locator('td')
            .evaluateAll((cells) => cells.map((cell) => getComputedStyle(cell).backgroundColor)),
        ).toEqual(colors)
      }
      await row.hover()
      await expect
        .poll(() =>
          row
            .locator('td')
            .evaluateAll((cells) => cells.map((cell) => getComputedStyle(cell).backgroundColor)),
        )
        .toEqual(colors.map(() => (theme === 'light' ? 'rgb(245, 244, 241)' : 'rgb(18, 21, 26)')))
      await page.goto('/schedule')
      await page
        .getByRole('button', { name: /demo-1-chat/ })
        .first()
        .click()
      const schedule = page.getByRole('table').first()
      await expect(schedule).toBeVisible()
      if (width > 860)
        expect(await presentation(schedule.getByRole('columnheader'))).toEqual(
          Array(8).fill(headers[0]),
        )
      const scheduleRow = schedule.getByRole('row').nth(width > 860 ? 1 : 0)
      await page.mouse.move(0, 0)
      expect(await scheduleRow.evaluate((row) => getComputedStyle(row).backgroundColor)).toBe(
        colors[0],
      )
      await scheduleRow.hover()
      await expect
        .poll(() => scheduleRow.evaluate((row) => getComputedStyle(row).backgroundColor))
        .toBe(theme === 'light' ? 'rgb(245, 244, 241)' : 'rgb(18, 21, 26)')
      await page.goto('/settings?section=credentials')
      const settings = page.locator('table').first()
      await expect(settings.locator('tbody tr').first()).toBeVisible()
      const settingsHeaders = await presentation(settings.locator('th'))
      expect(settingsHeaders.every((h) => JSON.stringify(h) === JSON.stringify(headers[0]))).toBe(
        true,
      )
      for (const path of ['/', '/monitor', '/groups/2?tab=models']) {
        await page.goto(path)
        const surface = page.getByRole('table').first()
        await expect(surface).toBeVisible()
        if (width > 860) {
          const cells = surface.locator('thead th, [role="columnheader"]')
          await expect(cells.first()).toBeVisible()
          const actual = await presentation(cells)
          expect(actual.every((h) => JSON.stringify(h) === JSON.stringify(headers[0]))).toBe(true)
        }
      }
      await installRequestLogTableRoutes(page)
      await page.goto('/logs')
      const logs = page.getByTestId('logs-list')
      await expect(logs).toBeVisible()
      if (width > 860) {
        const logHeaders = await presentation(logs.getByRole('columnheader'))
        expect(logHeaders.every((h) => JSON.stringify(h) === JSON.stringify(headers[0]))).toBe(true)
      }
      expect(
        await logs
          .getByTestId('logs-list__record')
          .first()
          .evaluate((row) => getComputedStyle(row).backgroundColor),
      ).toBe(theme === 'light' ? 'rgb(255, 255, 255)' : 'rgb(23, 27, 32)')
      const logRow = logs.getByTestId('logs-list__record').first()
      await logRow.hover()
      await expect
        .poll(() => logRow.evaluate((row) => getComputedStyle(row).backgroundColor))
        .toBe(theme === 'light' ? 'rgb(245, 244, 241)' : 'rgb(18, 21, 26)')
      await info.attach('logs-table', { body: await page.screenshot(), contentType: 'image/png' })
    })
  }
}
