import { expect, test, type Page } from '@playwright/test'

import { installRequestLogDisplayRoutes, requestIDs } from './fixtures/request-log-display'

// B11 spike(b): the log detail surface on the Astryx entry — the overlay
// primitive decision. Classic AppDrawer is a reka-ui Dialog styled as a
// right side panel; this spec pins the contract the Astryx replacement must
// satisfy regardless of the underlying primitive:
//   - row action opens a modal side panel driven by ?selected_request_id=
//   - focus is trapped while open and returns to the invoking action on close
//   - Escape and scrim click dismiss; the URL param leaves with the panel
//   - deep links open the panel directly
//   - <=520px viewports go full-bleed like the classic drawer

const dialog = (page: Page) => page.getByRole('dialog', { name: 'Request log details' })

for (const width of [390, 1280]) {
  for (const principal of ['admin', 'access_key'] as const) {
    test(`v7 receipt formula ${principal} at ${width}px`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 })
      const receipt = {
        schema_version: 7,
        method: 'unit_rate_sum',
        method_version: 1,
        currency: 'USD',
        pricing_mode: 'standard',
        rule: { channel_id: 'openai_compatible', model_id: 'gpt-5.6-luna' },
        context_threshold_tokens: null,
        total_nano_usd: '16000',
        line_items: [
          {
            code: 'cache_write_1h',
            quantity: '10',
            rate_nano_usd_per_million: '1000000000',
            multiplier: { numerator: '8', denominator: '5' },
            state: 'priced',
            amount_nano_usd: '16000',
          },
        ],
      }
      await installRequestLogDisplayRoutes(
        page,
        (items) =>
          items.map((item) => ({
            ...item,
            usage_state: 'complete',
            cost_state: 'priced',
            pricing_completeness: 'complete',
            pricing_mode: 'standard',
            estimated_cost_nano_usd: '16000',
          })),
        receipt,
        principal,
      )
      await openLogs(page, `?selected_request_id=${requestIDs.mapped}`)
      const panel = dialog(page)
      await expect(panel).toBeVisible()
      if (principal === 'admin') {
        await panel
          .locator('details')
          .filter({ has: page.getByText('Cost calculation', { exact: true }) })
          .locator('summary')
          .click()
        await expect(panel.getByText('Cost calculation', { exact: true })).toBeVisible()
        await expect(panel).toContainText('10 × $1.00/1M × 8/5')
        await expect(panel).toContainText('$0.000016')
        await expect(panel.getByText(/base total|final total|price multiplier/i)).toHaveCount(0)
        await panel.getByText('Cost calculation', { exact: true }).scrollIntoViewIfNeeded()
      } else {
        await expect(panel.getByText('Cost calculation', { exact: true })).toHaveCount(0)
        await expect(panel.getByText('Attempt chain', { exact: true })).toHaveCount(0)
      }
      await panel.screenshot({ path: testInfo.outputPath('log-v7.png') })
    })
  }
}

// First navigation to /logs on a cold vite dev server transforms the whole
// astryx module graph — past the default 30s test budget on this checkout.
test.setTimeout(90_000)

async function openLogs(page: Page, query = ''): Promise<void> {
  await page.goto(`/logs${query}`, { waitUntil: 'commit' })
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({
    timeout: 60_000,
  })
  await expect(page.getByRole('heading', { name: 'Request logs' })).toBeVisible()
}

test.beforeEach(async ({ page }) => {
  await installRequestLogDisplayRoutes(page)
})

test('row action opens the detail panel and deep-links it', async ({ page }) => {
  await openLogs(page)

  const rows = page.getByRole('button', { name: 'View details' })
  await expect(rows).toHaveCount(4)

  await rows.nth(0).click()

  await expect(dialog(page)).toBeVisible()
  await expect(dialog(page).getByText('gpt-5.6-luna').first()).toBeVisible()
  await expect(page).toHaveURL(/selected_request_id=aaaaaaaa-1111-4111-8111-111111111111/)
  // Detail fetch is server-driven, not lifted from the row payload.
  await expect(dialog(page).getByText('Attempt chain', { exact: true })).toBeVisible()
  await expect(dialog(page).getByText('Attempt #2')).toBeVisible()
})

test('focus stays trapped inside the panel', async ({ page }) => {
  await openLogs(page)
  await page.getByRole('button', { name: 'View details' }).nth(0).click()
  const panel = dialog(page)
  await expect(panel).toBeVisible()

  const focusInside = () =>
    // Astryx Dialog renders a native <dialog>; its dialog role is implicit, so
    // [role="dialog"] won't match — walk up to the open dialog element instead.
    page.evaluate(() => document.activeElement?.closest('dialog[open]') !== null)
  // showModal autofocus lands a beat after visibility; wait for it.
  await expect.poll(focusInside).toBe(true)

  // Focus starts inside the overlay and Tab/Shift+Tab cycle within it.
  for (const key of ['Tab', 'Tab', 'Tab', 'Tab', 'Shift+Tab', 'Shift+Tab']) {
    await page.keyboard.press(key)
    expect(await focusInside()).toBe(true)
  }
})

test('Escape closes the panel, drops the param, and returns focus to the row action', async ({
  page,
}) => {
  await openLogs(page)
  const trigger = page.locator(`#log-details-${requestIDs.mapped}`)
  await trigger.click()
  await expect(dialog(page)).toBeVisible()

  await page.keyboard.press('Escape')

  await expect(dialog(page)).toBeHidden()
  await expect(page).not.toHaveURL(/selected_request_id/)
  await expect(trigger).toBeFocused()
})

test('scrim click dismisses the panel', async ({ page }) => {
  await openLogs(page)
  await page.getByRole('button', { name: 'View details' }).nth(0).click()
  await expect(dialog(page)).toBeVisible()

  // The panel hugs the right edge; a click far left lands on the scrim.
  await page.mouse.click(16, 320)

  await expect(dialog(page)).toBeHidden()
  await expect(page).not.toHaveURL(/selected_request_id/)
})

test('?selected_request_id= opens the panel directly', async ({ page }) => {
  await openLogs(page, `?selected_request_id=${requestIDs.plain}`)

  await expect(dialog(page)).toBeVisible()
  await expect(dialog(page).getByText('gpt-4o').first()).toBeVisible()
})

test('narrow viewport renders the panel full-bleed', async ({ page }) => {
  await page.setViewportSize({ width: 480, height: 800 })
  await openLogs(page)
  await page.getByRole('button', { name: 'View details' }).nth(0).click()

  const panel = dialog(page)
  await expect(panel).toBeVisible()
  const box = await panel.boundingBox()
  expect(box).not.toBeNull()
  expect(Math.round(box!.width)).toBe(480)
})
