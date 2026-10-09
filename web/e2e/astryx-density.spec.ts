import { expect, test, type Page, type Route } from '@playwright/test'

// B13 gate #6: consolidated compact-density evidence — every metric in the
// migration plan's density table measured on the Astryx frontend with a
// ±1px tolerance, reached only through theme inputs, SizeContext, and
// Table density (no per-call-site xstyle).
//
// Token assertions prove the emitted theme values; rendered assertions
// prove what lands on real components. Three classic metrics have no
// Astryx counterpart and are recorded rather than asserted:
//   - --control-lg 42px: Astryx has no element-xl stop (sm/md/lg only);
//     no migrated surface renders a 42px control.
//   - --setting-control-height 26px: settings inline controls are not part
//     of the Phase 1 surfaces.
//   - --text-label-xs 10.5px: the smallest Astryx label stop is 11.5px.

const THEMED_SCOPE = '[data-astryx-theme]'

async function openLogin(page: Page) {
  await page.goto('/login')
  await expect(page.getByTestId('astryx-shell')).toBeVisible()
}

async function computedVar(page: Page, name: string) {
  return page.evaluate(
    ([scopeSelector, varName]) => {
      const scope = document.querySelector(scopeSelector) ?? document.documentElement
      return getComputedStyle(scope)
        .getPropertyValue(varName as string)
        .trim()
    },
    [THEMED_SCOPE, name],
  )
}

test('theme emits the classic density tokens', async ({ page }) => {
  await openLogin(page)
  const expectations: Array<[string, string]> = [
    // Control stops — classic compact/sm/md (30/34/38)
    ['--size-element-sm', '30px'],
    ['--size-element-md', '34px'],
    ['--size-element-lg', '38px'],
    // Text — classic body/meta/small/label/section/panel tokens
    ['--text-body-size', '13.5px'],
    ['--text-supporting-size', '12px'],
    ['--text-label-size', '11.5px'],
    ['--text-large-size', '16px'],
    ['--text-heading-3-size', '16px'],
    ['--text-heading-2-size', '22px'],
    // Radius — classic tag/control/sheet (6/7/10)
    ['--radius-inner', '6px'],
    ['--radius-element', '7px'],
    ['--radius-container', '10px'],
  ]
  for (const [name, expected] of expectations) {
    expect(await computedVar(page, name), name).toBe(expected)
  }
  // Spacing base: the Astryx spacing unit must stay the classic 4px grid.
  const spacing = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--spacing-1').trim(),
  )
  expect(spacing).toBe('4px')
})

test('rendered shell controls hold compact metrics', async ({ page }) => {
  await openLogin(page)
  // Preferences IconButton is a size-md control: classic --control-sm 34px.
  const trigger = page.locator('.preferences-trigger')
  const height = await trigger.evaluate((el) => el.getBoundingClientRect().height)
  expect(Math.abs(height - 34)).toBeLessThanOrEqual(1)
  // Body text on the shell.
  const main = page.getByTestId('astryx-shell')
  const fontSize = await main.evaluate((el) => getComputedStyle(el).fontSize)
  expect(fontSize).toBe('13.5px')
  // Radius reaching a component: classic --radius-control 7px.
  const radius = await trigger.evaluate((el) => getComputedStyle(el).borderRadius)
  expect(radius).toBe('7px')
})

test('narrow viewport keeps 44px touch targets', async ({ page }) => {
  await page.setViewportSize({ width: 480, height: 800 })
  await openLogin(page)
  const trigger = page.locator('.preferences-trigger')
  const box = await trigger.evaluate((el) => {
    const rect = el.getBoundingClientRect()
    return { width: rect.width, height: rect.height }
  })
  expect(
    Math.min(box.width, box.height),
    'preferences trigger touch target',
  ).toBeGreaterThanOrEqual(43)
})

test('compact table row height on the groups collection', async ({ page }) => {
  // Minimal groups fixture — the row-height measurement only needs the
  // collection to render real rows.
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route: Route) => {
    const path = new URL(route.request().url()).pathname
    const fulfill = (data: unknown) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data }),
      })
    if (path === '/api/auth/session') {
      return fulfill({
        authenticated: true,
        principal_type: 'admin',
      })
    }
    if (path === '/api/groups') {
      return fulfill({
        observed_at_ms: 1_700_000_000_000,
        summary: { total: 3, available: 3, unavailable: 0, disabled: 0 },
        items: [1, 2, 3].map((id) => ({
          id,
          name: `Group ${id}`,

          channel_id: 'openai',
          connection_type: 'api_key',
          params: {},
          provider_url: null,
          status: 'available',
          model_count: 4,
          client_model_count: 2,
          credential_configured: true,
          credential_status: 'available',
        })),
        pagination: { page: 1, page_size: 100, total_items: 3, total_pages: 1 },
      })
    }
    if (path === '/api/channels') {
      return fulfill({ items: [], total: 0 })
    }
    return fulfill({})
  })
  await page.goto('/groups')
  const row = page.locator('tbody tr').first()
  await expect(row).toBeVisible()
  const height = await row.evaluate((el) => el.getBoundingClientRect().height)
  // Compact rows fit their 30px controls without the old balanced-density whitespace.
  expect(height, 'row keeps room for its controls').toBeGreaterThanOrEqual(30)
  expect(height, 'compact collection row').toBeLessThanOrEqual(44)
  const padding = await row.locator('td').nth(1).evaluate((cell) => {
    const style = getComputedStyle(cell)
    return { block: style.paddingBlock, inline: style.paddingInline }
  })
  expect(padding).toEqual({ block: '4px', inline: '8px' })
})
