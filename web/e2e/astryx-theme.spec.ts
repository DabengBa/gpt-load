import { expect, test } from '@playwright/test'

// B3: static theme + layer order + single-owner data-theme on the Astryx
// entry. Rendered on /login — a flagged public route — so the checks run
// against the real shell (PublicShell topbar + LoginView).
//
// Precedence is proven two ways: the registered CSS layer order in
// document.styleSheets (reset → astryx-base → astryx-theme → tokens
// → app.*), and the theme token override landing on a real component
// (the preferences IconButton radius: theme 7px, neutral default 8px).

const ASTRYX_URL = '/login'
const THEME_KEY = 'gpt-load.theme'

async function openAstryx(
  page: import('@playwright/test').Page,
  options: { theme?: string; colorScheme?: 'light' | 'dark' } = {},
) {
  await page.emulateMedia({ colorScheme: options.colorScheme ?? 'light' })
  if (options.theme) {
    await page.addInitScript(
      ([key, value]) => window.localStorage.setItem(key, value),
      [THEME_KEY, options.theme],
    )
  }
  await page.goto(ASTRYX_URL)
  await expect(page.getByTestId('astryx-shell')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Sign in to GPT-Load' })).toBeVisible()
}

test('light mode applies the classic canvas without a flash', async ({ page }) => {
  await openAstryx(page, { theme: 'light' })
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
  const main = page.getByTestId('astryx-shell')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(238, 237, 233)') // --color-canvas #eeede9
})

test('dark mode applies the dark canvas', async ({ page }) => {
  await openAstryx(page, { theme: 'dark' })
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  const main = page.getByTestId('astryx-shell')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(11, 13, 16)') // --color-canvas dark #0b0d10
})

test('system mode removes data-theme and follows prefers-color-scheme', async ({ page }) => {
  await openAstryx(page, { theme: 'system', colorScheme: 'dark' })
  await expect(page.locator('html')).not.toHaveAttribute('data-theme', /.+/)
  const main = page.getByTestId('astryx-shell')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(11, 13, 16)')

  await page.emulateMedia({ colorScheme: 'light' })
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(238, 237, 233)')
})

test('precedence: app layers register after the theme layer', async ({ page }) => {
  await openAstryx(page, { theme: 'light' })

  // The emitted stylesheet declares the layer order; app.* (StyleX) must
  // come last so component xstyle overrides beat theme defaults. Both the
  // statement (`@layer a, b;`) and block (`@layer a { }`) forms count.
  const layers = await page.evaluate(() => {
    const names: string[] = []
    for (const sheet of document.styleSheets) {
      try {
        for (const rule of sheet.cssRules) {
          const match = /^@layer\s+([^;{]+)/.exec(rule.cssText)
          if (!match) continue
          names.push(...match[1].split(',').map((name) => name.trim()))
        }
      } catch {
        // cross-origin or virtual sheets are skipped
      }
    }
    return names
  })
  for (const required of ['reset', 'astryx-base', 'astryx-theme', 'tokens']) {
    expect(layers).toContain(required)
  }
  const themeIndex = layers.indexOf('astryx-theme')
  const tokensIndex = layers.indexOf('tokens')
  const firstAppIndex = layers.findIndex((name) => name.startsWith('app'))
  expect(themeIndex).toBeLessThan(tokensIndex)
  expect(firstAppIndex).toBeGreaterThanOrEqual(0)
  expect(firstAppIndex).toBeGreaterThan(tokensIndex)

  // Theme override reaching a component: the preferences IconButton gets
  // --radius-element 7px from gptload theme (neutral default is 8px).
  const trigger = page.locator('.preferences-trigger')
  await expect
    .poll(async () => trigger.evaluate((el) => getComputedStyle(el).borderRadius))
    .toBe('7px')
})

test('classic density: compact control height and body text', async ({ page }) => {
  await openAstryx(page, { theme: 'light' })
  const trigger = page.locator('.preferences-trigger')
  const height = await trigger.evaluate((el) => el.getBoundingClientRect().height)
  // --size-element-md = 34px (classic --control-sm)
  expect(Math.abs(height - 34)).toBeLessThanOrEqual(1)
  const main = page.getByTestId('astryx-shell')
  const fontSize = await main.evaluate((el) => getComputedStyle(el).fontSize)
  expect(fontSize).toBe('13.5px')
})
