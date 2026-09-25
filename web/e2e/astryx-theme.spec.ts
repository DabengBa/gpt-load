import { expect, test } from '@playwright/test'

// B3: static theme + layer order + single-owner data-theme on the Astryx entry.
// The shell renders two Buttons: one at theme defaults and one with an
// `xstyle` borderRadius override, so computed styles prove the precedence
// chain app.* > astryx-theme > astryx-base.

const ASTRYX_URL = '/astryx.html'
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
  await expect(page.getByRole('button', { name: 'GPT-Load' })).toBeVisible()
}

test('light mode applies the classic canvas without a flash', async ({ page }) => {
  await openAstryx(page, { theme: 'light' })
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
  const main = page.locator('main')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(238, 237, 233)') // --color-canvas #eeede9
})

test('dark mode applies the dark canvas', async ({ page }) => {
  await openAstryx(page, { theme: 'dark' })
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  const main = page.locator('main')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(11, 13, 16)') // --color-canvas dark #0b0d10
})

test('system mode removes data-theme and follows prefers-color-scheme', async ({
  page,
}) => {
  await openAstryx(page, { theme: 'system', colorScheme: 'dark' })
  await expect(page.locator('html')).not.toHaveAttribute('data-theme', /.+/)
  const main = page.locator('main')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(11, 13, 16)')

  await page.emulateMedia({ colorScheme: 'light' })
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(238, 237, 233)')
})

test('precedence: StyleX override beats theme token, which beats the default', async ({
  page,
}) => {
  await openAstryx(page, { theme: 'light' })
  const themed = page.getByRole('button', { name: 'GPT-Load' })
  const overridden = page.getByRole('button', { name: 'Override' })

  // Theme override: --radius-element is 7px (neutral default is 8px).
  await expect
    .poll(async () =>
      themed.evaluate((el) => getComputedStyle(el).borderRadius),
    )
    .toBe('7px')
  // App layer: xstyle override wins over the theme component default.
  await expect
    .poll(async () =>
      overridden.evaluate((el) => getComputedStyle(el).borderRadius),
    )
    .toBe('2px')
})

test('classic density: compact control height and body text', async ({ page }) => {
  await openAstryx(page, { theme: 'light' })
  const button = page.getByRole('button', { name: 'GPT-Load' })
  const height = await button.evaluate((el) => el.getBoundingClientRect().height)
  // --size-element-md = 34px (classic --control-sm)
  expect(Math.abs(height - 34)).toBeLessThanOrEqual(1)
  const main = page.locator('main')
  const fontSize = await main.evaluate((el) => getComputedStyle(el).fontSize)
  expect(fontSize).toBe('13.5px')
})
