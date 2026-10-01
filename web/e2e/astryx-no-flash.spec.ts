import { expect, test } from '@playwright/test'

// B13 gate #2: no theme flash on reload in light, dark, and system modes
// with theme-bootstrap.js unchanged.
//
// Mechanism under test: the document entry loads /theme-bootstrap.js as a
// synchronous script in <head>. It stamps (or removes) data-theme on
// <html> during initial document parse — before <body> exists and
// therefore before first paint. If theme initialization ever moved into
// app code (a React effect, a module), the attribute would be missing or
// wrong at DOMContentLoaded and users would see one frame of the wrong
// canvas.
//
// Each test installs an init script that captures data-theme inside a
// DOMContentLoaded listener. Init scripts run before any page script, so
// the captured value is whatever theme-bootstrap.js applied — the
// pre-paint state, not the settled post-load state.

const THEME_KEY = 'gpt-load.theme'

type ProbeWindow = Window & { __themeAtParse?: string | null }

async function openAstryxWithProbe(
  page: import('@playwright/test').Page,
  options: { theme?: string; colorScheme?: 'light' | 'dark' } = {},
) {
  await page.emulateMedia({ colorScheme: options.colorScheme ?? 'light' })
  await page.addInitScript(
    ([key, theme]) => {
      if (theme) window.localStorage.setItem(key, theme)
      const w = window as ProbeWindow
      document.addEventListener('DOMContentLoaded', () => {
        w.__themeAtParse = document.documentElement.getAttribute('data-theme')
      })
    },
    [THEME_KEY, options.theme ?? ''],
  )
  await page.goto('/login', { waitUntil: 'domcontentloaded' })
  return page.evaluate(() => (window as ProbeWindow).__themeAtParse)
}

test('light theme is applied during initial document parse', async ({ page }) => {
  const atParse = await openAstryxWithProbe(page, { theme: 'light' })
  expect(atParse).toBe('light')
  const main = page.getByTestId('astryx-shell')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(238, 237, 233)')
})

test('dark theme is applied during initial document parse', async ({ page }) => {
  const atParse = await openAstryxWithProbe(page, { theme: 'dark' })
  expect(atParse).toBe('dark')
  const main = page.getByTestId('astryx-shell')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(11, 13, 16)')
})

test('system mode leaves data-theme unset during initial parse', async ({ page }) => {
  const atParse = await openAstryxWithProbe(page, {
    theme: 'system',
    colorScheme: 'dark',
  })
  expect(atParse).toBeNull()
  const main = page.getByTestId('astryx-shell')
  await expect
    .poll(async () => main.evaluate((el) => getComputedStyle(el).backgroundColor))
    .toBe('rgb(11, 13, 16)')
})

test('theme bootstrap ships as a synchronous head script', async ({ page }) => {
  await page.goto('/login', { waitUntil: 'domcontentloaded' })
  // The entry module script lives in <body>; theme-bootstrap.js must be a
  // classic parser-blocking script in <head> — async/defer/module delivery
  // would let it race first paint.
  const bootstrap = await page.evaluate(() => {
    const el = [...document.head.querySelectorAll('script')].find((script) =>
      script.src.endsWith('/theme-bootstrap.js'),
    )
    if (!el) return null
    return { async: el.async, defer: el.defer, type: el.type }
  })
  expect(bootstrap).toEqual({ async: false, defer: false, type: '' })
})
