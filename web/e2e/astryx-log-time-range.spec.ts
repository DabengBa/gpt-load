import { expect, test, type Page } from '@playwright/test'

import { installRequestLogRangeRoutes } from './fixtures/request-log-display'

// Presets update the draft; Apply commits epoch-ms URL/API filters.
// Contract pins the log-filters semantics:
//   - URL carries from_ms/to_ms epoch-ms; absent/invalid params fall back to
//     the default window (now-24h .. now+24h) — the request always sends both
//   - selecting a preset updates a draft only; Apply writes the URL params
//   - Reset commits the default filter set and serializes the default
//     window explicitly, so the URL keeps concrete from_ms/to_ms
// The mock honors from_ms/to_ms, so the rendered rows prove the params drive
// the result set — not just that a request fired.

const DAY_MS = 24 * 60 * 60 * 1000

// First navigation to /logs pays the cold vite transform for the whole
// astryx module graph; give this spec the same budget as the detail spec.
test.setTimeout(90_000)

async function openLogs(page: Page, locale?: string) {
  const routes = await installRequestLogRangeRoutes(page)
  if (locale !== undefined) {
    await page.addInitScript((l) => {
      window.localStorage.setItem('gpt-load.locale', l)
    }, locale)
  }
  await page.goto('/logs', { waitUntil: 'commit' })
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({
    timeout: 60_000,
  })
  await expect(page.getByRole('heading', { name: /.+/ })).toBeVisible()
  return routes
}

const detailButtons = (page: Page) =>
  page.getByRole('button', { name: /^(View details|查看详情|詳細を表示)$/u })

test('preset applies to the request, reset restores the default window (en-US)', async ({
  page,
}) => {
  const { logRequests } = await openLogs(page)

  // Default load sends the default window (from now-24h to now+24h),
  // so the range-honoring mock returns only the 30-min-old row.
  await expect(detailButtons(page)).toHaveCount(1)
  const initial = logRequests.at(-1)!
  const from0 = Number(initial.searchParams.get('from_ms'))
  const to0 = Number(initial.searchParams.get('to_ms'))
  expect(Number.isSafeInteger(from0)).toBe(true)
  expect(Number.isSafeInteger(to0)).toBe(true)
  expect(to0 - from0).toBeGreaterThan(47 * 60 * 60 * 1000)
  expect(to0 - from0).toBeLessThan(49 * 60 * 60 * 1000)

  const requestCount = logRequests.length
  await expect(page.getByRole('group', { name: 'Quick time ranges' })).toBeVisible()
  await expect(page.getByRole('combobox', { name: 'From', exact: true })).toHaveCount(0)
  await expect(page.getByRole('combobox', { name: 'To', exact: true })).toHaveCount(0)
  await expect(
    page.getByRole('group', { name: 'Quick time ranges' }).getByRole('button'),
  ).toHaveText(['1h', '24h', '3d', '7d', '15d', '30d'])
  await page
    .getByRole('group', { name: 'Quick time ranges' })
    .getByRole('button', { name: '7d' })
    .click()
  expect(logRequests.length).toBe(requestCount)

  await page.getByRole('button', { name: 'Apply', exact: true }).click()

  await expect
    .poll(() => logRequests.length, { message: 'apply should issue a request' })
    .toBe(requestCount + 1)
  const applied = logRequests.at(-1)!
  expect(
    Number(applied.searchParams.get('to_ms')) - Number(applied.searchParams.get('from_ms')),
  ).toBeGreaterThan(6.9 * DAY_MS)
  await expect(page).toHaveURL(/from_ms=/)
  // The 7d window covers recent rows, not the 10-day-old row.
  await expect(detailButtons(page)).toHaveCount(2)
  await expect(page.getByText('old-model')).toBeHidden()

  const appliedUrl = page.url()
  await page.reload()
  await expect(detailButtons(page)).toHaveCount(2)
  expect(page.url()).toBe(appliedUrl)
  expect(logRequests.at(-1)?.searchParams.get('from_ms')).toBe(applied.searchParams.get('from_ms'))
  expect(logRequests.at(-1)?.searchParams.get('to_ms')).toBe(applied.searchParams.get('to_ms'))

  await page.getByRole('button', { name: 'Reset' }).click()
  // Reset commits the default filter set, which serializes
  // explicit from_ms/to_ms/limit — the URL carries the default window rather
  // than dropping the params.
  await expect(page).toHaveURL(/from_ms=/)
  await expect
    .poll(() => logRequests.length, { message: 'reset should issue a request' })
    .toBeGreaterThan(requestCount + 1)
  const reset = logRequests.at(-1)!
  expect(
    Number(reset.searchParams.get('to_ms')) - Number(reset.searchParams.get('from_ms')),
  ).toBeGreaterThan(47 * 60 * 60 * 1000)
  await expect(detailButtons(page)).toHaveCount(1)
})

test('zh-CN renders localized filter strings; a preset applies the range', async ({ page }) => {
  const { logRequests } = await openLogs(page, 'zh-CN')

  await expect(page.getByRole('button', { name: '应用', exact: true })).toBeVisible()
  await expect(page.getByRole('group', { name: '快捷时间范围' })).toBeVisible()

  const requestCount = logRequests.length
  await page
    .getByRole('group', { name: '快捷时间范围' })
    .getByRole('button', { name: '7d' })
    .click()

  // A preset writes the draft only; the request waits for Apply.
  await page.waitForTimeout(300)
  expect(logRequests.length).toBe(requestCount)

  await page.getByRole('button', { name: '应用', exact: true }).click()

  await expect
    .poll(() => logRequests.length, { message: 'apply should issue a request' })
    .toBe(requestCount + 1)
  const applied = logRequests.at(-1)!
  const span =
    Number(applied.searchParams.get('to_ms')) - Number(applied.searchParams.get('from_ms'))
  expect(span).toBeGreaterThan(6.9 * DAY_MS)
  expect(span).toBeLessThan(7.1 * DAY_MS)
  await expect(detailButtons(page)).toHaveCount(2)
})

test('ja-JP renders localized preset filter strings', async ({ page }) => {
  await openLogs(page, 'ja-JP')

  await expect(page.getByRole('group', { name: 'クイック時間範囲' })).toBeVisible()
  await expect(
    page.getByRole('group', { name: 'クイック時間範囲' }).getByRole('button', { name: '7d' }),
  ).toBeVisible()
})

test('all six shortcuts preserve draft-only selection and commit exact durations', async ({
  page,
}) => {
  const { logRequests } = await openLogs(page)
  await expect(detailButtons(page)).toHaveCount(1)
  const quickRanges = page.getByRole('group', { name: 'Quick time ranges' })
  for (const [label, duration] of [
    ['1h', DAY_MS / 24],
    ['24h', DAY_MS],
    ['3d', 3 * DAY_MS],
    ['7d', 7 * DAY_MS],
    ['15d', 15 * DAY_MS],
    ['30d', 30 * DAY_MS],
  ] as const) {
    const previousUrl = page.url()
    const requestCount = logRequests.length
    await quickRanges.getByRole('button', { name: label, exact: true }).click()
    await page.waitForTimeout(100)
    expect(page.url()).toBe(previousUrl)
    expect(logRequests.length).toBe(requestCount)
    await page.getByRole('button', { name: 'Apply', exact: true }).click()
    await expect.poll(() => logRequests.length).toBe(requestCount + 1)
    const request = logRequests.at(-1)!
    expect(
      Number(request.searchParams.get('to_ms')) - Number(request.searchParams.get('from_ms')),
    ).toBe(duration)
    const url = new URL(page.url())
    expect(url.searchParams.get('from_ms')).toBe(request.searchParams.get('from_ms'))
    expect(url.searchParams.get('to_ms')).toBe(request.searchParams.get('to_ms'))
  }
})
