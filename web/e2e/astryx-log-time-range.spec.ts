import { expect, test, type Page } from '@playwright/test'

import { installRequestLogRangeRoutes } from './fixtures/request-log-display'

// B12 spike(c): log time-range filter on Astryx DateTimeInput.
// Contract pins the classic semantics (log-filters.ts):
//   - URL carries from_ms/to_ms epoch-ms; absent/invalid params fall back to
//     the default window (now-24h .. now+24h) — the request always sends both
//   - editing the fields updates a draft only; Apply writes the URL params
//   - invalid range (from >= to) blocks Apply with a field error
//   - Reset drops the params back to the default window
// The mock honors from_ms/to_ms, so the rendered rows prove the params drive
// the result set — not just that a request fired.

const DAY_MS = 24 * 60 * 60 * 1000

const localDate = (ms: number) => {
  const d = new Date(ms)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}
const localMs = (date: string, time: string) => new Date(`${date}T${time}`).getTime()

async function openLogs(page: Page, locale?: string) {
  const routes = await installRequestLogRangeRoutes(page)
  if (locale !== undefined) {
    await page.addInitScript((l) => {
      window.localStorage.setItem('gpt-load.locale', l)
    }, locale)
  }
  await page.goto('/logs')
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible()
  await expect(page.getByRole('heading', { name: /.+/ })).toBeVisible()
  return routes
}

const detailButtons = (page: Page) => page.getByRole('button', { name: /details|详情|詳細/ })

test('typed range applies to the request, reset restores the default window (en-US)', async ({
  page,
}) => {
  const { logRequests } = await openLogs(page)

  // Default load sends the classic default window (from now-24h to now+24h),
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
  const fromDate = localDate(Date.now() - 8 * DAY_MS)
  const toDate = localDate(Date.now())

  // Typing edits the draft only — no request until Apply.
  await page.getByRole('combobox', { name: 'From' }).fill(fromDate)
  await page.getByLabel('From time').fill('00:00:00')
  await page.getByRole('combobox', { name: 'To' }).fill(toDate)
  await page.getByLabel('To time').fill('23:59:59')
  expect(logRequests.length).toBe(requestCount)

  await page.getByRole('button', { name: 'Apply', exact: true }).click()

  await expect
    .poll(() => logRequests.length, { message: 'apply should issue a request' })
    .toBe(requestCount + 1)
  const applied = logRequests.at(-1)!
  expect(Number(applied.searchParams.get('from_ms'))).toBe(localMs(fromDate, '00:00:00'))
  expect(Number(applied.searchParams.get('to_ms'))).toBe(localMs(toDate, '23:59:59'))
  await expect(page).toHaveURL(/from_ms=/)
  // now-8d .. end of today covers recent + two-days, not the 10-day-old row.
  await expect(detailButtons(page)).toHaveCount(2)
  await expect(page.getByText('old-model')).toBeHidden()

  await page.getByRole('button', { name: 'Reset' }).click()
  await expect(page).not.toHaveURL(/from_ms=/)
  await expect(detailButtons(page)).toHaveCount(1)
})

test('zh-CN renders localized field and DS strings; a preset chip applies the range', async ({
  page,
}) => {
  const { logRequests } = await openLogs(page, 'zh-CN')

  await expect(page.getByRole('combobox', { name: '开始时间' })).toBeVisible()
  await expect(page.getByRole('combobox', { name: '结束时间' })).toBeVisible()
  await expect(page.getByPlaceholder('选择日期').first()).toBeVisible()
  await expect(page.getByRole('button', { name: '打开日历' }).first()).toBeVisible()
  await expect(page.getByRole('button', { name: '应用', exact: true })).toBeVisible()
  await expect(page.getByRole('group', { name: '快捷时间范围' })).toBeVisible()

  const requestCount = logRequests.length
  await page.getByRole('group', { name: '快捷时间范围' }).getByRole('button', { name: '7d' }).click()

  await expect
    .poll(() => logRequests.length, { message: 'preset should apply a request' })
    .toBe(requestCount + 1)
  const applied = logRequests.at(-1)!
  const span = Number(applied.searchParams.get('to_ms')) - Number(applied.searchParams.get('from_ms'))
  expect(span).toBeGreaterThan(6.9 * DAY_MS)
  expect(span).toBeLessThan(7.1 * DAY_MS)
  await expect(detailButtons(page)).toHaveCount(2)
})

test('ja-JP renders localized strings; an inverted range blocks Apply', async ({ page }) => {
  const { logRequests } = await openLogs(page, 'ja-JP')

  await expect(page.getByRole('combobox', { name: '開始時刻' })).toBeVisible()
  await expect(page.getByRole('combobox', { name: '終了時刻' })).toBeVisible()
  await expect(page.getByPlaceholder('日付を選択').first()).toBeVisible()
  await expect(page.getByRole('button', { name: 'カレンダーを開く' }).first()).toBeVisible()

  const requestCount = logRequests.length
  await page.getByRole('combobox', { name: '開始時刻' }).fill(localDate(Date.now()))
  await page.getByLabel('開始時刻の時刻').fill('23:59:59')
  await page.getByRole('combobox', { name: '終了時刻' }).fill(localDate(Date.now() - DAY_MS))
  await page.getByLabel('終了時刻の時刻').fill('00:00:00')

  await page.getByRole('button', { name: '適用', exact: true }).click()
  // The status message renders in the field and is mirrored to the assertive
  // live region — assert the live-region copy to keep the locator unique. Its
  // appearance proves the apply was handled, so the negative checks after it
  // cannot race the request that validation rejected.
  await expect(
    page.getByRole('alert').getByText('終了時刻は開始時刻より後である必要があります。'),
  ).toBeVisible()
  // from > to is invalid: no request, params stay absent.
  expect(logRequests.length).toBe(requestCount)
  await expect(page).not.toHaveURL(/from_ms=/)
})
