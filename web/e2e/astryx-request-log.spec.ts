import { expect, test, type Page } from '@playwright/test'

import {
  PROVIDER_URL,
  installRequestLogDisplayRoutes,
  installRequestLogTableRoutes,
  requestIDs,
  type RequestLogDisplayRoutes,
  type RequestLogTableRoutes,
} from './fixtures/request-log-display'

// Task E: full request-log parity sweep on the Astryx entry. These cases pin
// the classic LogsTab semantics that the spike specs did not cover — row
// display affordances, route-identity actions, advanced filters, cursor
// pagination, stale-preserving errors, access-key scoping, and the legacy
// /monitor?tab=logs normalization.

test.setTimeout(90_000)

async function openLogs(page: Page, query = ''): Promise<void> {
  // `commit` returns once response headers arrive — the first cold vite
  // transform of the astryx graph outlasts the default navigation timeout.
  await page.goto(`/logs${query}`, { waitUntil: 'commit' })
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({
    timeout: 60_000,
  })
  await expect(page.getByRole('heading', { name: 'Request logs' })).toBeVisible()
}

const records = (page: Page) => page.getByTestId('logs-list__record')
const applyButton = (page: Page) => page.getByRole('button', { name: 'Apply', exact: true }).first()

function latestLogRequest(routes: RequestLogDisplayRoutes | RequestLogTableRoutes): URL {
  const request = routes.logRequests.at(-1)
  expect(request).toBeDefined()
  return request as URL
}

test('list renders full rows with headers, summary, and local timestamps', async ({ page }) => {
  await installRequestLogDisplayRoutes(page)
  await openLogs(page)

  const list = page.getByTestId('logs-list')
  await expect(records(page)).toHaveCount(4)
  // Admin surface keeps all nine tracks.
  await expect(list.getByRole('columnheader')).toHaveCount(9)
  await expect(page.getByTestId('logs-result-summary')).toHaveText(
    '4 logs on this page · Newest first',
  )

  const time = records(page).first().getByTestId('logs-list__time').locator('time')
  await expect(time).toHaveAttribute('datetime', /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)
  await expect(time).toHaveAttribute('title', /\d{4}-\d{2}-\d{2}/)
  // Day separators only appear on rows that start a new local day.
  await expect(records(page).first().getByTestId('logs-list__time')).toContainText(/\d{4}/)
})

test('route identity: in-place filters plus drawer maintenance and provider links', async ({
  page,
}) => {
  const routes = await installRequestLogDisplayRoutes(page)
  await openLogs(page)

  const groupFilters = page.locator('button[data-testid="log-route-identity__group"]')
  await expect(groupFilters).toHaveCount(3)
  await expect(groupFilters.nth(0)).toHaveAttribute('aria-label', 'Show only logs from group alpha')
  await expect(groupFilters.nth(1)).toHaveAttribute('aria-label', 'Show only logs from group beta')

  // In-place narrowing writes the URL and drives the next request.
  const before = routes.logRequests.length
  await groupFilters.first().click()
  await expect(page).toHaveURL(/group_id=1/u)
  expect(latestLogRequest(routes).searchParams.get('group_id')).toBe('1')
  expect(routes.logRequests.length).toBeGreaterThan(before)

  // Detail drawer: maintenance link for a live group, provider link opens a
  // new tab, deleted-attempt group renders the "Deleted · #id" text only.
  await openLogs(page, `?selected_request_id=${requestIDs.mapped}`)
  const dialog = page.getByRole('dialog', { name: 'Request log details' })
  await expect(dialog).toBeVisible()
  const upstream = dialog
    .locator('[data-testid="log-detail__section"]')
    .filter({ hasText: 'Upstream execution' })

  const groupLink = upstream.locator('[data-testid="log-route-identity__group-link"]')
  await expect(groupLink).toHaveAttribute('href', '/groups/1')
  await expect(groupLink).toHaveAttribute('aria-label', 'View group alpha maintenance page')

  const providerLink = upstream.locator(
    `[data-testid="log-route-identity__provider"][href="${PROVIDER_URL}"]`,
  )
  await expect(providerLink).toHaveAttribute('target', '_blank')
  await expect(providerLink).toHaveAttribute('rel', /noopener/u)
  await expect(providerLink).toHaveAttribute(
    'aria-label',
    `Open provider website ${PROVIDER_URL} in a new tab`,
  )

  const deletedAttempt = page.locator('[data-testid="log-attempt"]').first()
  await expect(deletedAttempt.locator('[data-testid="log-route-identity__group"]')).toHaveText(
    'Deleted · #99',
  )
  await expect(
    deletedAttempt.locator('[data-testid="log-route-identity__group-link"]'),
  ).toHaveCount(0)
})

test('slow first response tones only the number; model mapping stays inline', async ({ page }) => {
  await installRequestLogDisplayRoutes(page)
  await openLogs(page)

  const slow = page.locator('[data-testid="logs-list__timing"][data-tone="slow"]')
  await expect(slow).toHaveCount(1)
  await expect(slow).toHaveText('16s')
  const timingCell = records(page).first().locator('[role="cell"]').nth(-2)
  await expect(timingCell).toContainText('16s / 24s')
  await expect(records(page).nth(1).locator('[data-tone="slow"]')).toHaveCount(0)

  const mapping = page.getByTestId('logs-list__model-mapping')
  await expect(mapping).toHaveCount(1)
  await expect(mapping).toHaveText('->gpt-5.6-luna')
  await expect(page.getByRole('button', { name: 'View model mapping' })).toHaveCount(0)
})

test('group selector search narrows options and Apply commits group_id', async ({ page }) => {
  const routes = await installRequestLogDisplayRoutes(page)
  await openLogs(page)

  // hasSearch selectors expose a button trigger; the popup's search input
  // owns the combobox role.
  await page.getByRole('button', { name: 'Any-attempt Group' }).click()
  const listbox = page.getByRole('listbox')
  // "Any attempt" entry plus the two fixture groups.
  await expect(listbox.getByRole('option')).toHaveCount(3)

  const search = page.getByRole('combobox', { name: 'Search options' })
  await expect(search).toBeFocused()
  await search.fill('bet')
  const narrowed = listbox.getByRole('option')
  await expect(narrowed).toHaveCount(1)
  await expect(narrowed.first()).toHaveText('beta')
  await narrowed.first().click()

  await applyButton(page).click()
  await expect(page).toHaveURL(/group_id=2/u)
  expect(latestLogRequest(routes).searchParams.get('group_id')).toBe('2')
})

test('client model selector searches the union of group models', async ({ page }) => {
  const routes = await installRequestLogDisplayRoutes(page)
  await openLogs(page)

  await page.getByRole('button', { name: 'Client model' }).click()
  const listbox = page.getByRole('listbox')
  await expect(listbox.getByRole('option')).toHaveCount(4)

  const search = page.getByRole('combobox', { name: 'Search options' })
  await search.fill('zzz')
  await expect(listbox.getByRole('option')).toHaveCount(0)
  await expect(page.getByText('No matches').last()).toBeVisible()

  await search.fill('work')
  const narrowed = listbox.getByRole('option')
  await expect(narrowed).toHaveCount(1)
  await narrowed.first().click()

  await applyButton(page).click()
  await expect(page).toHaveURL(/client_model=worker/u)
  expect(latestLogRequest(routes).searchParams.get('client_model')).toBe('worker')
})

test('applied chips mirror advanced filters and removal issues a fresh query', async ({
  page,
}) => {
  const routes = await installRequestLogDisplayRoutes(page)
  await openLogs(page, '?upstream_model=gpt-5.6-luna')

  const chip = page.getByRole('button', {
    name: 'Remove filter Attempt upstream model gpt-5.6-luna',
  })
  await expect(chip).toBeVisible()
  // The More-filters badge counts advanced dimensions only.
  await expect(page.getByRole('button', { name: 'More filters' })).toContainText('1')

  const before = routes.logRequests.length
  await chip.click()
  await expect(page).not.toHaveURL(/upstream_model=/u)
  await expect.poll(() => routes.logRequests.length).toBe(before + 1)
  expect(latestLogRequest(routes).searchParams.has('upstream_model')).toBe(false)
})

test('advanced drawer opens via panel=filters and validates like classic', async ({ page }) => {
  const routes = await installRequestLogDisplayRoutes(page)
  await openLogs(page)

  await page.getByRole('button', { name: 'More filters' }).click()
  await expect(page).toHaveURL(/panel=filters/u)
  const drawer = page.getByRole('dialog', { name: 'More filters' })
  await expect(drawer).toBeVisible()
  await expect(drawer.getByText('Retry attempts', { exact: true })).toBeVisible()
  await expect(drawer.getByRole('combobox', { name: 'Access key' })).toBeVisible()

  // Invalid affinity input blocks the apply path — no request leaves.
  const before = routes.logRequests.length
  await drawer.getByRole('textbox', { name: 'Affinity scope key' }).fill('not-a-canonical-key')
  await drawer.getByRole('button', { name: 'Apply', exact: true }).click()
  await expect(page.locator('#logs-filter-error')).toContainText('canonical 16')
  await page.waitForTimeout(300)
  expect(routes.logRequests).toHaveLength(before)
})

test('cursor pagination: next pushes log_cursors, previous pops, page size commits limit', async ({
  page,
}) => {
  const routes = await installRequestLogTableRoutes(page)
  await openLogs(page, '?limit=20')

  await expect(records(page)).toHaveCount(4)
  const next = page.getByRole('button', { name: 'Next' })
  await expect(next).toBeEnabled()

  await next.click()
  await expect(page).toHaveURL(/log_cursors=/u)
  await expect.poll(() => latestLogRequest(routes).searchParams.get('cursor')).toBe('p2')
  await expect(page.getByText('page-two-0').first()).toBeVisible()
  // The page indicator advances past page 1.
  await expect(page.getByRole('navigation', { name: 'Pagination' })).toContainText('2')

  await page.getByRole('button', { name: 'Previous' }).click()
  await expect(page).not.toHaveURL(/log_cursors=/u)
  await expect
    .poll(() => latestLogRequest(routes).searchParams.get('cursor'))
    .toBeNull()

  const requestCount = routes.logRequests.length
  await page.getByRole('combobox', { name: 'Items per page' }).click()
  await page.getByRole('option', { name: '50 / page' }).click()
  await expect(page).toHaveURL(/limit=50/u)
  await expect.poll(() => routes.logRequests.length).toBe(requestCount + 1)
  expect(latestLogRequest(routes).searchParams.get('limit')).toBe('50')
})

test('a failed page transition restores the previous URL and preserves stale rows', async ({
  page,
}) => {
  const routes = await installRequestLogTableRoutes(page)
  await openLogs(page)
  await expect(records(page)).toHaveCount(4)

  routes.failNextList()
  await page.getByRole('button', { name: 'Next' }).click()

  // Classic semantics: the failed transition rolls the URL back to its origin
  // while the previous page's rows stay on screen. The stale banner is
  // transient — the restored query returns to its cached success state, so
  // the durable assertions are the restored URL and preserved rows.
  await expect(page).not.toHaveURL(/log_cursors=/u)
  await expect(records(page)).toHaveCount(4)

  // Retrying the transition is clicking Next again.
  await page.getByRole('button', { name: 'Next' }).click()
  await expect(page).toHaveURL(/log_cursors=/u)
  await expect(page.getByText('page-two-0').first()).toBeVisible()
})

test('initial load failure renders the error state and retry recovers', async ({ page }) => {
  const routes = await installRequestLogTableRoutes(page)
  routes.failNextList()
  await page.goto('/logs', { waitUntil: 'commit' })
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({ timeout: 60_000 })

  await expect(page.getByText('Unable to load request logs.')).toBeVisible()
  await page.getByRole('button', { name: 'Retry' }).click()
  await expect(records(page)).toHaveCount(4)
})

test('reapplying unchanged filters refetches instead of navigating', async ({ page }) => {
  const routes = await installRequestLogTableRoutes(page)
  await openLogs(page, '?from_ms=1700000000000&to_ms=1700003600000&status=success&limit=50')
  const initial = routes.logRequests.length

  await applyButton(page).click()
  await expect.poll(() => routes.logRequests.length).toBe(initial + 1)
  // Same signature → refresh path: URL must stay identical (still carries the
  // original params, nothing extra serialized).
  expect(latestLogRequest(routes).searchParams.get('status')).toBe('success')
})

test('invalid affinity key in the URL blocks the list and every apply path', async ({ page }) => {
  const routes = await installRequestLogTableRoutes(page)
  await page.goto('/logs?affinity_key=not-a-canonical-key', { waitUntil: 'commit' })
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({ timeout: 60_000 })

  await expect(page.locator('#logs-filter-error')).toContainText('canonical 16')
  expect(routes.logRequests).toHaveLength(0)

  await applyButton(page).click()
  await page.waitForTimeout(300)
  expect(routes.logRequests).toHaveLength(0)

  await page.getByRole('button', { name: 'More filters' }).click()
  await page.getByRole('dialog').getByRole('button', { name: 'Apply', exact: true }).click()
  await page.waitForTimeout(300)
  expect(routes.logRequests).toHaveLength(0)
})

test('access-key principal: scoped filters, seven tracks, no admin-only affordances', async ({
  page,
}) => {
  const routes = await installRequestLogTableRoutes(page, 'access_key')
  await openLogs(page, `?affinity_key=${encodeURIComponent('0123456789abcdef****fedcba9876543210')}`)

  await expect(records(page)).toHaveCount(4)
  await expect(page.getByTestId('logs-list').getByRole('columnheader')).toHaveCount(7)
  await expect(page.getByTestId('logs-affinity-key-filter')).toHaveCount(0)
  // The URL's affinity param is stripped from the outgoing request entirely.
  expect(latestLogRequest(routes).searchParams.has('affinity_key')).toBe(false)

  // No group selector; client model degrades to free text.
  await expect(page.getByRole('button', { name: 'Any-attempt Group' })).toHaveCount(0)
  await expect(page.getByRole('textbox', { name: 'Client model' })).toBeVisible()

  await page.getByRole('button', { name: 'More filters' }).click()
  const drawer = page.getByRole('dialog', { name: 'More filters' })
  await expect(drawer).toBeVisible()
  await expect(drawer.getByRole('combobox', { name: 'Access key' })).toHaveCount(0)
  await expect(
    drawer.getByRole('textbox', { name: 'Affinity scope key' }),
  ).toHaveCount(0)
  await expect(drawer.getByText('Retry attempts', { exact: true })).toHaveCount(0)
})

test('filter transitions swap rows for the skeleton instead of stale data', async ({ page }) => {
  const routes = await installRequestLogTableRoutes(page)
  await openLogs(page)
  await expect(records(page)).toHaveCount(4)

  routes.delayNextList(800)
  await page.getByRole('button', { name: 'Next' }).click()

  await expect(page.getByRole('status', { name: 'Loading request logs…' })).toBeVisible()
  await expect(records(page)).toHaveCount(0)
  await expect(page.getByText('page-two-0').first()).toBeVisible({ timeout: 10_000 })
})

test('legacy /monitor?tab=logs query normalizes to the principal default tab', async ({
  page,
}) => {
  const routes = await installRequestLogTableRoutes(page)
  await page.goto(
    '/monitor?tab=logs&from_ms=1700000000000&selected_request_id=11111111-1111-4111-8111-111111111111&status=success',
    { waitUntil: 'commit' },
  )
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({ timeout: 60_000 })

  await expect.poll(() => new URL(page.url()).searchParams.get('tab')).toBe('health')
  const url = new URL(page.url())
  for (const key of ['from_ms', 'selected_request_id', 'status']) {
    expect(url.searchParams.has(key)).toBe(false)
  }
  await expect(page.getByTestId('logs-tab')).toHaveCount(0)
  expect(routes.logRequests).toHaveLength(0)
})

test('narrow viewports keep results usable in the card layout', async ({ page }) => {
  await page.setViewportSize({ width: 480, height: 800 })
  await installRequestLogDisplayRoutes(page)
  await openLogs(page)

  await expect(records(page)).toHaveCount(4)
  // Card mode exposes per-cell labels that desktop columns hide.
  await expect(
    records(page).first().getByText('First / total', { exact: true }),
  ).toBeVisible()
  await expect(
    records(page).first().getByText('Model / protocol', { exact: true }),
  ).toBeVisible()
})
