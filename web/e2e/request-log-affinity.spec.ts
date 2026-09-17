import { expect, test, type Page } from '@playwright/test'

import {
  ACCESS_KEY,
  ADMIN_KEY,
  AFFINITY_KEY,
  DIFFERENT_AFFINITY_KEY,
  EMPTY_AFFINITY_KEY,
  installRequestLogAffinityRoutes,
  openRequestLogs,
} from './fixtures/request-log-affinity.ts'

const preservedQuery =
  '&from_ms=1700000000000&to_ms=1700003600000&status=success&client_model=gpt-4o&limit=50&selected_request_id=11111111-1111-4111-8111-111111111111&log_cursors=%5B%22old-cursor%22%5D'

function latestLogRequest(routes: { logRequests: URL[] }): URL {
  const request = routes.logRequests.at(-1)
  expect(request).toBeDefined()
  return request as URL
}

async function installHealthRoute(page: Page): Promise<void> {
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        code: 0,
        message: 'OK',
        data: {
          observed_at_ms: 1700000000000,
          version: 'e2e',
          uptime_seconds: 1,
          snapshot_revision: 1,
          stats_window_seconds: 300,
          counts: { credentials: 1, available: 0, cooldown: 1, blacklisted: 0 },
          groups: [
            {
              id: 1,
              name: 'alpha',
              enabled: true,
              counts: { credentials: 1, available: 0, cooldown: 1, blacklisted: 0 },
            },
          ],
          cooldown_credentials: [
            {
              credential_id: 2,
              group_id: 1,
              group_name: 'alpha',
              cooldown_until_ms: 1700000060000,
              failure_count: 1,
              recent_success_count: 0,
              recent_problem_count: 1,
              consecutive_problem_count: 1,
              recovery: {
                automatic: true,
                mode: 'cooldown_expiry',
                at_ms: 1700000060000,
              },
              identity: 'credential-alpha',
              last_failure_category: 'rate_limited',
              last_status_code: 429,
            },
          ],
          blacklisted_credentials: [],
          low_quota_credentials: [],
          expiring_reset_credits: [],
          blocked_access_keys: [],
          request_log: {
            enqueued_total: 0,
            persisted_total: 0,
            dropped_not_running_total: 0,
            dropped_queue_full_total: 0,
            dropped_stopping_total: 0,
            dropped_persist_failed_total: 0,
            dropped_shutdown_total: 0,
            dropped_total: 0,
            write_failure_total: 0,
            access_quota_checkpoint_write_failure_total: 0,
            access_quota_checkpoint_degraded: false,
            retention_delete_failure_total: 0,
            queue_depth: 0,
            queue_capacity: 1,
            last_write_failure_at_ms: null,
            last_access_quota_checkpoint_write_failure_at_ms: null,
            last_retention_failure_at_ms: null,
          },
          debug_capture: {
            enabled: false,
            running: false,
            retention_seconds: 0,
            active: 0,
            completed: 0,
            failed: 0,
            sweep_total: 0,
            removed_total: 0,
            sweep_failure_total: 0,
            error: '',
            last_sweep_at_ms: null,
            last_failure_at_ms: null,
          },
        },
      }),
    })
  })
}

async function expectCollectionSkeletonColumns(page: Page, expectedColumns: number): Promise<void> {
  const skeleton = page.locator('.skeleton-surface--collection')
  await expect(skeleton).toBeVisible()
  await expect(
    skeleton.locator('.skeleton-surface__collection-header .skeleton-block'),
  ).toHaveCount(expectedColumns)
  await expect(
    skeleton.locator('.skeleton-surface__collection-row').first().locator('.skeleton-block'),
  ).toHaveCount(expectedColumns)
}

test.describe('request log affinity filter', () => {
  test('direct logs route loads and is available in the primary navigation', async ({ page }) => {
    for (const principal of ['admin', 'access_key'] as const) {
      const routes = await installRequestLogAffinityRoutes(page, principal)
      await openRequestLogs(page, routes)

      await expect(page).toHaveURL(/\/logs$/u)
      await expect(page.getByRole('link', { name: 'Request logs' }).first()).toBeVisible()
      await expect(page.getByRole('link', { name: 'Request logs' }).first()).toHaveAttribute(
        'aria-current',
        'page',
      )
    }
  })

  test('legacy monitor logs query normalizes to the principal default without log filters', async ({
    page,
  }) => {
    const logQuery =
      'tab=logs&from_ms=1700000000000&to_ms=1700003600000&selected_request_id=11111111-1111-4111-8111-111111111111&log_cursors=%5B%22old-cursor%22%5D&affinity_key=0123456789abcdef%2A%2A%2A%2Afedcba9876543210&status=success&client_model=gpt-4o&upstream_model=legacy-log-model&limit=50'

    for (const [principal, expectedTab] of [
      ['admin', 'health'],
      ['access_key', 'usage'],
    ] as const) {
      const routes = await installRequestLogAffinityRoutes(page, principal)
      await page.goto(`/monitor?${logQuery}`)
      await expect(page).toHaveURL(/\/monitor(?:\?|$)/u)
      await expect.poll(() => new URL(page.url()).searchParams.get('tab')).toBe(expectedTab)

      const normalizedURL = new URL(page.url())
      expect(normalizedURL.pathname).toBe('/monitor')
      expect(normalizedURL.searchParams.get('tab')).toBe(expectedTab)
      for (const key of [
        'from_ms',
        'to_ms',
        'selected_request_id',
        'log_cursors',
        'affinity_key',
        'status',
        'client_model',
        'upstream_model',
        'limit',
      ]) {
        expect(normalizedURL.searchParams.has(key)).toBe(false)
      }
      await expect(page.getByRole('link', { name: 'Request logs' }).first()).toBeVisible()
      await expect(page.locator('.logs-tab')).toHaveCount(0)
      expect(routes.logRequests).toHaveLength(0)
    }
  })

  test('health shortcuts open the standalone logs page with scope filters', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await installHealthRoute(page)

    await page.goto('/monitor?tab=health')
    const groupLogsLink = page.getByRole('link', { name: 'View request logs for alpha' })
    await expect(groupLogsLink).toBeVisible()
    await groupLogsLink.click()
    await expect(page).toHaveURL(/\/logs\?group_id=1$/u)
    await expect(page.locator('.logs-tab')).toBeVisible()
    expect(latestLogRequest(routes).searchParams.get('group_id')).toBe('1')

    await page.goto('/monitor?tab=health')
    const credentialLogsLink = page.getByRole('link', {
      name: 'View request logs for credential credential-alpha',
    })
    await expect(credentialLogsLink).toBeVisible()
    await credentialLogsLink.click()
    await expect(page).toHaveURL(/\/logs\?group_id=1&credential_id=2$/u)
    await expect(page.locator('.logs-tab')).toBeVisible()
    const credentialRequest = latestLogRequest(routes)
    expect(credentialRequest.searchParams.get('group_id')).toBe('1')
    expect(credentialRequest.searchParams.get('credential_id')).toBe('2')
  })

  test('admin key supports click, Enter, Space, exact first page, and preserved filters', async ({
    page,
  }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, preservedQuery)

    const keyButtons = page.getByTestId('logs-affinity-key-filter')
    await expect(keyButtons).toHaveCount(3)
    const columnHeaders = page.locator('.ledger-record-list__header > [role="columnheader"]')
    await expect(columnHeaders).toHaveCount(9)
    await expect(columnHeaders.filter({ hasText: 'Access key' })).toHaveCount(0)
    await expect(keyButtons.first()).toHaveText('…543210')
    await expect(keyButtons.first()).toHaveAttribute(
      'aria-label',
      `Show only logs for affinity scope key ${AFFINITY_KEY}`,
    )
    const emptyAffinityCell = page.locator('.logs-list__affinity-key-cell').filter({ hasText: '—' })
    await expect(emptyAffinityCell).toHaveCount(1)
    await expect(emptyAffinityCell.locator('.logs-list__state--warning')).toHaveCount(0)
    await expect(page.locator('.logs-list__affinity-key-cell code')).toHaveCount(1)
    const timeColumnStyle = await page
      .locator('.logs-list__time')
      .first()
      .evaluate((element) => {
        const style = getComputedStyle(element)
        return { fontFamily: style.fontFamily, fontSize: style.fontSize }
      })
    expect(timeColumnStyle.fontFamily).toContain('ui-monospace')
    expect(timeColumnStyle.fontSize).toBe('10.5px')
    await test.info().attach('time-column-computed-style.json', {
      body: JSON.stringify(timeColumnStyle, null, 2),
      contentType: 'application/json',
    })
    await page.keyboard.press('Escape')
    await expect(page.locator('.app-drawer__overlay')).toBeHidden()

    await keyButtons.first().click()
    const clickedURL = new URL(page.url())
    expect(clickedURL.searchParams.get('affinity_key')).toBe(AFFINITY_KEY)
    expect(clickedURL.searchParams.get('from_ms')).toBe('1700000000000')
    expect(clickedURL.searchParams.get('to_ms')).toBe('1700003600000')
    expect(clickedURL.searchParams.get('status')).toBe('success')
    expect(clickedURL.searchParams.get('client_model')).toBe('gpt-4o')
    expect(clickedURL.searchParams.get('limit')).toBe('50')
    expect(clickedURL.searchParams.has('selected_request_id')).toBe(false)
    expect(clickedURL.searchParams.has('log_cursors')).toBe(false)
    const filteredRequest = latestLogRequest(routes)
    expect(filteredRequest.searchParams.get('affinity_key')).toBe(AFFINITY_KEY)
    expect(filteredRequest.searchParams.has('cursor')).toBe(false)
    await expect(page.getByText(AFFINITY_KEY).first()).toBeVisible()

    await page.getByRole('button', { name: /Remove filter.*Affinity scope key/u }).click()
    await expect(page).not.toHaveURL(/affinity_key=/u)
    expect(latestLogRequest(routes).searchParams.has('affinity_key')).toBe(false)

    await openRequestLogs(page, routes, preservedQuery)
    await page.keyboard.press('Escape')
    await expect(page.locator('.app-drawer__overlay')).toBeHidden()
    await expect(keyButtons.first()).toBeVisible()
    await keyButtons.first().press('Enter')
    await expect(page).toHaveURL(/affinity_key=/u)
    expect(latestLogRequest(routes).searchParams.get('affinity_key')).toBe(AFFINITY_KEY)

    await openRequestLogs(page, routes, preservedQuery)
    await page.keyboard.press('Escape')
    await expect(page.locator('.app-drawer__overlay')).toBeHidden()
    await keyButtons.first().press('Space')
    await expect(page).toHaveURL(/affinity_key=/u)
    expect(latestLogRequest(routes).searchParams.get('affinity_key')).toBe(AFFINITY_KEY)

    await openRequestLogs(
      page,
      routes,
      `&affinity_key=${encodeURIComponent(DIFFERENT_AFFINITY_KEY)}`,
    )
    await expect(page.getByText(DIFFERENT_AFFINITY_KEY).first()).toBeVisible()
  })

  test('access-key list keeps seven aligned tracks through the intermediate breakpoint', async ({
    page,
  }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'access_key')

    for (const width of [1080, 861]) {
      await page.setViewportSize({ width, height: 900 })
      await openRequestLogs(page, routes)
      const metrics = await page.locator('.logs-list').evaluate((list) => {
        const headers = Array.from(
          list.querySelectorAll<HTMLElement>('.ledger-record-list__header > [role="columnheader"]'),
        )
        const cells = Array.from(
          list.querySelectorAll<HTMLElement>('.logs-list__record:first-of-type [role="cell"]'),
        )
        const style = getComputedStyle(list)
        return {
          trackCount: style.gridTemplateColumns.trim().split(/\s+/u).length,
          headerCount: headers.length,
          cellCount: cells.length,
          alignment: headers.map((header, index) => {
            const cell = cells[index]
            if (!cell) return null
            const headerRect = header.getBoundingClientRect()
            const cellRect = cell.getBoundingClientRect()
            return {
              left: Math.abs(headerRect.left - cellRect.left),
              width: Math.abs(headerRect.width - cellRect.width),
            }
          }),
        }
      })

      expect(metrics.trackCount).toBe(7)
      expect(metrics.headerCount).toBe(7)
      expect(metrics.cellCount).toBe(7)
      for (const pair of metrics.alignment) {
        expect(pair).not.toBeNull()
        expect(pair?.left ?? Infinity).toBeLessThanOrEqual(2)
        expect(pair?.width ?? Infinity).toBeLessThanOrEqual(2)
      }
    }

    await page.setViewportSize({ width: 860, height: 900 })
    await openRequestLogs(page, routes)
    const mobileLayout = await page.locator('.logs-list').evaluate((list) => {
      const header = list.querySelector<HTMLElement>('.ledger-record-list__header')
      const record = list.querySelector<HTMLElement>('.logs-list__record')
      const cells = record ? Array.from(record.querySelectorAll('[role="cell"]')) : []
      return {
        headerDisplay: header ? getComputedStyle(header).display : 'missing',
        recordTrackCount: record
          ? getComputedStyle(record).gridTemplateColumns.trim().split(/\s+/u).length
          : 0,
        cellCount: cells.length,
        dataLabels: cells.map((cell) => cell.getAttribute('data-label')),
      }
    })

    expect(mobileLayout.headerDisplay).toBe('none')
    expect(mobileLayout.recordTrackCount).toBe(2)
    expect(mobileLayout.cellCount).toBe(7)
    expect(mobileLayout.dataLabels).toHaveLength(7)
  })

  test('collection skeleton renders nine columns for admin in pending and transition states', async ({
    page,
  }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin', { logDelayMs: 800 })
    await page.goto('/logs')
    await expectCollectionSkeletonColumns(page, 9)
    await expect(page.locator('.logs-list')).toBeVisible()

    await page.locator('.pagination-bar select').selectOption('50')
    await expectCollectionSkeletonColumns(page, 9)
    await expect(page.locator('.logs-list')).toBeVisible()
    expect(routes.logRequests.length).toBeGreaterThanOrEqual(2)
  })

  test('collection skeleton renders seven columns for access-key in pending and transition states', async ({
    page,
  }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'access_key', { logDelayMs: 800 })
    await page.goto('/logs')
    await expectCollectionSkeletonColumns(page, 7)
    await expect(page.locator('.logs-list')).toBeVisible()

    await page.locator('.pagination-bar select').selectOption('50')
    await expectCollectionSkeletonColumns(page, 7)
    await expect(page.locator('.logs-list')).toBeVisible()
    expect(routes.logRequests.length).toBeGreaterThanOrEqual(2)
  })

  test('reapplying unchanged filters refreshes the current log page', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(
      page,
      routes,
      '&from_ms=1700000000000&to_ms=1700003600000&status=success&client_model=gpt-4o&limit=50',
    )
    const initialRequestCount = routes.logRequests.length

    await page.getByRole('button', { name: 'Apply' }).first().click()
    await expect.poll(() => routes.logRequests.length).toBe(initialRequestCount + 1)
  })

  test('empty filtered pages show the filtered empty state', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, `&affinity_key=${encodeURIComponent(EMPTY_AFFINITY_KEY)}`)

    await expect(page.getByText('No logs match the current filters')).toBeVisible()
    expect(latestLogRequest(routes).searchParams.get('affinity_key')).toBe(EMPTY_AFFINITY_KEY)
  })

  test('invalid draft and URL filters do not issue an unfiltered request', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, '')
    const initialRequestCount = routes.logRequests.length

    await page.getByRole('button', { name: /More filters/u }).click()
    const affinityInput = page.locator('#logs-affinity-key')
    await affinityInput.fill('not-a-canonical-key')
    await page.getByRole('button', { name: 'Apply' }).last().click()
    await expect(affinityInput).toHaveAttribute('aria-invalid', 'true')
    await expect(page.getByRole('alert')).toContainText('canonical 16')
    expect(routes.logRequests).toHaveLength(initialRequestCount)

    await page.goto('/logs?affinity_key=not-a-canonical-key')
    await expect(page.getByRole('alert')).toContainText('canonical 16')
    expect(routes.logRequests).toHaveLength(initialRequestCount)
  })

  test('invalid URL filters remain visible and block both apply paths', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')

    for (const affinityKey of ['', 'not-a-canonical-key']) {
      await openRequestLogs(page, routes, `&affinity_key=${encodeURIComponent(affinityKey)}`)
      await expect(page.getByRole('alert')).toContainText('canonical 16')
      const initialRequestCount = routes.logRequests.length

      await page.getByRole('button', { name: 'Apply' }).first().click()
      await expect(page.getByRole('alert')).toContainText('canonical 16')
      expect(routes.logRequests).toHaveLength(initialRequestCount)

      await page.getByRole('button', { name: /More filters/u }).click()
      await page.getByRole('button', { name: 'Apply' }).last().click()
      await expect(page.getByRole('alert')).toContainText('canonical 16')
      expect(routes.logRequests).toHaveLength(initialRequestCount)
    }
  })
  test('access-key principals cannot see or send affinity filters', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'access_key')
    await openRequestLogs(
      page,
      routes,
      `&affinity_key=${encodeURIComponent(AFFINITY_KEY)}&from_ms=1700000000000&to_ms=1700003600000`,
    )

    const columnHeaders = page.locator('.ledger-record-list__header > [role="columnheader"]')
    await expect(columnHeaders).toHaveCount(7)
    await expect(columnHeaders.filter({ hasText: 'Affinity scope key' })).toHaveCount(0)
    await expect(page.getByTestId('logs-affinity-key-filter')).toHaveCount(0)
    await expect(page.locator('#logs-affinity-key')).toHaveCount(0)
    await expect(page.getByRole('button', { name: /Affinity scope key/u })).toHaveCount(0)
    expect(latestLogRequest(routes).searchParams.has('affinity_key')).toBe(false)

    await page.getByRole('button', { name: /More filters/u }).click()
    await expect(page.locator('#logs-affinity-key')).toHaveCount(0)
    expect(page.url()).not.toContain('affinity_key=')
  })

  test('fixture credentials are deterministic and scoped', async ({ page }) => {
    expect(ADMIN_KEY).toBe('e2e-admin-key')
    expect(ACCESS_KEY).toBe('e2e-access-key')
    await installRequestLogAffinityRoutes(page, 'admin')
  })
})

test.describe('request log date range picker', () => {
  const fixedRangeQuery =
    '&from_ms=1700000000000&to_ms=1700003600000&status=success&client_model=gpt-4o&limit=50'

  async function openPicker(page: Page): Promise<void> {
    await page.locator('.app-date-range__trigger').click()
    await expect(page.locator('.app-date-range__calendar')).toBeVisible()
  }

  // The fixture range starts on 2023-11-14, so the calendar opens on that month.
  function firstMonthDay(page: Page, day: number): ReturnType<Page['locator']> {
    return page
      .locator('.app-date-range__calendar-month')
      .first()
      .locator('.app-date-range__calendar-day')
      .filter({ hasText: String(day) })
      .first()
  }

  test('shows the applied from and to range in the trigger on load', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, fixedRangeQuery)

    const trigger = page.locator('.app-date-range__trigger')
    await expect(trigger).toBeVisible()
    // The trigger mirrors the applied draft from/to (local time, "YYYY-MM-DD HH:MM:SS").
    const triggerText = (await trigger.textContent()) ?? ''
    expect(triggerText).toContain('2023-11-')
    expect(triggerText).toContain('→')
  })

  test('selecting a full calendar range updates the draft and Apply submits the query', async ({
    page,
  }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, fixedRangeQuery)
    const initialRequestCount = routes.logRequests.length

    await openPicker(page)

    const startCell = firstMonthDay(page, 14)
    await expect(startCell).toBeVisible()
    await startCell.click()

    // Only start selected: the in-picker Apply must be disabled.
    const applyButton = page.locator('.app-date-range__apply')
    await expect(applyButton).toBeDisabled()

    // Complete the range by selecting a later day as the end.
    const endCell = firstMonthDay(page, 16)
    await endCell.click()
    await expect(applyButton).toBeEnabled()

    await applyButton.click()
    await expect(page.locator('.app-date-range__calendar')).toBeHidden()
    await expect.poll(() => routes.logRequests.length).toBe(initialRequestCount + 1)

    const submitted = latestLogRequest(routes).searchParams
    expect(submitted.has('from_ms')).toBe(true)
    expect(submitted.has('to_ms')).toBe(true)
    // The new range spans 2023-11-14..16; both bounds must differ from the fixture.
    expect(submitted.get('from_ms')).not.toBe('1700000000000')
    expect(submitted.get('to_ms')).not.toBe('1700003600000')
  })

  test('editing the time field preserves the selected date', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, fixedRangeQuery)

    await openPicker(page)
    const startCell = firstMonthDay(page, 14)
    await startCell.click()
    await firstMonthDay(page, 16).click()

    // The from time field renders HH:MM:SS segments; type a new hour into the first segment.
    const fromTimeField = page.locator('.app-date-range__fields .app-date-range__time-field').first()
    const hourSegment = fromTimeField.locator('.app-date-range__time-segment').first()
    await hourSegment.click()
    await page.keyboard.type('05')

    const applyButton = page.locator('.app-date-range__apply')
    await expect(applyButton).toBeEnabled()
    await applyButton.click()
    await expect.poll(() => routes.logRequests.length).toBeGreaterThan(0)

    const submitted = latestLogRequest(routes).searchParams
    const fromMs = Number(submitted.get('from_ms'))
    const fromDate = new Date(fromMs)
    const pad = (part: number) => String(part).padStart(2, '0')
    expect(`${pad(fromDate.getHours())}`).toBe('05')
  })

  test('cancel without applying does not submit a new query', async ({ page }) => {
    const routes = await installRequestLogAffinityRoutes(page, 'admin')
    await openRequestLogs(page, routes, fixedRangeQuery)
    const initialRequestCount = routes.logRequests.length

    await openPicker(page)
    const startCell = firstMonthDay(page, 14)
    await startCell.click()
    await firstMonthDay(page, 16).click()

    // Close the popover without applying.
    await page.keyboard.press('Escape')
    await expect(page.locator('.app-date-range__calendar')).toBeHidden()

    expect(routes.logRequests).toHaveLength(initialRequestCount)
    const url = new URL(page.url())
    expect(url.searchParams.get('from_ms')).toBe('1700000000000')
    expect(url.searchParams.get('to_ms')).toBe('1700003600000')
  })
})
