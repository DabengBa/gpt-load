import { expect, test, type Locator, type Page } from '@playwright/test'

import { installRequestLogDisplayRoutes, openRequestLogs } from './fixtures/request-log-display.ts'

const usageAggregate = {
  request_count: 2,
  success_count: 2,
  failure_count: 0,
  uncached_input_tokens: 120,
  cache_read_tokens: 0,
  cache_write_5m_tokens: 0,
  cache_write_1h_tokens: 0,
  cache_write_unknown_tokens: 0,
  output_tokens: 42,
  total_tokens: 162,
  estimated_cost_nano_usd: '10400000',
  duration_ms_total: 24800,
  duration_sample_count: 2,
  first_response_ms_total: 16500,
  first_response_sample_count: 2,
  usage_missing_count: 0,
  partial_count: 0,
  unpriced_request_count: 0,
  pricing_partial_count: 0,
}

const feedbackCounts = {
  attempt_count: 5,
  attempt_failure_count: 2,
  normal_attempt_count: 1,
  slow_attempt_count: 2,
  faulty_attempt_count: 1,
}

function distribution(
  dimension: 'model' | 'group' | 'access_key',
  metric: 'requests' | 'tokens' | 'cost',
  identity: Record<string, string | number>,
) {
  return {
    dimension,
    metric,
    items: [
      { request_count: 2, total_tokens: 162, estimated_cost_nano_usd: '10400000', ...identity },
    ],
    other: null,
  }
}

function distributionMetrics(
  dimension: 'model' | 'group' | 'access_key',
  identity: Record<string, string | number>,
) {
  return {
    requests: distribution(dimension, 'requests', identity),
    tokens: distribution(dimension, 'tokens', identity),
    cost: distribution(dimension, 'cost', identity),
  }
}

async function revealFeedbackCountColumns(container: Locator): Promise<void> {
  const found = await container.evaluate((element) => {
    const target = Array.from(element.querySelectorAll('th')).find(
      (header) => header.textContent?.trim() === 'Normal attempts',
    )
    if (!target) return false
    const containerLeft = element.getBoundingClientRect().left
    element.scrollLeft += target.getBoundingClientRect().left - containerLeft - 8
    return true
  })
  expect(found).toBe(true)
}

async function installUsageReport(page: Page, report: unknown): Promise<void> {
  await page.route(
    (url) => url.pathname === '/api/usage',
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'OK', data: report }),
      })
    },
  )
}

const rangeFrom = Date.UTC(2026, 0, 1)
const rangeTo = rangeFrom + 60 * 60 * 1000
const usageReport = {
  range: '1h',
  granularity: 'hour',
  bucket_width_ms: 60 * 60 * 1000,
  from_ms: rangeFrom,
  to_ms: rangeTo,
  observed_at_ms: rangeFrom + 60_000,
  summary: usageAggregate,
  series: [{ ...usageAggregate, bucket_start_ms: rangeFrom, bucket_end_ms: rangeTo }],
  distributions: {
    model: distributionMetrics('model', { model: 'worker' }),
    group: distributionMetrics('group', { group_id: 1 }),
    access_key: distributionMetrics('access_key', { access_key_id: 7 }),
  },
  collection_health: {
    scope: 'current_process',
    dropped_total: 0,
    write_failure_total: 0,
    last_write_failure_at_ms: null,
  },
  breakdown: {
    scope: 'admin',
    rows: [
      {
        ...usageAggregate,
        ...feedbackCounts,
        model: 'worker',
        group_id: 1,
        channel_id: 'openai_compatible',
      },
    ],
    total: usageAggregate,
    attempt_total: feedbackCounts,
    pagination: { page: 1, page_size: 20, total_items: 1, total_pages: 1 },
  },
}

test.describe('provider feedback', () => {
  test('colors final log and each attempt independently and keeps request outcome separate', async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    await installRequestLogDisplayRoutes(page, (items) => {
      const feedback = [
        {
          feedback_status: 'normal',
          feedback_reason: null,
          provider_first_response_ms: 16_000,
          provider_tokens_per_second: 25,
        },
        {
          feedback_status: 'slow',
          feedback_reason: 'output_rate_slow',
          provider_first_response_ms: 500,
          provider_tokens_per_second: 15,
        },
        {
          feedback_status: 'faulty',
          feedback_reason: 'first_response_slow',
          provider_first_response_ms: 30_001,
          provider_tokens_per_second: null,
        },
        {
          feedback_status: null,
          feedback_reason: null,
          provider_first_response_ms: null,
          provider_tokens_per_second: null,
        },
      ]
      return items.map((item, index) => ({ ...item, ...feedback[index] }))
    })
    await openRequestLogs(page)

    await expect(page.getByTestId('request-feedback')).toHaveCount(0)
    await expect(page.locator('.logs-list__feedback-metrics')).toHaveCount(0)

    await expect(page.getByTestId('request-outcome').first()).toContainText('Normal')

    await page.getByRole('button', { name: 'View details' }).first().click()
    await expect(page.locator('.log-detail__summary .status-badge')).toHaveText('Success')
    await expect(page.getByTestId('attempt-feedback')).toHaveCount(0)
    await expect(page.locator('.log-attempt__feedback-metrics')).toHaveCount(0)
    await expect(page.locator('.log-detail__timing--faulty')).toHaveCount(1)
    await page.screenshot({
      path: testInfo.outputPath('provider-feedback-desktop.png'),
      fullPage: true,
    })

    await page.setViewportSize({ width: 390, height: 844 })
    await openRequestLogs(page)
    await expect(page.locator('.ledger-record-list__header')).toBeHidden()
    await expect(page.getByTestId('request-feedback')).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
    await page.screenshot({
      path: testInfo.outputPath('provider-feedback-mobile.png'),
      fullPage: true,
    })
    await page.getByRole('button', { name: 'View details' }).first().click()
    await expect(page.getByTestId('attempt-feedback')).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
    const drawer = page.locator('.app-drawer__content')
    await expect(drawer).toBeVisible()
    const drawerRight = await drawer.evaluate((element) => element.getBoundingClientRect().right)
    const attemptHeaders = page.locator('.log-attempt > header')
    await expect(attemptHeaders).toHaveCount(2)
    for (let index = 0; index < 2; index += 1) {
      const headerWidth = await attemptHeaders.nth(index).evaluate((element) => ({
        client: element.clientWidth,
        scroll: element.scrollWidth,
      }))
      expect(headerWidth.scroll).toBeLessThanOrEqual(headerWidth.client + 1)
      expect(
        await attemptHeaders
          .nth(index)
          .evaluate((element) => element.getBoundingClientRect().right),
      ).toBeLessThanOrEqual(drawerRight + 1)
    }
    await page.locator('.app-drawer__body').evaluate((element) => {
      element.scrollTop = element.scrollHeight
    })
    await page.evaluate(() => {
      if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
    })
    await page.screenshot({
      path: testInfo.outputPath('provider-feedback-mobile-attempts.png'),
    })
  })

  test('shows normal, slow and faulty attempt counts on the breakdown and total rows', async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    await installRequestLogDisplayRoutes(page)
    await installUsageReport(page, usageReport)
    await page.goto('/monitor?tab=usage')
    const table = page.getByRole('table', {
      name: 'Usage and cost breakdown by model and route',
    })
    await expect(table).toBeVisible()
    const headers = table.getByRole('columnheader')
    await expect(headers.filter({ hasText: 'Normal attempts' })).toHaveCount(1)
    await expect(headers.filter({ hasText: 'Slow attempts' })).toHaveCount(1)
    await expect(headers.filter({ hasText: 'Faulty attempts' })).toHaveCount(1)
    const bodyRow = table.locator('tbody tr').first()
    await expect(bodyRow.locator('td').nth(7)).toHaveText('2')
    await revealFeedbackCountColumns(page.locator('.data-table__container').filter({ has: table }))
    await expect(bodyRow.locator('td').nth(8)).toHaveText('1')
    await expect(bodyRow.locator('td').nth(9)).toHaveText('2')
    await expect(bodyRow.locator('td').nth(10)).toHaveText('1')
    const totalRow = table.locator('tfoot tr')
    await expect(totalRow.locator('td').nth(5)).toHaveText('5')
    await expect(totalRow.locator('td').nth(6)).toHaveText('2')
    await revealFeedbackCountColumns(page.locator('.data-table__container').filter({ has: table }))
    await expect(totalRow.locator('td').nth(7)).toHaveText('1')
    await expect(totalRow.locator('td').nth(8)).toHaveText('2')
    await expect(totalRow.locator('td').nth(9)).toHaveText('1')
    const tableContainer = page.locator('.data-table__container').filter({ has: table })
    await revealFeedbackCountColumns(tableContainer)
    await page.screenshot({
      path: testInfo.outputPath('provider-feedback-usage-desktop.png'),
      fullPage: true,
    })

    await page.setViewportSize({ width: 390, height: 844 })
    await expect(tableContainer).toBeVisible()
    await revealFeedbackCountColumns(tableContainer)
    const dimensions = await tableContainer.evaluate((element) => ({
      clientWidth: element.clientWidth,
      scrollWidth: element.scrollWidth,
      right: element.getBoundingClientRect().right,
    }))
    expect(dimensions.scrollWidth).toBeGreaterThan(dimensions.clientWidth)
    expect(dimensions.right).toBeLessThanOrEqual(391)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
    await page.screenshot({
      path: testInfo.outputPath('provider-feedback-usage-mobile.png'),
      fullPage: true,
    })
  })

  test('rejects feedback counts whose sum exceeds the attempt total', async ({ page }) => {
    await installRequestLogDisplayRoutes(page)
    await installUsageReport(page, {
      ...usageReport,
      breakdown: {
        ...usageReport.breakdown,
        rows: [
          {
            ...usageReport.breakdown.rows[0],
            faulty_attempt_count: 3,
          },
        ],
        attempt_total: { ...feedbackCounts, faulty_attempt_count: 3 },
      },
    })
    await page.goto('/monitor?tab=usage')

    await expect(page.getByText('Unable to load usage report.')).toBeVisible()
    await expect(
      page.getByRole('table', { name: 'Usage and cost breakdown by model and route' }),
    ).toHaveCount(0)
  })
})
