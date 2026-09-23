import { expect, test } from '@playwright/test'

import {
  installRequestLogAffinityRoutes,
  AFFINITY_KEY,
  openRequestLogs as openAffinityLogs,
} from './fixtures/request-log-affinity.ts'
import { installRequestLogDisplayRoutes, openRequestLogs } from './fixtures/request-log-display.ts'

async function openAffinity(page: Parameters<typeof installRequestLogAffinityRoutes>[0]) {
  const routes = await installRequestLogAffinityRoutes(page)
  await openAffinityLogs(page, routes)
}

test.describe('request log page polish', () => {
  test('shows complete local timestamps and current-page newest-first summary', async ({
    page,
  }) => {
    await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    const timestamp = page.locator('.logs-list__time time').first()
    await expect(timestamp).toHaveAttribute('datetime', /^2023-/u)
    await expect(timestamp).toHaveText(/\b2023\b/u)
    await expect(page.getByTestId('logs-result-summary')).toHaveText(
      '4 logs on this page · Newest first',
    )
    await expect(page.getByRole('columnheader', { name: 'Time, newest first' })).toBeVisible()
  })

  test('exposes the complete redacted affinity scope key for identification and copying', async ({
    page,
    context,
  }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write'])
    await openAffinity(page)

    await expect(page.getByRole('columnheader', { name: 'Affinity scope key' })).toBeVisible()
    const keyFilter = page.getByTestId('logs-affinity-key-filter').first()
    await expect(keyFilter).toHaveAttribute(
      'aria-label',
      `Show only logs for affinity scope key ${AFFINITY_KEY}`,
    )
    await expect(keyFilter).toHaveAttribute('title', AFFINITY_KEY)

    const copy = page.getByRole('button', { name: 'Copy full redacted affinity scope key' }).first()
    await copy.click()
    await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(AFFINITY_KEY)
  })

  test('labels model, protocol, response status, token usage, timing, and details', async ({
    page,
  }) => {
    await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    await expect(page.locator('.logs-list__model').first()).toContainText('worker')
    await expect(page.locator('.logs-list__model-mapping')).toContainText('gpt-5.6-luna')
    await expect(page.locator('.logs-list__protocol-line').first()).toContainText(
      'openai-completions',
    )
    await expect(page.getByRole('columnheader', { name: 'Response', exact: true })).toBeVisible()
    await expect(page.locator('.logs-list__response-meta').first()).toContainText('HTTP 200')
    await expect(
      page.getByRole('columnheader', {
        name: 'Tokens (input / output; cache hit rate explained on focus)',
      }),
    ).toBeVisible()
    await expect(page.locator('.logs-list__token-values').first()).toContainText('120')
    await expect(page.locator('.logs-list__token-values').first()).toContainText('42')
    await expect(
      page.getByRole('columnheader', {
        name: 'First response / total duration; hover for output rate',
      }),
    ).toBeVisible()
    await expect(page.locator('.logs-list__timing--slow')).toHaveText('16s')
    await expect(page.locator('.logs-list__record').first()).toContainText('16s / 24s')
    await expect(page.getByRole('button', { name: 'View details' }).first()).toBeVisible()
  })

  test('keeps results in card layout on narrow screens', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    await expect(page.getByTestId('logs-result-summary')).toBeVisible()
    await expect(page.locator('.logs-list__record').first()).toBeVisible()
    await expect(page.locator('.ledger-record-list__header')).toBeHidden()
    await expect(
      page.locator('.logs-list__record').first().getByRole('button', { name: 'View details' }),
    ).toBeVisible()
  })
})
