import { expect, test } from '@playwright/test'

import {
  PROVIDER_URL,
  installRequestLogDisplayRoutes,
  openRequestLogs,
} from './fixtures/request-log-display.ts'

function latestLogRequest(routes: { logRequests: URL[] }): URL {
  const request = routes.logRequests.at(-1)
  expect(request).toBeDefined()
  return request as URL
}

test.describe('request log display', () => {
  test('provider url icon opens the provider website in a new tab', async ({ page }) => {
    await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    const providerLink = page.locator(`.log-route-identity__provider[href="${PROVIDER_URL}"]`)
    // 仅 provider_url 非空的 alpha 分组渲染官网入口，beta 行不出现。
    await expect(providerLink).toHaveCount(1)
    await expect(providerLink).toHaveAttribute('target', '_blank')
    await expect(providerLink).toHaveAttribute('rel', /noopener/u)
    await expect(providerLink).toHaveAttribute(
      'aria-label',
      `Open provider website ${PROVIDER_URL} in a new tab`,
    )
  })

  test('first response over 15s colors only the time number', async ({ page }) => {
    await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    const slowNumber = page.locator('.logs-list__timing--slow')
    await expect(slowNumber).toHaveCount(1)
    await expect(slowNumber).toHaveText('16s')
    // 慢速标记只包首响数字，同一单元格里的总耗时保持常规颜色。
    const timingCell = page
      .locator('.logs-list__record')
      .first()
      .locator('[data-label="First / total"]')
    await expect(timingCell).toContainText('16s / 24s')
    await expect(timingCell.locator('.logs-list__timing--slow')).toHaveCount(1)
    await expect(page.locator('.logs-list__record').nth(1).locator('.logs-list__timing--slow'))
      .toHaveCount(0)
  })

  test('client to upstream model mapping renders inline without the hint icon', async ({
    page,
  }) => {
    await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    const mapping = page.locator('.logs-list__model-mapping')
    await expect(mapping).toHaveCount(1)
    await expect(mapping).toHaveText('->gpt-5.6-luna')
    await expect(
      page.getByRole('button', { name: 'View model mapping' }),
    ).toHaveCount(0)
    // 客户端模型仍可点击收窄，上游模型只是展示文本。
    const controlRow = page.locator('.logs-list__record').nth(1)
    await expect(controlRow.locator('.logs-list__model-mapping')).toHaveCount(0)
  })

  test('group filter search narrows the option list', async ({ page }) => {
    const routes = await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    await page.getByRole('button', { name: 'Any-attempt Group' }).click()
    const searchInput = page.locator('.searchable-select__search')
    await expect(searchInput).toBeFocused()
    await expect(
      page.locator('.searchable-select__content .app-select__item'),
    ).toHaveCount(3)

    await searchInput.fill('bet')
    const narrowed = page.locator('.searchable-select__content .app-select__item')
    await expect(narrowed).toHaveCount(1)
    await expect(narrowed.first()).toHaveText('beta')
    await narrowed.first().click()
    await expect(page.getByRole('button', { name: 'Any-attempt Group' })).toContainText('beta')

    await page.getByRole('button', { name: 'Apply' }).first().click()
    await expect(page).toHaveURL(/group_id=2/u)
    expect(latestLogRequest(routes).searchParams.get('group_id')).toBe('2')
  })

  test('client model filter offers a searchable option list', async ({ page }) => {
    const routes = await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    await page.getByRole('button', { name: 'Client model' }).click()
    const searchInput = page.locator('.searchable-select__search')
    await expect(
      page.locator('.searchable-select__content .app-select__item'),
    ).toHaveCount(4)

    await searchInput.fill('zzz')
    await expect(page.locator('.searchable-select__content .app-select__item')).toHaveCount(0)
    await expect(page.locator('.searchable-select__empty')).toHaveText('No matches')

    await searchInput.fill('work')
    const narrowed = page.locator('.searchable-select__content .app-select__item')
    await expect(narrowed).toHaveCount(1)
    await expect(narrowed.first()).toHaveText('worker')
    await narrowed.first().click()

    await page.getByRole('button', { name: 'Apply' }).first().click()
    await expect(page).toHaveURL(/client_model=worker/u)
    expect(latestLogRequest(routes).searchParams.get('client_model')).toBe('worker')
  })

  test('usage and cost section is collapsed by default in the detail drawer', async ({
    page,
  }) => {
    await installRequestLogDisplayRoutes(page)
    await openRequestLogs(page)

    await page.getByRole('button', { name: 'View details' }).first().click()
    const usage = page.locator('.log-usage-chain')
    await expect(usage).toBeVisible()
    await expect(usage).not.toHaveAttribute('open', '')
    await expect(page.getByText('Usage state')).toBeHidden()

    await usage.locator('summary').click()
    await expect(usage).toHaveAttribute('open', '')
    await expect(page.getByText('Usage state')).toBeVisible()
  })
})
