import { expect, test } from '@playwright/test'

import {
  PROVIDER_URL,
  envelope,
  installRequestLogDisplayRoutes,
  openRequestLogs,
  requestIDs,
  requestLogDetail,
} from './fixtures/request-log-display.ts'

const longGroupName = 'alpha-production-routing-group-with-a-deliberately-long-name'

test('long route names truncate in the row and stay actionable in the drawer', async ({
  page,
}) => {
  await installRequestLogDisplayRoutes(page)
  await page.route('**/api/groups/options', async (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        code: 0,
        message: 'OK',
        data: [
          {
            id: 1,
            name: longGroupName,
            channel_id: 'openai_compatible',
            connection_type: 'api_key',
            params: { base_url: 'https://alpha.example/v1' },
            provider_url: PROVIDER_URL,
            enabled: true,
            models: ['worker', 'gpt-4o'],
          },
          {
            id: 2,
            name: 'beta',
            channel_id: 'openai_compatible',
            connection_type: 'api_key',
            params: { base_url: 'https://beta.example/v1' },
            provider_url: null,
            enabled: true,
            models: ['gpt-5'],
          },
        ],
      }),
    }),
  )

  await openRequestLogs(page)

  // 行内 compact 身份承载截断与悬浮提示；分组与凭据保持同一行。
  const rowIdentity = page.locator('[data-testid="log-route-identity"]').first()
  await rowIdentity.evaluate((element) => {
    element.style.width = '180px'
  })
  const rowGroup = rowIdentity.locator('[data-testid="log-route-identity__group"]')
  const rowCredential = rowIdentity.locator('[data-testid="log-route-identity__credential"]')
  await expect(rowGroup).toHaveText(longGroupName)
  await expect
    .poll(() => rowGroup.evaluate((element) => element.scrollWidth > element.clientWidth))
    .toBe(true)
  await rowIdentity.hover()
  await expect(page.locator('[data-testid="app-tooltip__content"]')).toContainText(longGroupName)

  const [rowGroupBounds, rowCredentialBounds] = await Promise.all([
    rowGroup.boundingBox(),
    rowCredential.boundingBox(),
  ])
  if (!rowGroupBounds || !rowCredentialBounds) {
    throw new Error('Row route entities must have measurable bounds')
  }
  expect(
    Math.abs(
      rowGroupBounds.y + rowGroupBounds.height / 2 -
        (rowCredentialBounds.y + rowCredentialBounds.height / 2),
    ),
  ).toBeLessThanOrEqual(1)

  // 维护页与供应商外链只在详情抽屉（plain）呈现。抽屉里的分组名取自末次尝试
  // 记录而非 options，因此单独覆盖详情响应给出长名称。
  await page.route(`**/api/logs/${requestIDs.mapped}`, async (route) => {
    const detail = requestLogDetail(requestIDs.mapped)
    detail.attempts = detail.attempts.map((attempt) =>
      attempt.group_id === 1 ? { ...attempt, group_name: longGroupName } : attempt,
    )
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(envelope(detail)),
    })
  })
  await openRequestLogs(page, `?selected_request_id=${requestIDs.mapped}`)
  const identity = page
    .locator('[data-testid="log-detail__section"]')
    .filter({ hasText: 'Upstream execution' })
    .locator('[data-testid="log-route-identity"]')

  const groupName = identity.locator('[data-testid="log-route-identity__group"]')
  const maintenanceLink = identity.locator('[data-testid="log-route-identity__group-link"]')
  const providerLink = identity.locator('[data-testid="log-route-identity__provider"]')

  await expect(groupName).toHaveText(longGroupName)
  await expect(maintenanceLink).toHaveAttribute('href', '/groups/1')
  await expect(maintenanceLink).toHaveAttribute(
    'aria-label',
    `View group ${longGroupName} maintenance page`,
  )
  await expect(providerLink).toHaveAttribute('href', PROVIDER_URL)
  await expect(providerLink).toHaveAttribute('target', '_blank')
  await expect(providerLink).toHaveAttribute('rel', 'noopener noreferrer')
  await expect(providerLink).toHaveAttribute(
    'aria-label',
    `Open provider website ${PROVIDER_URL} in a new tab`,
  )
})
