import { expect, test } from '@playwright/test'

import {
  PROVIDER_URL,
  installRequestLogDisplayRoutes,
  openRequestLogs,
} from './fixtures/request-log-display.ts'

const longGroupName = 'alpha-production-routing-group-with-a-deliberately-long-name'

test('long route names truncate while maintenance and provider actions stay on one line', async ({
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

  const identity = page.locator('.log-route-identity').first()
  await identity.evaluate((element) => {
    element.style.width = '180px'
  })

  const groupName = identity.locator('.log-route-identity__group')
  const maintenanceLink = identity.locator('.log-route-identity__group-link')
  const providerLink = identity.locator('.log-route-identity__provider')

  await expect(groupName).toHaveText(longGroupName)
  await expect
    .poll(() => groupName.evaluate((element) => element.scrollWidth > element.clientWidth))
    .toBe(true)

  const [groupBounds, maintenanceBounds, providerBounds, identityBounds] = await Promise.all([
    groupName.boundingBox(),
    maintenanceLink.boundingBox(),
    providerLink.boundingBox(),
    identity.boundingBox(),
  ])
  if (!groupBounds || !maintenanceBounds || !providerBounds || !identityBounds) {
    throw new Error('Route name, actions, and identity must have measurable bounds')
  }
  const groupCenter = groupBounds.y + groupBounds.height / 2
  expect(
    Math.abs(groupCenter - (maintenanceBounds.y + maintenanceBounds.height / 2)),
  ).toBeLessThanOrEqual(1)
  expect(
    Math.abs(groupCenter - (providerBounds.y + providerBounds.height / 2)),
  ).toBeLessThanOrEqual(1)
  expect(maintenanceBounds.x).toBeGreaterThanOrEqual(identityBounds.x)
  expect(providerBounds.x + providerBounds.width).toBeLessThanOrEqual(
    identityBounds.x + identityBounds.width,
  )

  await identity.hover()
  await expect(page.locator('.app-tooltip__content')).toContainText(longGroupName)

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
