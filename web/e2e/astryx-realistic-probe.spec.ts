import { expect, test } from '@playwright/test'
import { mockGroupModels } from './fixtures/group-models-mock'

test('group probe submits its saved target and exposes its result log', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  const requestId = 'c9fd0246-b27a-4ef1-b6a6-d52442efcb6d'
  let target: unknown
  await page.route('**/api/model-probe', async (route) => {
    target = route.request().postDataJSON()
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: { results: [{
        group_id: 1, group_name: 'Discovery Group', model: 'model-a', outcome: 'passed',
        reason: null, protocol: 'openai-completions', route_mode: 'native', status_code: 200,
        latency_ms: 12, credential_id: 3, credential_label: 'masked', recovered: false,
        log_id: requestId, tested_at_ms: 1700000000000,
      }] } }),
    })
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Probe', exact: true }).click()
  await expect(page.getByRole('button', { name: 'View log' })).toBeVisible()
  expect(target).toEqual({ targets: [{ group_id: 1, model: 'model-a' }] })
  await expect(page.getByLabel('Access key')).toHaveCount(0)
  await expect(page.getByLabel('Request JSON')).toHaveCount(0)
  await page.getByRole('button', { name: 'View log' }).click()
  await expect(page).toHaveURL(new RegExp('selected_request_id=' + requestId))
})
