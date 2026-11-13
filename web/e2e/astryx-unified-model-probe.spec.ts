import { expect, test } from '@playwright/test'
import { mockGroupModels } from './fixtures/group-models-mock'

const baseResult = {
  group_id: 1,
  group_name: 'Discovery Group',
  model: 'model-a',
  outcome: 'passed',
  reason: null,
  protocol: 'openai-completions',
  route_mode: 'native',
  status_code: 200,
  latency_ms: 438,
  credential_id: 1,
  credential_label: 'credential',
  recovered: false,
  log_id: '123e4567-e89b-42d3-a456-426614174000',
  tested_at_ms: 1700000000000,
}

for (const scenario of [
  {
    name: 'passed',
    result: {},
    conclusion: 'Passed',
    next: 'Verify with an actual client request',
    log: true,
  },
  {
    name: 'failed',
    result: { outcome: 'failed', reason: 'rate_limited', status_code: 429 },
    conclusion: 'Failed',
    next: 'Review the available diagnostic details',
    log: true,
  },
  {
    name: 'inconclusive',
    result: {
      outcome: 'inconclusive',
      reason: 'no_schedulable_credential',
      status_code: null,
      latency_ms: null,
      credential_id: null,
      credential_label: null,
      log_id: null,
    },
    conclusion: 'Inconclusive',
    next: 'Check the group’s schedulable credentials',
    log: false,
  },
  {
    name: 'executed without log',
    result: { log_id: null },
    conclusion: 'Passed',
    next: 'Verify with an actual client request',
    log: false,
  },
  {
    name: 'target unavailable',
    result: { outcome: 'inconclusive', reason: 'target_unavailable', log_id: null },
    conclusion: 'Inconclusive',
    next: 'Check that the group and model exist',
    log: false,
  },
]) {
  test(`single result: ${scenario.name}`, async ({ page }) => {
    await mockGroupModels(page, { models: [{ id: 'model-a' }] })
    await page.route('**/api/model-probe', (route) =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          code: 0,
          message: 'ok',
          data: { results: [{ ...baseResult, ...scenario.result }] },
        }),
      }),
    )
    await page.goto('/groups/1?tab=models')
    await page.getByRole('button', { name: 'Probe', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Model liveness' })
    await expect(dialog.getByRole('heading', { name: scenario.conclusion })).toBeVisible()
    await expect(dialog.getByText(scenario.next, { exact: false })).toBeVisible()
    await expect(dialog.getByText('Adjust group switches by probe result')).toHaveCount(0)
    await expect(dialog.getByText('No recovery change was needed')).toHaveCount(0)
    await expect(
      dialog.getByText('No upstream request was sent, so there is no request ID'),
    ).toHaveCount(0)
    await expect(dialog.getByText('Protocol', { exact: true })).not.toBeVisible()
    await dialog.getByText('Technical details', { exact: true }).click()
    await expect(dialog.getByText('Protocol', { exact: true })).toBeVisible()
    if (scenario.conclusion === 'Passed') {
      await expect(dialog.getByText('No recovery was reported', { exact: false })).toBeVisible()
    }
    if (scenario.log) {
      await expect(dialog.getByText('123e4567-e89b-42d3-a456-426614174000')).toBeVisible()
      await dialog.getByRole('button', { name: 'View log' }).click()
      await expect(page).toHaveURL(/logs.*123e4567-e89b-42d3-a456-426614174000/)
    } else {
      await expect(dialog.getByText('Request log ID unavailable')).toBeVisible()
      await expect(dialog.getByRole('button', { name: 'View log' })).toHaveCount(0)
    }
  })
}

test('saved model offers one group probe rather than an independent request workspace', async ({
  page,
}) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  await page.goto('/groups/1?tab=models')
  await expect(page.getByRole('switch', { name: 'Enable model model-a' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Probe', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Actual request for model-a' })).toHaveCount(0)
})

test('schedule does not offer an independent request workspace', async ({ page }) => {
  await mockGroupModels(page)
  await page.route('**/api/model-route/schedule**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const data = path.endsWith('/detail')
      ? {
          observed_at_ms: 1,
          snapshot_revision: 1,
          external_model: 'worker',
          protocol: 'openai-responses',
          operation: 'responses_create',
          route_requirement: 'any',
          access_key: null,
          routable: false,
          reason_code: 'no_available_group',
          groups: [],
        }
      : {
          items: [
            {
              external_model: 'worker',
              protocol: 'openai-responses',
              operation: 'responses_create',
              candidate_count: 0,
              group_count: 0,
              cooled_candidates: 0,
              blacklisted_candidates: 0,
            },
          ],
        }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data }),
    })
  })
  await page.goto('/schedule?schedule_model=worker')
  await expect(page.getByText('No entries are available.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Actual request', exact: true })).toHaveCount(0)
})

test('disabled group requires confirmation before its probe', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }], enabled: false })
  let calls = 0
  await page.route('**/api/model-probe', async (route) => {
    calls++
    expect(route.request().postDataJSON()).toEqual({ targets: [{ group_id: 1, model: 'model-a' }] })
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        code: 0,
        message: 'ok',
        data: { results: [{ ...baseResult, log_id: null }] },
      }),
    })
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Probe', exact: true }).click()
  const confirm = page.getByRole('alertdialog')
  await expect(confirm.getByText('This group is disabled')).toBeVisible()
  expect(calls).toBe(0)
  await confirm.getByRole('button', { name: 'Probe', exact: true }).click()
  await expect.poll(() => calls).toBe(1)
  const dialog = page.getByRole('dialog', { name: 'Model liveness' })
  await expect(
    dialog.getByText('This group remains disabled and does not serve traffic.'),
  ).toBeVisible()
  await expect(dialog.getByText('Adjust group switches by probe result')).toHaveCount(0)
})

test('single result remains usable on a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 700 })
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  await page.route('**/api/model-probe', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: { results: [baseResult] } }),
    }),
  )
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Probe', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Model liveness' })
  await expect(dialog.getByRole('heading', { name: 'Passed' })).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'View log' })).toBeVisible()
  await expect(dialog.getByText('123e4567-e89b-42d3-a456-426614174000')).toBeVisible()
  expect(await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
  await dialog.getByText('Technical details', { exact: true }).click()
  await expect(dialog.getByText('Protocol', { exact: true })).toBeVisible()
})

test('retry single group probe keeps its selected protocol and shows the new log', async ({
  page,
}) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }], extraChannel: true })
  const requests: unknown[] = []
  await page.route('**/api/model-probe', (route) => {
    requests.push(route.request().postDataJSON())
    const log_id =
      requests.length === 1 ? baseResult.log_id : '223e4567-e89b-42d3-a456-426614174000'
    return route.fulfill({
      json: { code: 0, message: 'ok', data: { results: [{ ...baseResult, log_id }] } },
    })
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('combobox', { name: 'Probe protocol' }).click()
  await page.getByRole('option', { name: 'OpenAI embeddings' }).click()
  await page.getByRole('button', { name: 'Probe', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Model liveness' })
  await expect(dialog.getByText(baseResult.log_id)).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'Probe again' })).toBeVisible()
  await dialog.getByRole('button', { name: 'Probe again' }).click()
  await expect(dialog.getByText('223e4567-e89b-42d3-a456-426614174000')).toBeVisible()
  await expect(dialog.getByText(baseResult.log_id)).toHaveCount(0)
  expect(requests).toEqual([
    { targets: [{ group_id: 1, model: 'model-a', protocol: 'openai-embeddings' }] },
    { targets: [{ group_id: 1, model: 'model-a', protocol: 'openai-embeddings' }] },
  ])
  await dialog.getByRole('button', { name: 'View log' }).click()
  await expect(page).toHaveURL(/logs.*223e4567-e89b-42d3-a456-426614174000/)
})

test('retry without a new log does not reuse the previous log ID', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  let calls = 0
  await page.route('**/api/model-probe', (route) => {
    calls += 1
    return route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: { results: [{ ...baseResult, log_id: calls === 1 ? baseResult.log_id : null }] },
      },
    })
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Probe', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Model liveness' })
  await expect(dialog.getByText(baseResult.log_id)).toBeVisible()
  await dialog.getByRole('button', { name: 'Probe again' }).click()
  await expect(dialog.getByText('Request log ID unavailable')).toBeVisible()
  await expect(dialog.getByText(baseResult.log_id)).toHaveCount(0)
  await expect(dialog.getByRole('button', { name: 'View log' })).toHaveCount(0)
  expect(calls).toBe(2)
})

test('retry disabled group asks again and cancel retains the previous result', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }], enabled: false })
  const requests: unknown[] = []
  await page.route('**/api/model-probe', (route) => {
    requests.push(route.request().postDataJSON())
    const log_id =
      requests.length === 1 ? baseResult.log_id : '223e4567-e89b-42d3-a456-426614174000'
    return route.fulfill({
      json: { code: 0, message: 'ok', data: { results: [{ ...baseResult, log_id }] } },
    })
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Probe', exact: true }).click()
  await page.getByRole('alertdialog').getByRole('button', { name: 'Probe', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Model liveness' })
  await expect(dialog.getByText(baseResult.log_id)).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'Probe again' })).toBeVisible()
  await dialog.getByRole('button', { name: 'Probe again' }).click()
  await page.getByRole('alertdialog').getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog.getByText(baseResult.log_id)).toBeVisible()
  expect(requests).toHaveLength(1)
  await dialog.getByRole('button', { name: 'Probe again' }).click()
  await page.getByRole('alertdialog').getByRole('button', { name: 'Probe', exact: true }).click()
  await expect(dialog.getByText('223e4567-e89b-42d3-a456-426614174000')).toBeVisible()
  expect(requests).toEqual([
    { targets: [{ group_id: 1, model: 'model-a' }] },
    { targets: [{ group_id: 1, model: 'model-a' }] },
  ])
})
