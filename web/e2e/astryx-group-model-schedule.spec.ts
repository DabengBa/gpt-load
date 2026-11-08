import { expect, test } from '@playwright/test'
import { mockGroupModels } from './fixtures/group-models-mock'

test('group row edits scheduling and discard restores the saved values', async ({ page }) => {
  const { puts } = await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  await page.goto('/groups/1?tab=models')
  const reasoning = page.getByLabel('Reasoning policy for model-a')
  const enabled = page.getByLabel('Enable model model-a')
  await expect(reasoning).toHaveValue('')
  await expect(enabled).toBeChecked()
  await reasoning.selectOption('high')
  await enabled.uncheck()
  await page.getByRole('button', { name: 'Discard', exact: true }).click()
  await expect(reasoning).toHaveValue('')
  await expect(enabled).toBeChecked()
  await reasoning.selectOption('low')
  await enabled.uncheck()
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  await expect.poll(() => puts.length).toBe(1)
  expect(puts[0]).toMatchObject({
    models: [{ id: 'model-a', reasoning_effort: 'low', enabled: false }],
  })
})

test('failed save refreshes server state without discarding the draft', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  let gets = 0
  const row = {
    id: 'model-a',
    alias: '',
    alias_enabled: false,
    client_model: 'model-a',
    test_alias: 'abc123',
    concurrency_limit: 0,
    weight: null,
    priority: null,
    pricing_status: 'configured',
    entry_id: 'entry1',
    reasoning_effort: '',
    enabled: true,
  }
  await page.route('**/api/groups/1/models?**', async (route) => {
    if (route.request().method() === 'GET') {
      gets++
      await route.fulfill({
        json: { code: 0, message: 'ok', data: { items: [row], total: 1, pending: 0 } },
      })
    } else {
      row.reasoning_effort = 'medium'
      await route.fulfill({ status: 500, json: { code: 'ERROR', message: 'failed', data: null } })
    }
  })
  await page.goto('/groups/1?tab=models')
  const reasoning = page.getByLabel('Reasoning policy for model-a')
  await reasoning.selectOption('high')
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  await expect(page.getByText(/save outcome is uncertain/i)).toBeVisible()
  await expect.poll(() => gets).toBeGreaterThan(1)
  await expect(reasoning).toHaveValue('high')
  await page.getByRole('button', { name: 'Discard draft and review server' }).click()
  await expect(reasoning).toHaveValue('medium')
})

test('schedule controls remain operable at mobile width', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 })
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  await page.goto('/groups/1?tab=models')
  const reasoning = page.getByLabel('Reasoning policy for model-a')
  await reasoning.selectOption('medium')
  await page.getByLabel('Enable model model-a').uncheck()
  await expect(reasoning).toHaveValue('medium')
  await expect(page.getByLabel('Enable model model-a')).not.toBeChecked()
})

test('conflicting remote edits block retry until the server state is reviewed', async ({
  page,
}) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  let puts = 0
  const row = {
    id: 'model-a',
    alias: '',
    alias_enabled: false,
    client_model: 'model-a',
    test_alias: 'abc123',
    concurrency_limit: 0,
    weight: null,
    priority: null,
    pricing_status: 'configured',
    entry_id: 'entry1',
    reasoning_effort: '',
    enabled: true,
  }
  await page.route('**/api/groups/1/models?**', async (route) => {
    if (route.request().method() === 'PUT') {
      puts++
      if (puts === 1) {
        row.reasoning_effort = 'medium'
        return route.fulfill({
          status: 500,
          json: { code: 'ERROR', message: 'failed', data: null },
        })
      }
      expect(route.request().postDataJSON().models[0].reasoning_effort).toBe('high')
      row.reasoning_effort = 'high'
    }
    await route.fulfill({
      json: { code: 0, message: 'ok', data: { items: [row], total: 1, pending: 0 } },
    })
  })
  await page.goto('/groups/1')
  await page.getByLabel('Reasoning policy for model-a').selectOption('high')
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Discard draft and review server' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeDisabled()
  expect(puts).toBe(1)
  await page.getByRole('button', { name: 'Discard draft and review server' }).click()
  await expect(page.getByLabel('Reasoning policy for model-a')).toHaveValue('medium')
})

test('aliases of the same model have distinct accessible scheduling controls', async ({ page }) => {
  await mockGroupModels(page, {
    models: [
      { id: 'model-a', alias: 'first', alias_enabled: true },
      { id: 'model-a', alias: 'second', alias_enabled: true },
      { id: 'model-b', alias: 'first', alias_enabled: true },
    ],
  })
  await page.goto('/groups/1')
  for (const name of ['model-a (first)', 'model-a (second)', 'model-b (first)']) {
    await expect(page.getByLabel(`Reasoning policy for ${name}`, { exact: true })).toHaveCount(1)
    await expect(page.getByLabel(`Enable model ${name}`, { exact: true })).toHaveCount(1)
  }
})

test('unified settings and model edits use a single combined write', async ({ page }) => {
  const { puts } = await mockGroupModels(page, { models: [{ id: 'model-a' }], extraChannel: true })
  const requests: unknown[] = []
  let separateSettingsWrites = 0
  await page.route('**/api/groups/1/configuration', async (route) => {
    requests.push(route.request().postDataJSON())
    await route.fulfill({ status: 500, json: { code: 'ERROR', message: 'failed', data: null } })
  })
  page.on('request', (request) => {
    if (request.method() === 'PUT' && new URL(request.url()).pathname === '/api/groups/1/settings')
      separateSettingsWrites++
  })
  await page.goto('/groups/1')
  await page.getByRole('button', { name: 'Gemini', exact: true }).click()
  await page.getByLabel('Reasoning policy for model-a').selectOption('high')
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  await expect.poll(() => requests.length).toBe(1)
  expect(requests[0]).toMatchObject({
    settings: { channel_id: 'gemini' },
    models: [{ id: 'model-a', reasoning_effort: 'high' }],
  })
  expect(puts).toHaveLength(0)
  expect(separateSettingsWrites).toBe(0)
})

test('reasoning options follow the unsaved target channel', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }], extraChannel: true })
  await page.goto('/groups/1')
  const reasoning = page.getByLabel('Reasoning policy for model-a')
  await expect(reasoning.locator('option[value="none"]')).toHaveCount(1)
  await expect(page.getByText('Probe protocol', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Gemini', exact: true }).click()
  await expect(reasoning.locator('option[value="none"]')).toHaveCount(0)
  await expect(page.getByText('Probe protocol', { exact: true })).toBeVisible()
})

for (const edited of ['reasoning', 'enabled'] as const) {
  test(`recovery retains remote unedited ${edited === 'reasoning' ? 'enabled' : 'reasoning'} on retry`, async ({
    page,
  }) => {
    await mockGroupModels(page, { models: [{ id: 'model-a' }] })
    const row = {
      id: 'model-a',
      alias: '',
      alias_enabled: false,
      client_model: 'model-a',
      test_alias: 'abc123',
      concurrency_limit: 0,
      weight: null,
      priority: null,
      pricing_status: 'configured',
      entry_id: 'entry1',
      reasoning_effort: '',
      enabled: true,
    }
    const bodies: Record<string, unknown>[] = []
    await page.route('**/api/groups/1/models?**', async (route) => {
      if (route.request().method() === 'PUT') {
        bodies.push(route.request().postDataJSON().models[0])
        if (bodies.length === 1) {
          if (edited === 'reasoning') row.enabled = false
          else row.reasoning_effort = 'medium'
          return route.fulfill({
            status: 500,
            json: { code: 'ERROR', message: 'failed', data: null },
          })
        }
      }
      await route.fulfill({
        json: { code: 0, message: 'ok', data: { items: [row], total: 1, pending: 0 } },
      })
    })
    await page.goto('/groups/1')
    if (edited === 'reasoning')
      await page.getByLabel('Reasoning policy for model-a').selectOption('high')
    else await page.getByLabel('Enable model model-a').uncheck()
    await page.getByRole('button', { name: 'Save settings', exact: true }).click()
    await expect(
      page.getByRole('button', { name: 'Discard draft and review server' }),
    ).toBeVisible()
    await page.getByRole('button', { name: 'Save settings', exact: true }).click()
    await expect.poll(() => bodies.length).toBe(2)
    expect(bodies[1]).not.toHaveProperty(edited === 'reasoning' ? 'enabled' : 'reasoning_effort')
    expect(bodies[1][edited === 'reasoning' ? 'reasoning_effort' : 'enabled']).toBe(
      edited === 'reasoning' ? 'high' : false,
    )
  })
}
