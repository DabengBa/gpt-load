import { expect, test } from '@playwright/test'

import { fingerprintLogID, mockGroupFingerprint } from './fixtures/group-fingerprint-mock'

// U003: the model row offers fingerprint detection next to the liveness probe,
// the dialog states the closed-set nature of the result, and a detection
// failure stays explicit instead of inventing an attribution.

test.setTimeout(120_000)

test('fingerprint detection shows the attribution evidence and links its log', async ({ page }) => {
  const mock = await mockGroupFingerprint(page)
  await page.goto('/groups/1?tab=models', { waitUntil: 'load' })

  const button = page.getByRole('button', { name: 'Fingerprint', exact: true })
  await expect(button).toBeVisible({ timeout: 60_000 })
  await expect(button).toBeEnabled()
  await button.click()

  expect(mock.calls).toEqual([{ group_id: 1, model: 'gpt-5.4' }])

  const dialog = page.getByRole('dialog', { name: 'Model fingerprint' })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByText('gpt-5.4', { exact: true }).first()).toBeVisible()
  await expect(dialog.getByText('91.2%', { exact: true }).first()).toBeVisible()
  await expect(dialog.getByText('GPT-5.6', { exact: true })).toBeVisible()
  await expect(dialog.getByText('8.1%', { exact: true })).toBeVisible()
  await expect(dialog.getByText('Attribution used 3/3 usable outputs')).toBeVisible()
  await expect(dialog.getByText(/closed-set probability/)).toBeVisible()

  await dialog.getByRole('button', { name: 'View log' }).first().click()
  await expect(page).toHaveURL(new RegExp(`selected_request_id=${fingerprintLogID}`))
})

test('a disabled detection service reports the reason and can be retried', async ({ page }) => {
  const mock = await mockGroupFingerprint(page)
  mock.failNext('MODELTRACE_DISABLED')
  await page.goto('/groups/1?tab=models', { waitUntil: 'load' })

  const button = page.getByRole('button', { name: 'Fingerprint', exact: true })
  await expect(button).toBeVisible({ timeout: 60_000 })
  await button.click()

  const dialog = page.getByRole('dialog', { name: 'Model fingerprint' })
  await expect(dialog).toBeVisible()
  await expect(
    dialog.getByText('Fingerprint detection is unavailable; try again later.'),
  ).toBeVisible()

  await dialog.getByRole('button', { name: 'Retry' }).click()
  await expect(dialog.getByText('Attribution used 3/3 usable outputs')).toBeVisible()
  expect(mock.calls).toHaveLength(2)
})

for (const decision of ['partial', 'unscorable', 'not_model'] as const) {
  test(`fingerprint ${decision} preserves challenge logs without inventing confidence`, async ({
    page,
  }) => {
    await mockGroupFingerprint(page)
    const serialized = await import(`./fixtures/${decision}.json`, { with: { type: 'json' } })
    await page.route('**/api/modeltrace-probe', async (route) => {
      const data = structuredClone(serialized.default)
      data.challenges[0].log_id = fingerprintLogID
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data }),
      })
    })
    await page.goto('/groups/1?tab=models')
    await page.getByRole('button', { name: 'Fingerprint', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Model fingerprint' })
    await expect(
      dialog.getByText(
        decision === 'partial'
          ? 'Partial ranking (probability unavailable)'
          : decision === 'unscorable'
            ? 'Not enough valid samples'
            : 'Response does not resemble a model reply',
      ),
    ).toBeVisible()
    await expect(dialog.getByText('Probability', { exact: true })).toHaveCount(0)
    if (decision !== 'partial')
      await expect(dialog.getByText('Predicted model', { exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('button', { name: 'View log' })).toBeVisible()
  })
}

test('partial ranking displays merged model names and labeled scores with ID fallback', async ({
  page,
}) => {
  await mockGroupFingerprint(page)
  const serialized = await import('./fixtures/partial.json', { with: { type: 'json' } })
  await page.route('**/api/modeltrace-probe', async (route) => {
    const data = structuredClone(serialized.default)
    const merged = data.analysis.results.find((item) => item.model === 'gpt-6-astra')
    if (!merged) throw new Error('merged SDK candidate missing')
    data.analysis.prediction = merged.model
    data.analysis.prediction_name = merged.display_name
    data.analysis.results = [merged, { ...data.analysis.results[0], display_name: '' }]
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data }),
    })
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Fingerprint', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Model fingerprint' })
  await expect(dialog.getByText('gpt-6-astra/6.1-sol', { exact: true })).toHaveCount(2)
  await expect(dialog.getByText('Ranking score: -0.805', { exact: true })).toBeVisible()
  await expect(dialog.getByText('qwen3.8-27b', { exact: true })).toBeVisible()
  await expect(dialog.getByText('Probability', { exact: true })).toHaveCount(0)
})
