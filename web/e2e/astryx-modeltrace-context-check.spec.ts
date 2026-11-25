import { expect, test } from '@playwright/test'

import {
  fingerprintLogID,
  fingerprintSuccess,
  mockGroupFingerprint,
} from './fixtures/group-fingerprint-mock'

// The context-fidelity check is a diagnostic on the same route and credential.
// It never confirms an injection: the dialog uses hedged labels and shows
// token counts and the diagnostic log link when available.

test.setTimeout(120_000)

test('context fidelity shows status, counts and a log link when available', async ({ page }) => {
  await mockGroupFingerprint(page)
  await page.route('**/api/modeltrace-probe', async (route) => {
    const result = fingerprintSuccess()
    result.context_check = {
      status: 'suspected_injection',
      expected_input_tokens: 6,
      reported_input_tokens: 34,
      delta: 28,
      log_id: fingerprintLogID,
    }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: result }),
    })
  })
  await page.goto('/groups/1?tab=models', { waitUntil: 'load' })
  await page.getByRole('button', { name: 'Fingerprint', exact: true }).click()

  const dialog = page.getByRole('dialog', { name: 'Model fingerprint' })
  const section = dialog.getByRole('region', { name: 'Context fidelity' })
  await expect(dialog.getByRole('heading', { level: 3 })).toHaveText([
    'Candidates',
    'Context fidelity',
    'Challenge runs',
  ])
  const candidates = dialog.getByRole('region', { name: 'Candidates' })
  const [rankingBounds, fidelityBounds] = await Promise.all([
    candidates.boundingBox(),
    section.boundingBox(),
  ])
  expect(rankingBounds?.y).toBe(fidelityBounds?.y)
  expect(rankingBounds!.x).toBeLessThan(fidelityBounds!.x)
  await expect(section.getByText('Suspected extra context')).toBeVisible()
  await expect(section.getByText('Expected input tokens')).toBeVisible()
  await expect(section.getByText('6', { exact: true })).toBeVisible()
  await expect(section.getByText('34', { exact: true })).toBeVisible()
  await expect(section.getByText('+28', { exact: true })).toBeVisible()
  await section.getByRole('button', { name: 'View log' }).click()
  await expect(page).toHaveURL(new RegExp(`selected_request_id=${fingerprintLogID}`))
})

test('unavailable context fidelity states uncertainty without counts or a log link', async ({
  page,
}) => {
  await mockGroupFingerprint(page)
  await page.route('**/api/modeltrace-probe', async (route) => {
    const result = fingerprintSuccess()
    result.context_check = {
      status: 'unavailable',
      expected_input_tokens: 6,
      reported_input_tokens: null,
      delta: null,
      log_id: null,
    }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: result }),
    })
  })
  await page.goto('/groups/1?tab=models', { waitUntil: 'load' })
  await page.getByRole('button', { name: 'Fingerprint', exact: true }).click()

  const dialog = page.getByRole('dialog', { name: 'Model fingerprint' })
  await expect(dialog.getByRole('heading', { name: 'Context fidelity' })).toBeVisible()
  await expect(dialog.getByText('Unavailable', { exact: true })).toBeVisible()
  await expect(dialog.getByText('Expected input tokens')).toBeVisible()
  const section = dialog.getByRole('region', { name: 'Context fidelity' })
  await expect(section.getByRole('button', { name: 'View log' })).toHaveCount(0)
  await expect(section.getByText('6', { exact: true })).toBeVisible()
  await expect(section.getByText('—', { exact: true })).toHaveCount(2)
})
