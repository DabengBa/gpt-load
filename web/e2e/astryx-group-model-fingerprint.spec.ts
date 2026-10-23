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
  await expect(dialog.getByText(/gpt-5\.6-sol/)).toBeVisible()
  await expect(dialog.getByText('8.1%', { exact: true })).toBeVisible()
  await expect(dialog.getByText('Attribution used 2/2 usable outputs')).toBeVisible()
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
    dialog.getByText('Fingerprint detection is not configured; set the ModelTrace URL and token.'),
  ).toBeVisible()

  await dialog.getByRole('button', { name: 'Retry' }).click()
  await expect(dialog.getByText('Attribution used 2/2 usable outputs')).toBeVisible()
  expect(mock.calls).toHaveLength(2)
})
