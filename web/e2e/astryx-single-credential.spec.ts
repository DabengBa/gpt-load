import { expect, test } from '@playwright/test'

test.setTimeout(90000)

for (const width of [1280, 390]) {
  test(`single credential actions at ${width}px`, async ({ page }) => {
    page.on('pageerror', (error) => console.error(error.message))
    await page.setViewportSize({ width, height: 900 })
    await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'fixture-auth'))
    let credential: Record<string, unknown> | null = {
      credential_id: 11,
      secret_version: 1,
      mask: 'fixture-mask',

      effective_status: 'blacklisted',
      recent_success_count: 0,
      recent_failure_count: 1,
      consecutive_failure_count: 1,
      last_failure_category: 'invalid_key',
      last_status_code: 401,
      cooldown_until_ms: null,
      recovery: { mode: 'scheduled_release', automatic: true, at_ms: null },
    }
    const requests: string[] = []
    await page.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname
      if (path.includes('/credential') && route.request().method() !== 'GET')
        expect(route.request().headers()['x-credential-id']).toBe('11')
      requests.push(`${route.request().method()} ${path}`)
      let data: unknown = null
      if (path === '/api/auth/session') data = { authenticated: true, principal_type: 'admin' }
      else if (path === '/api/groups/7')
        data = {
          id: 7,
          name: 'Fixture group',
          channel_id: 'openai',
          params: {},
          provider_url: null,

          service_status: 'unavailable',
          service_status_reason: 'no_models',
          credential_configured: credential !== null,
          credential_status: credential?.effective_status ?? null,
          model_count: 0,
        }
      else if (path === '/api/groups/7/credential/test')
        data = {
          outcome: 'passed',
          model: 'fixture-model',
          protocol: 'openai-completions',
          latency_ms: 1,
          reason: null,
          recovered: true,
          log_id: null,
          tested_at_ms: 1700000000000,
        }
      else if (path === '/api/groups/7/credential/restore') {
        credential = {
          ...credential,
          effective_status: 'available',
          recovery: { mode: 'none', automatic: false, at_ms: null },
        }
        data = credential
      } else if (path === '/api/groups/7/credential') {
        if (route.request().method() === 'DELETE') credential = null
        if (route.request().method() === 'PUT') {
          expect(route.request().postDataJSON()).toEqual({ credential: 'fixture-new-value' })
          credential = { ...credential, secret_version: 2, mask: 'updated-mask' }
          data = credential
        } else data = { credential }
      } else if (path === '/api/channels') data = { items: [] }
      else {
        await route.fulfill({
          status: 404,
          json: { code: 'NOT_FOUND', message: 'fixture unavailable' },
        })
        return
      }
      await route.fulfill({ json: { code: 0, message: 'ok', data } })
    })
    await page.goto('/groups/7?tab=credentials', { waitUntil: 'commit' })
    await expect(page.getByRole('button', { name: 'Copy key' })).toContainText('fixture-mask', {
      timeout: 60000,
    })
    await expect(page.locator('input[type=checkbox]')).toHaveCount(0)
    await page.getByRole('button', { name: 'More actions' }).click()
    await page.getByRole('button', { name: 'Test connection', exact: true }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
    await expect(page.getByText('fixture-model', { exact: true })).toBeVisible()
    await page
      .getByRole('dialog')
      .getByRole('button', { name: 'Close', exact: true })
      .filter({ hasText: 'Close' })
      .click()
    await page.getByRole('button', { name: 'More actions' }).click()
    await page.getByRole('button', { name: 'Restore now', exact: true }).click()
    await expect.poll(() => requests).toContain('POST /api/groups/7/credential/restore')
    await page.getByRole('button', { name: 'Update key', exact: true }).click()
    await page.getByPlaceholder('Paste a new API key').fill('fixture-new-value')
    await page.getByRole('button', { name: 'Save key', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Copy key' })).toContainText('updated-mask')
    await page.screenshot({
      path: test.info().outputPath(`credential-${width}.png`),
      fullPage: true,
    })
    await page.getByRole('button', { name: 'More actions' }).click()
    await page.getByRole('button', { name: 'Delete', exact: true }).click()
    await page.getByRole('alertdialog').getByRole('button', { name: 'Delete', exact: true }).click()
    await expect(page.getByText('No keys yet', { exact: true })).toBeVisible()
    await page.screenshot({ path: test.info().outputPath(`empty-${width}.png`), fullPage: true })
  })
}
