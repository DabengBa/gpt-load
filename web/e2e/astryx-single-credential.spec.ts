import { expect, test } from '@playwright/test'

test.setTimeout(90000)

for (const width of [1280, 390]) {
  test(`single credential actions at ${width}px`, async ({ page }) => {
    page.on('pageerror', (error) => console.error(error.message))
    await page.setViewportSize({ width, height: 900 })
    await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'fixture-auth'))
    let credential: Record<string, unknown> | null = {
      credential_id: 11,
      connection_type: 'api_key',
      secret_version: 1,
      mask: 'fixture-mask',
      account: {},
      auth_state: 'ready',
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
          connection_type: 'api_key',
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
        } else data = { credential, observation: null }
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

for (const failure of ['network', 'unknown', 'pending', 'rejected', 'replaced'] as const) {
  test(`${failure} reset outcome handles operation key across close and reopen`, async ({
    page,
  }) => {
    await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'fixture-auth'))
    const keys: string[] = []
    let credentialId = 11
    const observation = {
      state: 'fresh',
      observation_version: 1,
      observed_at_ms: 1700000000000,
      last_attempt_at_ms: 1700000000000,
      snapshot: { plan_summary: {}, quota_windows: [], reset_credits_available: 2 },
    }
    await page.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname
      let data: unknown = null
      if (path === '/api/auth/session') data = { authenticated: true, principal_type: 'admin' }
      else if (path === '/api/groups/7')
        data = {
          id: 7,
          name: 'Subscription fixture',
          channel_id: 'claude',
          connection_type: 'subscription',
          params: {},
          provider_url: null,

          service_status: 'unavailable',
          service_status_reason: 'no_models',
          credential_configured: true,
          credential_status: 'available',
          model_count: 0,
        }
      else if (path === '/api/groups/7/credential')
        data = {
          credential: {
            credential_id: credentialId,
            connection_type: 'subscription',
            secret_version: 1,
            mask: 'subscription-mask',
            account: {},
            auth_state: 'ready',
            effective_status: 'available',
            recent_success_count: 0,
            recent_failure_count: 0,
            consecutive_failure_count: 0,
            last_failure_category: 'ok',
            last_status_code: null,
            cooldown_until_ms: null,
            recovery: { mode: 'none', automatic: false, at_ms: null },
            observation,
          },
          observation,
        }
      else if (path === '/api/channels')
        data = {
          items: [
            {
              channel_id: 'claude',
              name: 'Claude',
              mark: 'CS',
              icon: 'claude',
              search_terms: [],
              description: '',
              default_base_url: 'https://api.anthropic.com',
              notices: [],
              param_fields: [],
              credential_fields: [],
              connection: {
                type: 'subscription',
                credential_input: 'authorization',
                authorization_methods: ['browser_oauth'],
              },
              routes: [],
              client_protocols: [],
              capabilities: {
                model_discovery: false,
                quota_observation: true,
                credential_actions: ['reset_credit'],
                outbound_proxy: false,
              },
            },
          ],
          total: 1,
        }
      else if (path.endsWith('/observation-refresh')) {
        credentialId = 12
        data = observation
      } else if (path.endsWith('/reset-credits/consume')) {
        keys.push(route.request().headers()['idempotency-key'] ?? '')
        expect(route.request().headers()['x-credential-id']).toBe(String(credentialId))
        if (keys.length === 1) {
          if (failure === 'network' || failure === 'replaced') await route.abort('failed')
          else
            await route.fulfill({
              status: failure === 'rejected' ? 502 : 503,
              json: {
                code:
                  failure === 'pending'
                    ? 'CONTROL_RECOVERY_PENDING'
                    : failure === 'unknown'
                      ? 'RESET_CREDIT_OUTCOME_UNKNOWN'
                      : 'RESET_CREDIT_REJECTED',
                message: 'fixture reset failure',
              },
            })
          return
        }
        data = { status: 'succeeded', windows_reset: 1, replayed: true }
      } else {
        await route.fulfill({
          status: 404,
          json: { code: 'NOT_FOUND', message: 'fixture unavailable' },
        })
        return
      }
      await route.fulfill({ json: { code: 0, message: 'ok', data } })
    })
    await page.goto('/groups/7?tab=credentials', { waitUntil: 'commit' })
    await page.getByRole('button', { name: 'Reset quota', exact: true }).click()
    await page
      .getByRole('alertdialog')
      .getByRole('button', { name: 'Use reset credit', exact: true })
      .click()
    await expect(
      page.getByText('Unable to use reset credit', { exact: true }).first(),
    ).toBeVisible()
    await expect(
      page.getByRole('alertdialog').getByRole('button', { name: 'Use reset credit', exact: true }),
    ).toBeEnabled()
    await page.getByRole('alertdialog').getByRole('button', { name: 'Cancel', exact: true }).click()
    if (failure === 'replaced') {
      await page.getByRole('button', { name: 'Sync', exact: true }).click()
      await expect.poll(() => credentialId).toBe(12)
      await expect(page.getByRole('button', { name: 'Sync', exact: true })).toBeEnabled()
    }
    await page.getByRole('button', { name: 'Reset quota', exact: true }).click()
    await page
      .getByRole('alertdialog')
      .getByRole('button', { name: 'Use reset credit', exact: true })
      .click()
    await expect.poll(() => keys.length).toBe(2)
    expect(keys[0]).not.toBe('')
    if (failure === 'rejected' || failure === 'replaced') expect(keys[1]).not.toBe(keys[0])
    else expect(keys[1]).toBe(keys[0])
    await expect(page.getByRole('alertdialog')).toHaveCount(0)
    await page.getByRole('button', { name: 'Reset quota', exact: true }).click()
    await page
      .getByRole('alertdialog')
      .getByRole('button', { name: 'Use reset credit', exact: true })
      .click()
    await expect.poll(() => keys.length).toBe(3)
    expect(keys[2]).not.toBe(keys[1])
  })
}
