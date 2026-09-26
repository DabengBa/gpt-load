import { expect, test, type Page, type Route } from '@playwright/test'

// Phase 2 settings-domain coverage for the Astryx entry: section navigation
// with canonical `?section=`, runtime override/restore lifecycle, validation
// gating, header-rule editing, the discard dialog, the global unsaved-change
// interception, and the system-info surface.
//
// The mock mirrors the server contract from `projectSettings`: `data` carries
// `{values, overrides, read_only}` and PUT `/api/settings` accepts
// `{settings: patch}` where null entries restore the built-in default.

interface MockSettingsDto {
  values: Record<string, unknown>
  overrides: string[]
  read_only: string[]
}

function baseSettingsDto(): MockSettingsDto {
  return {
    values: {
      route_strategy: 'native_first',
      first_byte_timeout: 30,
      request_timeout: 600,
      stream_idle_timeout: 60,
      retry_count: 2,
      blacklist_threshold: 3,
      blacklist_release_seconds: 30,
      header_rules: { set: {}, remove: [] },
      cors: {
        enabled: false,
        allowed_origins: [],
        allowed_methods: [],
        allowed_headers: [],
        exposed_headers: [],
        allow_credentials: false,
        max_age: 600,
      },
      response_header_rules: { set: {}, remove: [] },
      affinity_enabled: true,
      responses_websocket_enabled: false,
      affinity_ttl: 300,
      affinity_capacity: 100,
      request_log_retention_days: 7,
      models_dev_auto_sync_enabled: true,
      proxy_config: {
        configured_mode: 'inherit',
        effective_mode: 'direct',
        effective_source: 'default',
        has_auth: false,
      },
    },
    overrides: ['request_timeout'],
    read_only: ['models_dev_auto_sync_enabled'],
  }
}

const systemInfoDto = {
  version: '9.9.9-e2e',
  deployment: {
    instance_mode: 'single',
    database: 'sqlite',
    distribution: 'single_binary',
  },
  data_dir: '/srv/gpt-load',
  auth_key: { source: 'environment', path: null },
  encryption: { enabled: true, source: 'key_file', path: '/srv/gpt-load/keys/enc.key' },
}

interface PutCapture {
  patch: Record<string, unknown>
}

// Stateful settings mock: GET returns the current DTO; PUT applies the patch
// (null entries restore defaults → leave overrides) and echoes the result,
// like the real handler.
async function mockSettings(page: Page, dto: MockSettingsDto) {
  const puts: PutCapture[] = []
  let current = structuredClone(dto)
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route: Route) => {
    const url = new URL(route.request().url())
    const path = url.pathname
    if (path === '/api/auth/session') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          code: 0,
          message: 'ok',
          data: { authenticated: true, principal_type: 'admin' },
        }),
      })
      return
    }
    if (path === '/api/settings') {
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON() as {
          settings: Record<string, unknown>
        }
        const patch = body.settings
        puts.push({ patch })
        const overrides = new Set(current.overrides)
        for (const [key, value] of Object.entries(patch)) {
          if (value === null) {
            overrides.delete(key)
          } else {
            overrides.add(key)
            current.values[key] = value
          }
        }
        current.overrides = [...overrides]
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data: current }),
      })
      return
    }
    if (path === '/api/system/info') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data: systemInfoDto }),
      })
      return
    }
    if (path === '/api/system/update') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data: { update: null } }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: {} }),
    })
  })
  return puts
}

async function expectAstryxDocument(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible()
  await expect(page.locator('[data-testid="desktop-nav"]')).toBeVisible()
}

function saveButton(page: Page) {
  return page.getByRole('button', { name: 'Save changes' })
}

test('renders all sections and canonicalizes the section route query', async ({
  page,
}) => {
  await mockSettings(page, baseSettingsDto())
  await page.goto('/settings', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  const nav = page.getByRole('navigation', { name: 'Settings sections' })
  await expect(nav).toBeVisible()
  for (const name of [
    'Routing and scheduling',
    'Connection and timeouts',
    'Retries and credential health',
    'Headers and CORS',
    'Data and maintenance',
    'System information',
  ]) {
    await expect(nav.getByRole('link', { name })).toBeVisible()
    await expect(
      page.getByRole('heading', { name, exact: true }),
    ).toBeVisible()
  }

  // Nav link writes the sparse `?section=` query; the canonical section stays
  // bare `/settings`.
  await nav.getByRole('link', { name: 'Data and maintenance' }).click()
  await expect(page).toHaveURL(/section=data-maintenance/)
  await nav.getByRole('link', { name: 'Routing and scheduling' }).click()
  await expect(page).not.toHaveURL(/section=/)

  // A bogus section deep-link canonicalizes back to bare /settings.
  await page.goto('/settings?section=bogus', { waitUntil: 'load' })
  await expect(page).not.toHaveURL(/section=/)
  await expect(
    page.getByRole('heading', { name: 'Settings', exact: true }),
  ).toBeVisible()
})

test('override → save publishes only the changed keys', async ({ page }) => {
  const puts = await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=connection', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  // Baseline: request_timeout is overridden (fixture) — the row shows the
  // override affordance; stream_idle_timeout shows "Override".
  await page
    .getByRole('button', { name: 'Override · Stream idle timeout' })
    .click()
  const input = page.getByLabel('Value for Stream idle timeout')
  await input.fill('90')
  await expect(page.getByText('1 unsaved runtime settings')).toBeVisible()

  await saveButton(page).click()
  await expect(page.getByText('Settings saved at')).toBeVisible()
  expect(puts).toHaveLength(1)
  expect(puts[0].patch).toEqual({ stream_idle_timeout: 90 })
})

test('restore default sends a null patch entry', async ({ page }) => {
  const puts = await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=connection', { waitUntil: 'load' })

  await page
    .getByRole('button', { name: 'Restore default · Total request timeout' })
    .click()
  await expect(page.getByText('Pending restore')).toBeVisible()

  await saveButton(page).click()
  await expect(page.getByText('Settings saved at')).toBeVisible()
  expect(puts.at(-1)?.patch).toEqual({ request_timeout: null })
})

test('invalid intermediate input blocks save with a validation summary', async ({
  page,
}) => {
  await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=connection', { waitUntil: 'load' })

  await page
    .getByRole('button', { name: 'Override · Stream idle timeout' })
    .click()
  const input = page.getByLabel('Value for Stream idle timeout')
  await input.fill('0')

  await expect(
    page.getByText('Fix these settings before saving:'),
  ).toBeVisible()
  await expect(
    page.getByText(
      'Enter a positive safe integer no greater than 9,223,372,036.',
    ),
  ).toBeVisible()
  await expect(saveButton(page)).toBeDisabled()

  // Correcting the value re-enables save.
  await input.fill('45')
  await expect(saveButton(page)).toBeEnabled()
})

test('read-only settings are locked and cannot enter the draft', async ({
  page,
}) => {
  const puts = await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=reliability', { waitUntil: 'load' })

  const lockedRow = page
    .locator('div')
    .filter({ hasText: 'Automatically sync the Models.dev catalog' })
    .last()
  await expect(page.getByText('Locked by environment')).toBeVisible()
  await expect(lockedRow.getByRole('button', { name: /Override/ })).toHaveCount(0)

  // Editing a normal field and saving must not include the locked key.
  await page.getByRole('button', { name: 'Override · Maximum attempts' }).click()
  await page.getByLabel('Value for Maximum attempts').fill('4')
  await saveButton(page).click()
  expect(puts.at(-1)?.patch).toEqual({ retry_count: 4 })
})

test('discard dialog restores the published baseline', async ({ page }) => {
  await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=connection', { waitUntil: 'load' })

  await page
    .getByRole('button', { name: 'Override · Stream idle timeout' })
    .click()
  const input = page.getByLabel('Value for Stream idle timeout')
  await input.fill('90')
  await expect(page.getByText('1 unsaved runtime settings')).toBeVisible()

  await page.getByRole('button', { name: 'Discard', exact: true }).click()
  const dialog = page.getByRole('dialog', {
    name: 'Discard all unsaved changes?',
  })
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: 'Discard changes' }).click()

  // Baseline restored: dirty badge gone and the override button is back.
  await expect(page.getByText('unsaved runtime settings')).toHaveCount(0)
  await expect(
    page.getByRole('button', { name: 'Override · Stream idle timeout' }),
  ).toBeVisible()
})

test('unsaved-change interception blocks leaving the page', async ({ page }) => {
  await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=connection', { waitUntil: 'load' })

  await page
    .getByRole('button', { name: 'Override · Stream idle timeout' })
    .click()
  await page.getByLabel('Value for Stream idle timeout').fill('90')

  await page
    .locator('[data-testid="desktop-nav"]')
    .getByRole('link', { name: 'Groups' })
    .click()
  const dialog = page.getByRole('alertdialog', {
    name: 'Discard unsaved changes?',
  })
  await expect(dialog).toBeVisible()

  // Continue editing stays on /settings with the draft intact.
  await dialog.getByRole('button', { name: 'Continue editing' }).click()
  await expect(page).toHaveURL(/settings/)
  await expect(page.getByLabel('Value for Stream idle timeout')).toHaveValue(
    '90',
  )

  // Confirming discards and navigates away.
  await page
    .locator('[data-testid="desktop-nav"]')
    .getByRole('link', { name: 'Groups' })
    .click()
  await page
    .getByRole('alertdialog', { name: 'Discard unsaved changes?' })
    .getByRole('button', { name: 'Discard changes' })
    .click()
  await expect(page).toHaveURL(/\/groups/)
})

test('header-rule rows edit, validate, and publish to the save patch', async ({
  page,
}) => {
  const puts = await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=browser-access', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  // The upstream-rules block shows the effective summary until overridden.
  // (leaf articles only — the page-level article matches descendant text too)
  const block = page
    .getByRole('article')
    .filter({ hasText: 'Upstream request header rules' })
    .filter({ hasNot: page.getByRole('article') })
  await expect(block.getByText('0 rules')).toBeVisible()

  await page
    .getByRole('button', { name: 'Override · Upstream request header rules' })
    .click()
  const editor = block.getByRole('region', { name: 'Header rules' })

  await editor.getByRole('button', { name: 'Add rule' }).click()
  await editor.getByLabel('Header name').first().fill('X-Trace-Id')
  await editor.getByLabel('Header value').first().fill('abc-123')
  await expect(block.getByText('1 rules')).toBeVisible()

  // Duplicate names fail validation case-insensitively and block save.
  await editor.getByRole('button', { name: 'Add rule' }).click()
  await editor.getByLabel('Header name').nth(1).fill('x-trace-id')
  // Per-row error is surfaced via the describedby span (tooltip variant keeps
  // the visible copy hidden until focus); the global summary + disabled save
  // are the user-visible contract.
  await expect(
    page.locator('span[id$="-header-name-2-error"]'),
  ).toContainText('Duplicate header names')
  await expect(
    page.getByText('Fix these settings before saving:'),
  ).toBeVisible()
  await expect(saveButton(page)).toBeDisabled()

  await editor.getByRole('button', { name: 'Delete rule' }).nth(1).click()
  await expect(saveButton(page)).toBeEnabled()

  await saveButton(page).click()
  await expect(page.getByText('Settings saved at')).toBeVisible()
  expect(puts.at(-1)?.patch).toMatchObject({
    header_rules: { set: { 'X-Trace-Id': 'abc-123' }, remove: [] },
  })
})

test('CORS enablement round-trips through the save patch', async ({ page }) => {
  const puts = await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=browser-access', { waitUntil: 'load' })

  // CORS edits require the override like every other runtime setting.
  await page.getByRole('button', { name: 'Override · CORS policy' }).click()
  await page.getByRole('switch', { name: 'Enable CORS' }).click()
  await page.getByLabel('Allowed origins').fill('https://app.example')
  await page.getByLabel('Allowed methods').fill('GET, POST')
  // Enabled CORS requires a non-empty allowed-headers list.
  await page
    .getByLabel('Allowed request headers')
    .fill('Content-Type, Authorization')

  await saveButton(page).click()
  await expect(page.getByText('Settings saved at')).toBeVisible()
  const cors = puts.at(-1)?.patch.cors as Record<string, unknown>
  expect(cors).toMatchObject({
    enabled: true,
    allowed_origins: ['https://app.example'],
    allowed_methods: ['GET', 'POST'],
  })
})

test('system section renders deployment facts and the latest-version state', async ({
  page,
}) => {
  await mockSettings(page, baseSettingsDto())
  await page.goto('/settings?section=system', { waitUntil: 'load' })

  await expect(page.getByText('9.9.9-e2e')).toBeVisible()
  await expect(page.getByText('SQLite')).toBeVisible()
  await expect(page.getByText('/srv/gpt-load', { exact: true })).toBeVisible()
  // Update checks are manual — the button drives a force=true fetch.
  await page.getByRole('button', { name: 'Check for updates' }).click()
  await expect(
    page.getByText('You are already on the latest version'),
  ).toBeVisible()

  // Secret sources render labels; key-file path exposes the copy affordance.
  await expect(page.getByText('Environment variable', { exact: true })).toBeVisible()
  await expect(page.getByText('Key file', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Copy path' })).toBeVisible()
})
