import { expect, test, type Page } from '@playwright/test'

const adminKey = 'e2e-admin-key'

type SchedulePatch = {
  snapshot_revision: number
  protocol: string
  external_model: string
  access_key_id: number
  operation: string
  updates: Array<{
    group_id: number
    entry_id: string
    weight?: number | null
    priority?: number | null
    enabled?: boolean
    reasoning_effort?: string | null
  }>
}

function envelope(data: unknown) {
  return { code: 0, message: 'OK', data }
}

function response(data: unknown, status = 200) {
  return {
    status,
    contentType: 'application/json',
    body: JSON.stringify(envelope(data)),
  }
}

function conflictResponse() {
  return {
    status: 409,
    contentType: 'application/json',
    body: JSON.stringify({
      code: 'MODEL_ROUTE_SCHEDULE_REVISION_CONFLICT',
      message: 'schedule changed',
      data: null,
    }),
  }
}

function entry(entryID: string, modelID: string, weight: number, priority = 1, override = false) {
  return {
    entry_id: entryID,
    model_id: modelID,
    alias: '',
    weight,
    priority,
    fallback: priority > 1,
    enabled: true,
    circuit_breaker: {
      configured: { blacklist_threshold: null, cooldown_seconds: null },
      effective: { blacklist_threshold: 3, cooldown_seconds: 60 },
      sources: { blacklist_threshold: 'default', cooldown_seconds: 'default' },
    },
    reasoning: {
      configured: override ? 'max' : (null as string | null),
      effective: override ? 'max' : (null as string | null),
      source: override ? 'entry' : 'provider_default',
    },
    runtime: {
      state: 'available',
      cooldown_until_ms: null,
      blacklist_release_at_ms: null,
      failure_count: 0,
      failure_version: 0,
    },
    included: true,
    routable: true,
    reason_code: null as string | null,
    configured_share: 0.5,
    effective_share: 0.5,
    credentials: [],
  }
}

function detailForModel(externalModel: string) {
  const suffix = externalModel === 'worker-b' ? '-b' : ''
  const firstEntryID = externalModel === 'worker-b' ? 'derived:worker-b#model-a-b' : 'entry-1'
  return {
    observed_at_ms: 1_700_000_000_000,
    snapshot_revision: externalModel === 'worker-b' ? 21 : 11,
    external_model: externalModel,
    protocol: 'openai-completions',
    operation: 'chat_completion',
    route_requirement: 'any',
    access_key: { id: 7, name: 'e2e access key', status: 'active' },
    routable: true,
    reason_code: null,
    groups: [
      {
        group_id: 1,
        group_name: `primary group${suffix}`,
        channel_id: `channel-1${suffix}`,
        enabled: true,
        request_count: 10,
        success_rate: 1,
        entries: [entry(firstEntryID, `model-a${suffix}`, 50)],
      },
      {
        group_id: 2,
        group_name: `secondary group${suffix}`,
        channel_id: `channel-2${suffix}`,
        enabled: false,
        request_count: 20,
        success_rate: 1,
        entries: [entry(`entry-2${suffix}`, `model-b${suffix}`, 50, 1, true)],
      },
    ],
  }
}

type PatchOutcome = 'success' | 'error' | 'conflict'
type DetailOutcome = 'success' | 'loading' | 'error' | 'empty'

async function installScheduleEditingRoutes(
  page: Page,
  patchOutcome: PatchOutcome = 'success',
  detailOutcome: DetailOutcome = 'success',
  projectDetail = detailForModel,
): Promise<SchedulePatch[]> {
  await page.addInitScript((authKey) => {
    window.localStorage.setItem('gpt-load.auth-key', authKey)
  }, adminKey)

  const patches: SchedulePatch[] = []
  await page.route(
    (url) => url.pathname === '/api' || url.pathname.startsWith('/api/'),
    async (route) => {
      const request = route.request()
      const url = new URL(request.url())

      if (url.pathname === '/api/auth/session') {
        await route.fulfill(response({ authenticated: true, principal_type: 'admin' }))
        return
      }
      if (url.pathname === '/api/model-route/schedule' && request.method() === 'GET') {
        await route.fulfill(
          response({
            items: [
              {
                external_model: 'worker',
                protocol: 'openai-completions',
                operation: 'chat_completion',
                candidate_count: 2,
                group_count: 2,
                has_fallback: false,
                cooled_candidates: 0,
                blacklisted_candidates: 0,
              },
              {
                external_model: 'worker-b',
                protocol: 'openai-completions',
                operation: 'chat_completion',
                candidate_count: 2,
                group_count: 2,
                has_fallback: false,
                cooled_candidates: 0,
                blacklisted_candidates: 0,
              },
            ],
          }),
        )
        return
      }
      if (url.pathname === '/api/access-keys/options') {
        await route.fulfill(response([{ id: 7, name: 'e2e access key', status: 'active' }]))
        return
      }
      if (url.pathname === '/api/model-route/schedule/detail') {
        if (detailOutcome === 'loading') await new Promise((resolve) => setTimeout(resolve, 600))
        if (detailOutcome === 'error') {
          await route.fulfill(response({}, 500))
          return
        }
        if (detailOutcome === 'empty') {
          await route.fulfill(
            response({
              ...detailForModel(url.searchParams.get('external_model') ?? 'worker'),
              routable: false,
              reason_code: 'no_available_group',
              groups: [],
            }),
          )
          return
        }
        await route.fulfill(
          response(projectDetail(url.searchParams.get('external_model') ?? 'worker')),
        )
        return
      }
      if (url.pathname === '/api/model-route/schedule' && request.method() === 'PATCH') {
        patches.push(request.postDataJSON() as SchedulePatch)
        if (patchOutcome === 'conflict') {
          await route.fulfill(conflictResponse())
        } else if (patchOutcome === 'error') {
          await route.fulfill(response({}, 500))
        } else {
          await route.fulfill(response({ snapshot_revision_new: 12, detail: null }))
        }
        return
      }

      await route.fulfill(response({}, 404))
    },
  )

  return patches
}

async function openSchedule(page: Page): Promise<void> {
  await page.goto('/schedule?schedule_model=worker')
  await page.getByRole('heading', { name: 'Schedule detail' }).waitFor()
  await expect(page.locator('.schedule-row')).toHaveCount(3)
}

async function waitForScheduleDetailReady(page: Page): Promise<void> {
  await page.getByRole('heading', { name: 'Schedule detail' }).waitFor({ timeout: 30_000 })
  await page.getByRole('table', { name: 'Schedule detail' }).waitFor({ timeout: 30_000 })
}

test.describe('schedule editing', () => {
  test('sorts flat candidates by priority and identity, keeps editing stationary, then reorders on save', async ({
    page,
  }) => {
    let saved = false
    const patches = await installScheduleEditingRoutes(page, 'success', 'success', (model) => {
      const detail = detailForModel(model)
      const first = detail.groups[0]!.entries[0]!
      const second = detail.groups[1]!.entries[0]!
      first.priority = saved ? 1 : 3
      first.fallback = !saved
      second.priority = 2
      second.fallback = true
      first.configured_share = 1
      second.configured_share = 0.7
      second.effective_share = 0.3
      const third = entry('entry-0', 'model-zero', 20, 2)
      third.configured_share = 0.3
      third.effective_share = 0.2
      detail.groups[0]!.entries.push(third)
      return detail
    })
    page.on('request', (request) => {
      if (request.url().includes('/api/model-route/schedule') && request.method() === 'PATCH')
        saved = true
    })
    await page.goto('/schedule?schedule_model=worker&schedule_group=2&schedule_row=2%3Aentry-2')
    await waitForScheduleDetailReady(page)
    const candidateKeys = page.locator('.schedule-row[data-row-key]')
    await expect(candidateKeys).toHaveCount(3)
    const keys = async () =>
      candidateKeys.evaluateAll((nodes) => nodes.map((node) => node.getAttribute('data-row-key')))
    await expect.poll(keys).toEqual(['1:entry-0', '2:entry-2', '1:entry-1'])
    await expect(page.locator('.schedule-row--selected')).toHaveAttribute(
      'data-row-key',
      '2:entry-2',
    )
    await page
      .locator('.schedule-row', { hasText: 'model-a' })
      .getByRole('button', { name: 'Edit details' })
      .click()
    await page.locator('#priority-2').fill('1')
    await expect.poll(keys).toEqual(['1:entry-0', '2:entry-2', '1:entry-1'])
    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    await expect.poll(keys).toEqual(['1:entry-1', '1:entry-0', '2:entry-2'])
  })

  test('keeps long names and expanded editing readable at desktop and 620px with a save bar', async ({
    page,
  }) => {
    await installScheduleEditingRoutes(page, 'success', 'success', (model) => {
      const detail = detailForModel(model)
      detail.groups[0]!.group_name =
        'A very long upstream group name that must wrap across multiple lines'
      detail.groups[0]!.entries[0]!.alias =
        'A very long external upstream model alias with multiple segments'
      detail.groups[0]!.entries[0]!.priority = 2
      detail.groups[0]!.entries[0]!.fallback = true
      detail.groups[0]!.entries[0]!.configured_share = 1
      detail.groups[1]!.entries[0]!.configured_share = 1
      detail.groups[1]!.entries[0]!.priority = 1
      return detail
    })
    const evidence =
      '/mnt/projects/repos/gpt-load/.tmp/schedule-center-row-controls_20260924T085306+0800/units/U002/evidence'
    for (const width of [1280, 620]) {
      await page.setViewportSize({ width, height: 900 })
      await page.goto('/schedule?schedule_model=worker')
      await waitForScheduleDetailReady(page)
      await expect(page.locator('.schedule-row[data-row-key]')).toHaveCount(2)
      await expect(page.locator('.schedule-row[data-row-key]').first()).toHaveAttribute(
        'data-row-key',
        '2:entry-2',
      )
      await page
        .locator('.schedule-row[data-row-key="1:entry-1"]')
        .getByRole('button', { name: 'Edit details' })
        .click()
      await page
        .locator('.schedule-row[data-row-key="1:entry-1"]')
        .locator('input[id^="priority-"]')
        .fill('3')
      await expect(page.getByRole('button', { name: 'Save' })).toBeEnabled()
      if (width === 620) {
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
          ),
        ).toBeLessThanOrEqual(1)
      }
      await page.screenshot({ path: `${evidence}/schedule-${width}.png`, fullPage: true })
    }
  })

  test('projects a disabled entry and keeps probe results free of group switch controls', async ({
    page,
  }) => {
    await installScheduleEditingRoutes(page, 'success', 'success', (model) => {
      const detail = detailForModel(model)
      const disabled = detail.groups[0]!.entries[0]!
      disabled.enabled = false
      disabled.routable = false
      disabled.reason_code = 'entry_disabled'
      disabled.effective_share = 0
      detail.groups[1]!.entries[0]!.effective_share = 1
      return detail
    })
    const groupSettingsCalls: string[] = []
    page.on('request', (request) => {
      if (request.url().includes('/api/groups/') && request.method() === 'PATCH') {
        groupSettingsCalls.push(request.url())
      }
    })
    await page.route('**/api/model-probe', (route) =>
      route.fulfill(
        response({
          results: [
            {
              group_id: 1,
              group_name: 'primary group',
              model: 'model-a',
              outcome: 'passed',
              reason: null,
              protocol: 'openai-completions',
              route_mode: 'native',
              status_code: 200,
              latency_ms: 5,
              credential_id: 1,
              credential_label: 'credential',
              recovered: false,
              log_id: null,
              tested_at_ms: 1_700_000_000_000,
            },
          ],
        }),
      ),
    )
    await openSchedule(page)
    await expect(page.locator('.schedule-row', { hasText: 'model-a' })).toBeVisible()
    await expect(page.getByRole('switch', { name: 'Toggle entry model-a' })).not.toBeChecked()
    await page
      .locator('.schedule-row', { hasText: 'model-a' })
      .getByRole('button', { name: 'Probe' })
      .click()
    await expect(page.getByRole('dialog', { name: 'Model liveness' })).toBeVisible()
    await expect(page.getByRole('dialog').getByText('Passed', { exact: true })).toBeVisible()
    await expect(
      page.getByRole('dialog').getByText('Adjust group switches by probe result'),
    ).toHaveCount(0)
    await expect(page.getByRole('dialog').getByText('primary group', { exact: true })).toHaveCount(
      1,
    )
    expect(groupSettingsCalls).toEqual([])
  })

  test('toggles only the selected entry through schedule PATCH, never group settings', async ({
    page,
  }) => {
    const patches = await installScheduleEditingRoutes(page)
    const groupSettingsCalls: string[] = []
    page.on('request', (request) => {
      if (request.url().includes('/api/groups/') && request.method() === 'PATCH') {
        groupSettingsCalls.push(request.url())
      }
    })
    await openSchedule(page)
    await page.getByRole('switch', { name: 'Toggle entry model-a' }).click()
    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]?.updates).toEqual([{ group_id: 1, entry_id: 'entry-1', enabled: false }])
    expect(groupSettingsCalls).toEqual([])
  })

  test('uses item overrides without group default controls or updates', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page)
    await openSchedule(page)

    await expect(page.getByRole('combobox', { name: /Group default/ })).toHaveCount(0)
    await page.getByRole('combobox', { name: 'Entry override model-a' }).click()
    await page.locator('.app-select__item[data-value="high"]').click()
    await page.getByRole('button', { name: 'Save' }).click()

    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]).toMatchObject({
      updates: [{ group_id: 1, entry_id: 'entry-1', reasoning_effort: 'high' }],
    })
    expect(patches[0]).not.toHaveProperty('group_updates')
  })

  async function editBothGroups(page: Page): Promise<void> {
    const firstWeight = page.locator('#weight-0')
    const secondWeight = page.locator('#weight-1')
    await firstWeight.fill('61')
    await secondWeight.fill('72')
    await expect(firstWeight).toHaveValue('61')
    await expect(secondWeight).toHaveValue('72')
    await page.getByRole('combobox', { name: 'Entry override model-b' }).click()
    await page.locator('.app-select__item[data-value=""]').click()
  }

  test('renders loading, error, and empty schedule states', async ({ page }) => {
    await installScheduleEditingRoutes(page, 'success', 'loading')
    await page.goto('/schedule?schedule_model=worker')
    await expect(
      page.getByRole('status').filter({ hasText: 'Loading schedule details' }),
    ).toBeVisible()

    await page.unrouteAll({ behavior: 'wait' })
    await installScheduleEditingRoutes(page, 'success', 'error')
    await page.reload()
    await expect(page.getByRole('alert')).toBeVisible()

    await page.unrouteAll({ behavior: 'wait' })
    await installScheduleEditingRoutes(page, 'success', 'empty')
    await page.reload()
    await expect(page.getByText('No entries are available.')).toBeVisible()
  })

  test('keeps drafts from multiple groups and submits them in one save', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page)
    await openSchedule(page)
    await editBothGroups(page)

    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]).toMatchObject({
      snapshot_revision: 11,
      protocol: 'openai-completions',
      external_model: 'worker',
      access_key_id: 7,
      operation: 'chat_completion',
      updates: [
        { group_id: 1, entry_id: 'entry-1', weight: 61 },
        { group_id: 2, entry_id: 'entry-2', weight: 72 },
      ],
    })
  })

  test('submits an item override in one revisioned save', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page)
    await openSchedule(page)

    await page.getByRole('combobox', { name: 'Entry override model-a' }).click()
    await page.locator('.app-select__item[data-value="low"]').click()
    await page.getByRole('button', { name: 'Save' }).click()

    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]).toMatchObject({
      snapshot_revision: 11,
      updates: [{ group_id: 1, entry_id: 'entry-1', reasoning_effort: 'low' }],
    })
    expect(patches[0]).not.toHaveProperty('group_updates')
  })

  test('clears an item reasoning override with an explicit null', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page)
    await openSchedule(page)

    await page.getByRole('combobox', { name: 'Entry override model-b' }).click()
    await page.locator('.app-select__item[data-value=""]').click()
    await page.getByRole('button', { name: 'Save' }).click()

    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]).toMatchObject({
      updates: [{ group_id: 2, entry_id: 'entry-2', reasoning_effort: null }],
    })
    expect(patches[0]).not.toHaveProperty('group_updates')
  })

  test('offers arbitrary canonical efforts for an entry override', async ({ page }) => {
    await installScheduleEditingRoutes(page)
    await openSchedule(page)

    const override = page.getByRole('combobox', { name: 'Entry override model-b' })
    await override.click()
    for (const effort of ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']) {
      await expect(page.locator(`.app-select__item[data-value="${effort}"]`)).toBeVisible()
    }
    await page.locator('.app-select__item[data-value="xhigh"]').click()
    await expect(override).toContainText('xhigh')
    await expect(page.getByText('requested effort is not supported')).toHaveCount(0)
    await expect(page.getByText('Unsupported', { exact: true })).toHaveCount(0)
    await expect(page.getByText('Disabled', { exact: true })).toBeVisible()
    await expect(override).toBeEnabled()
  })

  test('keeps derived reasoning controls read-only and contains narrow-screen overflow', async ({
    page,
  }) => {
    await installScheduleEditingRoutes(page)
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('/schedule?schedule_model=worker-b')
    await page.getByRole('heading', { name: 'Schedule detail' }).waitFor()

    await expect(page.getByRole('combobox', { name: 'Entry override model-a-b' })).toBeDisabled()
    await expect(page.locator('#weight-0')).toBeDisabled()
    await page.getByRole('combobox', { name: 'Entry override model-b-b' }).focus()
    await expect(page.getByRole('combobox', { name: 'Entry override model-b-b' })).toBeFocused()
    const pageOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    )
    expect(pageOverflow).toBeLessThanOrEqual(1)
  })

  test('keeps drafts visible after a revision conflict', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page, 'conflict')
    await openSchedule(page)
    await editBothGroups(page)
    await page.getByRole('combobox', { name: 'Entry override model-a' }).click()
    await page.locator('.app-select__item[data-value="low"]').click()

    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    await expect(page.locator('#weight-0')).toHaveValue('61')
    await expect(page.locator('#weight-1')).toHaveValue('72')
    await expect(page.getByRole('combobox', { name: 'Entry override model-a' })).toContainText(
      'low',
    )
  })

  test('keeps drafts visible after a failed save', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page, 'error')
    await openSchedule(page)
    await editBothGroups(page)

    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    await expect(page.locator('#weight-0')).toHaveValue('61')
    await expect(page.locator('#weight-1')).toHaveValue('72')
  })

  test('does not restore replace drafts into an externally navigated model context', async ({
    page,
  }) => {
    await installScheduleEditingRoutes(page)
    await page.goto('/schedule?schedule_model=worker')
    await page.getByRole('heading', { name: 'Schedule detail' }).waitFor()
    await expect(page.locator('.schedule-row')).toHaveCount(3)

    await page.evaluate(() => {
      const input = document.querySelector<HTMLInputElement>('#weight-0')
      if (!input) throw new Error('weight input is missing')
      input.value = '61'
      input.dispatchEvent(new Event('input', { bubbles: true }))
      window.history.pushState({}, '', '/schedule?schedule_model=worker-b')
      window.dispatchEvent(new PopStateEvent('popstate'))
    })

    await expect(page).toHaveURL(/schedule\?schedule_model=worker-b(?:&|$)/u)
    await expect(
      page.locator('.schedule-panel .app-select__trigger[aria-label="External model"]'),
    ).toContainText('worker-b')
    await expect(page.locator('#weight-0')).toHaveValue('')
  })

  test('a superseded pending model change cannot send old drafts to the new model', async ({
    page,
  }) => {
    const patches = await installScheduleEditingRoutes(page)
    await page.goto('/schedule?schedule_model=worker')
    await page.getByRole('heading', { name: 'Schedule detail' }).waitFor()
    await expect(page.locator('.schedule-row')).toHaveCount(3)
    await page.locator('#weight-0').fill('61')

    await page.locator('.schedule-panel .app-select__trigger[aria-label="External model"]').click()
    const modelSelection = page.locator('.app-select__item[data-value="worker-b"]').click()
    await page.evaluate(() => {
      window.history.pushState({}, '', '/schedule?schedule_model=worker-b&schedule_row=2:entry-2')
      window.dispatchEvent(new PopStateEvent('popstate'))
    })
    await modelSelection
    await expect(page).toHaveURL(/schedule\?schedule_model=worker-b(?:&|$)/u)
    await expect(
      page.locator('.schedule-panel .app-select__trigger[aria-label="External model"]'),
    ).toContainText('worker-b')
    await expect(page.locator('#weight-0')).toHaveValue('')
    expect(patches).toHaveLength(0)
  })
})
