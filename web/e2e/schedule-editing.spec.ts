import { expect, test, type Page } from '@playwright/test'

const adminKey = 'e2e-admin-key'

type SchedulePatch = {
  snapshot_revision: number
  protocol: string
  external_model: string
  access_key_id: number
  operation: string
  group_updates?: Array<{
    group_id: number
    reasoning_effort_default: string | null
  }>
  updates: Array<{
    group_id: number
    entry_id: string
    weight?: number | null
    priority?: number | null
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
    circuit_breaker: {
      configured: { blacklist_threshold: null, cooldown_seconds: null },
      effective: { blacklist_threshold: 3, cooldown_seconds: 60 },
      sources: { blacklist_threshold: 'default', cooldown_seconds: 'default' },
    },
    reasoning: {
      configured: override ? 'max' : (null as string | null),
      effective: override ? 'max' : ('medium' as string | null),
      source: override ? 'entry' : 'group',
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
    reason_code: null,
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
        reasoning_effort_default: 'medium' as string | null,
        request_count: 10,
        success_rate: 1,
        reasoning_entries: [entry(firstEntryID, `model-a${suffix}`, 50)].map(
          ({ entry_id, model_id, reasoning }) => ({ entry_id, model_id, reasoning }),
        ),
        entries: [entry(firstEntryID, `model-a${suffix}`, 50)],
      },
      {
        group_id: 2,
        group_name: `secondary group${suffix}`,
        channel_id: `channel-2${suffix}`,
        enabled: false,
        reasoning_effort_default: 'medium',
        request_count: 20,
        success_rate: 1,
        reasoning_entries: [entry(`entry-2${suffix}`, `model-b${suffix}`, 50, 1, true)].map(
          ({ entry_id, model_id, reasoning }) => ({ entry_id, model_id, reasoning }),
        ),
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

test.describe('schedule editing', () => {
  function geminiDetail(imageOverride = false, showImage = false) {
    const detail = detailForModel('worker')
    const flash = entry('flash-entry', 'unlisted-model', 50)
    const image = entry('image-entry', 'gemini-3.1-flash-lite-image', 50)
    for (const item of [flash, image]) {
      item.reasoning.configured = null
      item.reasoning.effective = null
      item.reasoning.source = 'provider_default'
    }
    if (imageOverride) {
      image.reasoning.configured = 'high'
      image.reasoning.effective = 'high'
      image.reasoning.source = 'entry'
    }
    const group = detail.groups[0]!
    group.reasoning_effort_default = null
    group.entries = showImage
      ? [flash, image]
      : [{ ...flash, configured_share: 1, effective_share: 1 }]
    Object.assign(group, {
      reasoning_entries: [flash, image].map(({ entry_id, model_id, reasoning }) => ({
        entry_id,
        model_id,
        reasoning,
      })),
    })
    detail.groups = [group]
    return detail
  }

  test('offers every effort for unlisted models and previews the entire group', async ({
    page,
  }) => {
    const patches = await installScheduleEditingRoutes(page, 'success', 'success', () =>
      geminiDetail(),
    )
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('/schedule?schedule_model=worker')
    await expect(page.locator('.schedule-row')).toHaveCount(2)
    await page.getByRole('combobox', { name: 'Group default primary group' }).click()
    for (const effort of ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']) {
      await expect(page.locator(`.app-select__item[data-value="${effort}"]`)).toBeVisible()
    }
    await page.locator('.app-select__item[data-value="max"]').click()
    await expect(page.locator('.schedule-reasoning-preview')).toContainText(
      'gemini-3.1-flash-lite-image',
    )
    await expect(page.locator('.schedule-reasoning-preview')).toContainText('unlisted-model')
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      ),
    ).toBeLessThanOrEqual(1)
    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]?.group_updates).toEqual([{ group_id: 1, reasoning_effort_default: 'max' }])
  })

  test('clearing an image override keeps its group default editable and saves both', async ({
    page,
  }) => {
    const patches = await installScheduleEditingRoutes(page, 'success', 'success', () =>
      geminiDetail(true, true),
    )
    await openSchedule(page)
    await page.getByRole('combobox', { name: 'Group default primary group' }).click()
    await page.locator('.app-select__item[data-value="low"]').click()
    await expect(page.getByRole('button', { name: 'Save' })).toBeEnabled()
    await page.getByRole('combobox', { name: 'Entry override gemini-3.1-flash-lite-image' }).click()
    await page.locator('.app-select__item[data-value=""]').click()
    await expect(page.locator('.schedule-reasoning-preview')).toContainText(
      'gemini-3.1-flash-lite-image',
    )
    await expect(page.locator('.schedule-reasoning-preview')).toContainText(
      'gemini-3.1-flash-lite-image: low',
    )
    await expect(page.getByRole('button', { name: 'Save' })).toBeEnabled()
    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]?.group_updates).toEqual([{ group_id: 1, reasoning_effort_default: 'low' }])
    expect(patches[0]?.updates).toEqual([
      { group_id: 1, entry_id: 'image-entry', reasoning_effort: null },
    ])
  })

  test('saves a group default with an image override in the same draft', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page, 'success', 'success', () =>
      geminiDetail(false, true),
    )
    await openSchedule(page)
    await page.getByRole('combobox', { name: 'Entry override gemini-3.1-flash-lite-image' }).click()
    await page.locator('.app-select__item[data-value="high"]').click()
    await page.getByRole('combobox', { name: 'Group default primary group' }).click()
    await page.locator('.app-select__item[data-value="low"]').click()
    await expect(page.locator('.schedule-reasoning-preview')).toContainText(
      'gemini-3.1-flash-lite-image: high',
    )
    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]).toMatchObject({
      group_updates: [{ group_id: 1, reasoning_effort_default: 'low' }],
      updates: [{ group_id: 1, entry_id: 'image-entry', reasoning_effort: 'high' }],
    })
  })

  test('previews a hidden image override when editing group defaults', async ({ page }) => {
    await installScheduleEditingRoutes(page, 'success', 'success', () => geminiDetail(true))
    await page.goto('/schedule?schedule_model=worker')
    await expect(page.locator('.schedule-row')).toHaveCount(2)
    await page.getByRole('combobox', { name: 'Group default primary group' }).click()
    await page.locator('.app-select__item[data-value="low"]').click()
    await expect(page.locator('.schedule-reasoning-preview')).toContainText(
      'gemini-3.1-flash-lite-image: high',
    )
    await expect(page.getByRole('button', { name: 'Save' })).toBeEnabled()
  })

  async function editBothGroups(page: Page): Promise<void> {
    const firstWeight = page.locator('#weight-0')
    const secondWeight = page.locator('#weight-1')
    await firstWeight.fill('61')
    await secondWeight.fill('72')
    await expect(firstWeight).toHaveValue('61')
    await expect(secondWeight).toHaveValue('72')
    await page.getByRole('combobox', { name: 'Group default secondary group' }).click()
    await page.locator('.app-select__item[data-value=""]').click()
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

  test('submits group default and entry override in one revisioned save', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page)
    await openSchedule(page)

    await page.getByRole('combobox', { name: 'Group default primary group' }).click()
    await page.locator('.app-select__item[data-value="high"]').click()
    await page.getByRole('combobox', { name: 'Entry override model-a' }).click()
    await page.locator('.app-select__item[data-value="low"]').click()
    await page.getByRole('button', { name: 'Save' }).click()

    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]).toMatchObject({
      snapshot_revision: 11,
      group_updates: [{ group_id: 1, reasoning_effort_default: 'high' }],
      updates: [{ group_id: 1, entry_id: 'entry-1', reasoning_effort: 'low' }],
    })
  })

  test('clears configured group and entry reasoning with explicit nulls', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page)
    await openSchedule(page)

    await page.getByRole('combobox', { name: 'Group default secondary group' }).click()
    await page.locator('.app-select__item[data-value=""]').click()
    await page.getByRole('combobox', { name: 'Entry override model-b' }).click()
    await page.locator('.app-select__item[data-value=""]').click()
    await page.getByRole('button', { name: 'Save' }).click()

    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]).toMatchObject({
      group_updates: [{ group_id: 2, reasoning_effort_default: null }],
      updates: [{ group_id: 2, entry_id: 'entry-2', reasoning_effort: null }],
    })
  })

  test('edits an existing max entry override without model catalogue data', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page)
    await openSchedule(page)

    const override = page.getByRole('combobox', { name: 'Entry override model-b' })
    await expect(override).toContainText('max')
    await expect(override).toBeEnabled()
    await override.click()
    await page.locator('.app-select__item[data-value="xhigh"]').click()
    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    expect(patches[0]?.updates).toEqual([
      { group_id: 2, entry_id: 'entry-2', reasoning_effort: 'xhigh' },
    ])
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
    await page.getByRole('combobox', { name: 'Group default primary group-b' }).focus()
    await expect(
      page.getByRole('combobox', { name: 'Group default primary group-b' }),
    ).toBeFocused()
    const pageOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    )
    expect(pageOverflow).toBeLessThanOrEqual(1)
  })

  test('keeps drafts visible after a revision conflict', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page, 'conflict')
    await openSchedule(page)
    await editBothGroups(page)
    await page.getByRole('combobox', { name: 'Group default primary group' }).click()
    await page.locator('.app-select__item[data-value="high"]').click()
    await page.getByRole('combobox', { name: 'Entry override model-a' }).click()
    await page.locator('.app-select__item[data-value="low"]').click()

    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    await expect(page.locator('#weight-0')).toHaveValue('61')
    await expect(page.locator('#weight-1')).toHaveValue('72')
    await expect(page.getByRole('combobox', { name: 'Group default primary group' })).toContainText(
      'high',
    )
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
