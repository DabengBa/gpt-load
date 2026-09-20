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

function entry(entryID: string, modelID: string, weight: number, priority = 1) {
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
        entries: [entry(`entry-1${suffix}`, `model-a${suffix}`, 50)],
      },
      {
        group_id: 2,
        group_name: `secondary group${suffix}`,
        channel_id: `channel-2${suffix}`,
        enabled: true,
        request_count: 20,
        success_rate: 1,
        entries: [entry(`entry-2${suffix}`, `model-b${suffix}`, 50)],
      },
    ],
  }
}

type PatchOutcome = 'success' | 'error' | 'conflict'

async function installScheduleEditingRoutes(
  page: Page,
  patchOutcome: PatchOutcome = 'success',
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
        await route.fulfill(
          response(detailForModel(url.searchParams.get('external_model') ?? 'worker')),
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
  async function editBothGroups(page: Page): Promise<void> {
    const firstWeight = page.locator('#weight-0')
    const secondWeight = page.locator('#weight-1')
    await firstWeight.fill('61')
    await secondWeight.fill('72')
    await expect(firstWeight).toHaveValue('61')
    await expect(secondWeight).toHaveValue('72')
  }

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

  test('keeps drafts visible after a revision conflict', async ({ page }) => {
    const patches = await installScheduleEditingRoutes(page, 'conflict')
    await openSchedule(page)
    await editBothGroups(page)

    await page.getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => patches.length).toBe(1)
    await expect(page.locator('#weight-0')).toHaveValue('61')
    await expect(page.locator('#weight-1')).toHaveValue('72')
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

  test('does not restore replace drafts into an externally navigated model context', async ({ page }) => {
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
    await expect(page.locator('.schedule-panel .app-select__trigger[aria-label="External model"]'))
      .toContainText('worker-b')
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
    await expect(page.locator('.schedule-panel .app-select__trigger[aria-label="External model"]'))
      .toContainText('worker-b')
    await expect(page.locator('#weight-0')).toHaveValue('')
    expect(patches).toHaveLength(0)
  })
})
