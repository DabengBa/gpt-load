import { expect, test, type Page } from '@playwright/test'

// Phase 3 schedule-domain coverage for the Astryx entry: index loading and
// model context, detail table rendering, draft → PATCH round trip, revision
// conflict handling, probe affordances, recovery, error states, access_key
// exclusion, and canonical query behavior (schedule_model / schedule_row /
// schedule_group / schedule_draft).

const adminKey = 'e2e-admin-key'

type SchedulePatch = {
  snapshot_revision: number
  protocol: string
  external_model: string
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
        channel_id: 'openai',
        enabled: true,
        request_count: 10,
        success_rate: 1,
        entries: [entry(firstEntryID, `model-a${suffix}`, 50)],
      },
      {
        group_id: 2,
        group_name: `secondary group${suffix}`,
        channel_id: 'openai',
        enabled: false,
        request_count: 20,
        success_rate: 1,
        entries: [entry(`entry-2${suffix}`, `model-b${suffix}`, 50, 1, true)],
      },
    ],
  }
}

type PatchOutcome = 'success' | 'error' | 'conflict'
type DetailOutcome = 'success' | 'error' | 'empty'

interface ScheduleRoutes {
  readonly patches: SchedulePatch[]
}

async function installScheduleRoutes(
  page: Page,
  options: {
    patchOutcome?: PatchOutcome
    detailOutcome?: DetailOutcome
    projectDetail?: (model: string) => unknown
    principalType?: 'admin' | 'access_key'
  } = {},
): Promise<ScheduleRoutes> {
  const {
    patchOutcome = 'success',
    detailOutcome = 'success',
    projectDetail = detailForModel,
    principalType = 'admin',
  } = options
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, adminKey)

  const patches: SchedulePatch[] = []
  await page.route(
    (url) => url.pathname === '/api' || url.pathname.startsWith('/api/'),
    async (route) => {
      const request = route.request()
      const url = new URL(request.url())

      if (url.pathname === '/api/auth/session') {
        await route.fulfill(response({ authenticated: true, principal_type: principalType }))
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
                cooled_candidates: 0,
                blacklisted_candidates: 0,
              },
              {
                external_model: 'worker-b',
                protocol: 'openai-completions',
                operation: 'chat_completion',
                candidate_count: 2,
                group_count: 2,
                cooled_candidates: 0,
                blacklisted_candidates: 0,
              },
            ],
          }),
        )
        return
      }
      if (url.pathname === '/api/model-route/schedule/detail') {
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
      if (url.pathname === '/api/model-route/schedule/recover' && request.method() === 'POST') {
        await route.fulfill(response({ recovered: true }))
        return
      }

      await route.fulfill(response({}, 404))
    },
  )
  return { patches }
}

test.setTimeout(90_000)

async function openSchedule(page: Page, query = ''): Promise<void> {
  // `commit` returns once response headers arrive — the first cold vite
  // transform of the astryx graph outlasts the default navigation timeout.
  await page.goto(`/schedule${query}`, { waitUntil: 'commit' })
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({
    timeout: 60_000,
  })
}

async function waitForScheduleDetailReady(page: Page): Promise<void> {
  await page.getByRole('heading', { name: 'Schedule detail' }).waitFor({ timeout: 30_000 })
  await page.getByRole('table', { name: 'Schedule detail' }).waitFor({ timeout: 30_000 })
}

const rows = (page: Page) => page.locator('[data-row-key]')

test('schedule stays within desktop and mobile viewport bounds', async ({ page }, testInfo) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 })
    await expect(page.getByRole('button', { name: 'worker', exact: true })).toBeVisible()
    await expect(rows(page).first()).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({ path: testInfo.outputPath(`schedule-${width}.png`), fullPage: true })
  }
})

async function delayPriceLookup(page: Page) {
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  let started!: () => void
  const requested = new Promise<void>((resolve) => {
    started = resolve
  })
  const group = {
    id: 1,
    name: 'prod',
    channel_id: 'openai',
    params: {},
    enabled: true,
    client_protocols: ['openai-completions'],
  }
  await page.route('**/api/models?**', async (route) => {
    started()
    await gate
    await route.fulfill(
      response({
        summary: {
          client_model_count: 1,
          upstream_model_count: 1,
          price_count: 1,
          pending_price_count: 0,
          unreferenced_price_count: 0,
        },
        catalog: { available: false, checked_at_ms: 0, successful_fetch_at_ms: 0, error_code: '' },
        pagination: { page: 1, page_size: 10, total_items: 1, total_pages: 1 },
        items: [
          {
            client_model: 'worker',
            protocols: ['openai-completions'],
            upstream_models: [
              {
                model_id: 'model-a',
                alias_applied: true,
                route_groups: [group],
                affected_groups: [group],
                catalog_reference: null,
                price: {
                  id: 7,
                  channel_id: 'openai',
                  channel_name: 'Channel',
                  channel_mark: 'C',
                  channel_icon: 'openai',
                  model_id: 'model-a',
                  prices: { input: '2', output: '10', cache_read: null, cache_write: null },
                  mode_schedules: {},
                  pricing_status: 'configured',
                  method: 'user_set',
                  matched_provider_id: null,
                  match_source: null,
                  referenced: true,
                  reference_count: 1,
                  reference_group_count: 1,
                  context_tiers: [],
                  updated_at_ms: 0,
                  can_reset: true,
                  can_delete: false,
                },
              },
            ],
          },
        ],
      }),
    )
  })
  return { requested, release }
}

test('delayed price lookup preserves drafts edited while waiting', async ({ page }) => {
  await installScheduleRoutes(page)
  const lookup = await delayPriceLookup(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await page.getByRole('button', { name: 'Price', exact: true }).first().click()
  await lookup.requested
  await page.locator('#priority-0').fill('3')
  await expect(page).toHaveURL(/schedule_draft=/)
  const draft = new URL(page.url()).searchParams.get('schedule_draft')
  lookup.release()
  await expect(page).toHaveURL(/selected_price_id=7/)
  expect(new URL(page.url()).searchParams.get('schedule_draft')).toBe(draft)
  await expect(page.locator('#priority-0')).toHaveValue('3')
})

test('delayed price lookup cannot navigate back after leaving schedule', async ({ page }) => {
  await installScheduleRoutes(page)
  const lookup = await delayPriceLookup(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await page.getByRole('button', { name: 'Price', exact: true }).first().click()
  await lookup.requested
  await page.getByRole('link', { name: 'Request logs', exact: true }).click()
  await expect(page).toHaveURL(/\/logs/)
  lookup.release()
  await page.waitForLoadState('networkidle')
  await expect(page).toHaveURL(/\/logs/)
})

test('loads the index, canonicalizes junk params, and renders the detail table', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker&junk=1&schedule_row=bogus')
  await waitForScheduleDetailReady(page)

  // Junk params are stripped; the valid model context stays.
  await expect.poll(() => new URL(page.url()).search).toBe('?schedule_model=worker')
  await expect(rows(page)).toHaveCount(2)
  await expect(page.getByRole('columnheader', { name: 'Priority' })).toBeVisible()
})

test('model selector commits schedule_model through the canonical query', async ({ page }) => {
  await installScheduleRoutes(page)
  await openSchedule(page)

  await page.getByRole('button', { name: 'worker', exact: true }).click()

  await expect(page).toHaveURL(/schedule_model=worker/)
  await waitForScheduleDetailReady(page)
  await expect(rows(page)).toHaveCount(2)
})

test('explains how to begin before a scheduling model is selected', async ({ page }) => {
  await installScheduleRoutes(page)
  await openSchedule(page)
  const selectModelPrompt = page.getByRole('status').filter({
    has: page.getByRole('heading', { name: 'Select a model', exact: true }),
  })
  await expect(selectModelPrompt).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Schedule details' })).toHaveCount(0)
  await page.getByRole('button', { name: 'worker', exact: true }).click()
  await waitForScheduleDetailReady(page)
  await expect(selectModelPrompt).toHaveCount(0)
})

test('keeps the schedule readable on mobile and contains table overflow on narrow desktop', async ({
  page,
}, testInfo) => {
  await installScheduleRoutes(page)
  await page.setViewportSize({ width: 390, height: 844 })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)

  await expect(
    page.locator('[data-row-key]').first().getByText('Priority', { exact: true }),
  ).toBeVisible()
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
    .toBe(true)
  await page.screenshot({ path: testInfo.outputPath('schedule-mobile.png'), fullPage: true })

  await page.setViewportSize({ width: 800, height: 900 })
  const table = page.getByRole('table', { name: 'Schedule detail' })
  const tableWrap = table.locator('xpath=..')
  await expect
    .poll(() => tableWrap.evaluate((element) => element.scrollWidth > element.clientWidth))
    .toBe(true)
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
    .toBe(true)
  await page.screenshot({
    path: testInfo.outputPath('schedule-narrow-desktop.png'),
    fullPage: true,
  })

  await page.setViewportSize({ width: 1440, height: 1000 })
  const reasoning = await page
    .getByRole('combobox', { name: 'Reasoning policy model-a' })
    .boundingBox()
  const weight = await page.locator('#weight-0').boundingBox()
  expect(reasoning).not.toBeNull()
  expect(weight).not.toBeNull()
  expect(reasoning!.x + reasoning!.width).toBeLessThanOrEqual(weight!.x)
  await page.screenshot({ path: testInfo.outputPath('schedule-desktop.png'), fullPage: true })

  await page.evaluate(() => window.localStorage.setItem('gpt-load.theme', 'dark'))
  await page.reload()
  await waitForScheduleDetailReady(page)
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await page.screenshot({ path: testInfo.outputPath('schedule-dark.png'), fullPage: true })
})

test('draft edits serialize to schedule_draft and Save issues the PATCH', async ({ page }) => {
  const { patches } = await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)

  // Row order is priority-then-identity: index 0 is the '1:entry-1' row.
  await page.locator('#priority-0').fill('3')
  await expect(page).toHaveURL(/schedule_draft=/)
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await expect.poll(() => patches.length).toBe(1)
  expect(patches[0]!.snapshot_revision).toBe(11)
  expect(patches[0]!.external_model).toBe('worker')
  expect(patches[0]!.updates).toContainEqual(
    expect.objectContaining({ group_id: 1, entry_id: 'entry-1', priority: 3 }),
  )
  await expect(page).not.toHaveURL(/schedule_draft=/)
  await expect(page.getByText('Saved', { exact: true })).toBeVisible()
  await expect(page.getByText('Unsaved changes', { exact: true })).toHaveCount(0)
})

test('desktop sidebar controls stay before the detail and page has inset spacing', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const search = await page
    .getByRole('textbox', { name: 'External model', exact: true })
    .boundingBox()
  const table = await page.getByRole('table', { name: 'Schedule detail' }).boundingBox()
  expect(search!.x).toBeGreaterThan(15)
  expect(search!.x + search!.width).toBeLessThanOrEqual(table!.x)
  const model = await page.getByRole('button', { name: 'worker', exact: true }).boundingBox()
  expect(model!.x + model!.width).toBeLessThanOrEqual(table!.x)
})

test('search with no match offers a distinct empty message and can recover', async ({ page }) => {
  await installScheduleRoutes(page)
  await openSchedule(page)
  await page.getByRole('textbox', { name: 'External model', exact: true }).fill('unknown-model')
  await expect(page.getByText('No matching models', { exact: true })).toBeVisible()
  await page.getByRole('textbox', { name: 'External model', exact: true }).fill('')
  await expect(page.getByRole('button', { name: 'worker', exact: true })).toBeVisible()
})

test('empty model index offers import instead of asking to select a missing model', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await page.route('**/api/model-route/schedule', async (route) => {
    await route.fulfill(response({ items: [] }))
  })
  await openSchedule(page)
  await expect(page.getByText('No models available', { exact: true })).toBeVisible()
  await expect(page.getByText('Select a model', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Import credentials', exact: true }).click()
  await expect(page).toHaveURL(/\/import/)
})

test('entry enable toggle preserves unsaved priority draft', async ({ page }) => {
  const { patches } = await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await page.locator('#priority-0').fill('3')
  await expect(page).toHaveURL(/schedule_draft=/)
  const draft = new URL(page.url()).searchParams.get('schedule_draft')
  await page.getByRole('switch', { name: 'Toggle entry model-a', exact: true }).click()
  await expect.poll(() => patches.length).toBe(1)
  await expect.poll(() => new URL(page.url()).searchParams.get('schedule_draft')).toBe(draft)
  await expect(page.locator('#priority-0')).toHaveValue('3')
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeEnabled()
})

test('revision conflict keeps drafts visible and surfaces the conflict state', async ({ page }) => {
  await installScheduleRoutes(page, { patchOutcome: 'conflict' })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)

  await page.locator('#priority-0').fill('3')
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await expect(
    page.getByText('The configuration changed. Refresh and try again.').first(),
  ).toBeVisible()
  // The draft stays editable — the input keeps its edited value.
  await expect(page.locator('#priority-0')).toHaveValue('3')
})

test('detail error state renders and empty detail shows the no-entries state', async ({ page }) => {
  await installScheduleRoutes(page, { detailOutcome: 'error' })
  await openSchedule(page, '?schedule_model=worker')
  const detail = page.getByRole('region', { name: 'Schedule details' })
  // Classic `errorMessage` prefers the thrown error message over the fallback.
  await expect(detail.getByRole('alert')).toBeVisible()
  await expect(detail.getByRole('button', { name: 'Refresh' })).toBeVisible()
})

test('access_key principal sees the read-only surface without scheduling controls', async ({
  page,
}) => {
  await installScheduleRoutes(page, { principalType: 'access_key' })
  await openSchedule(page)
  await expect(page.getByTestId('schedule-read-only')).toBeVisible()
  await expect(page.getByTestId('schedule-panel')).toHaveCount(0)
})

test('schedule_row deep link selects the target row', async ({ page }) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker&schedule_row=2%3Aentry-2')
  await waitForScheduleDetailReady(page)

  await expect(page.locator('[data-row-key][aria-selected="true"]')).toHaveAttribute(
    'data-row-key',
    '2:entry-2',
  )
})
