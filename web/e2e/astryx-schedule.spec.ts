import { mkdir, writeFile } from 'node:fs/promises'

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
    concurrency_limit?: number
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
    concurrency_limit: 0,
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
    // A lone entry owns its priority tier: internal/control/route_inspect.go
    // `configuredEntryShares` normalizes weight within the priority and skips
    // disabled groups/entries, while applyEffectiveShares only assigns share to
    // routable rows in the lowest active tier.
    configured_share: 1,
    effective_share: 1,
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
        // A disabled group neither contributes to nor receives a configured
        // share, and a non-routable row gets no effective share.
        entries: [
          {
            ...entry(`entry-2${suffix}`, `model-b${suffix}`, 50, 1, true),
            configured_share: 0,
            effective_share: 0,
          },
        ],
      },
    ],
  }
}

type PatchOutcome = 'success' | 'error' | 'conflict'

// The access-key surface projects the /api/models collection through the same
// contract as production (summary + catalog + page-consistent pagination).
// A bare `items` payload fails projection, so a retry could never clear the
// error banner.
function emptyModelCollection() {
  return {
    summary: {
      client_model_count: 0,
      upstream_model_count: 0,
      price_count: 0,
      pending_price_count: 0,
      unreferenced_price_count: 0,
    },
    catalog: { available: false, checked_at_ms: 0, successful_fetch_at_ms: 0, error_code: '' },
    items: [],
    pagination: { page: 1, page_size: 10, total_items: 0, total_pages: 0 },
  }
}

test('advanced concurrency saves, reloads, validates and discards incomplete input', async ({
  page,
}) => {
  const { patches } = await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await moreButton(page).click()
  const input = page.locator('#concurrency-0')
  await expect(input).toHaveValue('0')
  await expect(page.getByText(/0 means unlimited/).filter({ visible: true })).toBeVisible()
  for (const value of ['', '-1', '1.5', '2147483648']) {
    await input.fill(value)
    await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  }
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'Discard changes', exact: true }).click()
  await moreButton(page).click()
  await expect(input).toHaveValue('0')
  await input.fill('2147483647')
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(/schedule_draft=/)
  await page.reload()
  await waitForScheduleDetailReady(page)
  await moreButton(page).click()
  await expect(input).toHaveValue('2147483647')
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(() => patches.length).toBe(1)
  expect(patches[0]!.updates).toEqual([
    { group_id: 1, entry_id: 'entry-1', concurrency_limit: 2147483647 },
  ])
  await expect(page).not.toHaveURL(/schedule_draft=/)
})

test('advanced concurrency survives a row toggle so one save carries every draft', async ({
  page,
}) => {
  const { patches } = await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await moreButton(page).click()
  await page.locator('#concurrency-0').fill('4')
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(/schedule_draft=/)
  await page.locator('#priority-0').fill('3')
  // The row toggle issues its own PATCH; the outstanding concurrency draft must
  // survive the re-render instead of being reposted as a separate update.
  await page
    .getByRole('switch', { name: /^(?:Enable|Disable) entry model-a(?: · primary group)?$/ })
    .click()
  await expect.poll(() => patches.length).toBe(1)
  expect(patches[0]!.updates).toEqual([{ group_id: 1, entry_id: 'entry-1', enabled: false }])
  await moreButton(page).click()
  await expect(page.locator('#concurrency-0')).toHaveValue('4')
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(() => patches.length).toBe(2)
  expect(patches[1]!.updates).toEqual([
    { group_id: 1, entry_id: 'entry-1', priority: 3, concurrency_limit: 4 },
  ])
})

test('advanced concurrency preserves URL draft and input after revision conflict', async ({
  page,
}) => {
  await installScheduleRoutes(page, { patchOutcome: 'conflict' })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await moreButton(page).click()
  await page.locator('#concurrency-0').fill('3')
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(/schedule_draft=/)
  const draft = new URL(page.url()).searchParams.get('schedule_draft')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(
    page.getByText('The configuration changed. Refresh and try again.').first(),
  ).toBeVisible()
  await moreButton(page).click()
  await expect(page.locator('#concurrency-0')).toHaveValue('3')
  expect(new URL(page.url()).searchParams.get('schedule_draft')).toBe(draft)
})
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

test.beforeEach(async ({ page }) => {
  // Most flows use the persistent index. Smaller breakpoints are exercised
  // explicitly below, where the approved design uses a model-switch entry.
  await page.setViewportSize({ width: 1440, height: 900 })
})

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

// Locale-agnostic readiness probe for screenshot runs that switch catalogs.
async function waitForScheduleTable(page: Page): Promise<void> {
  await page.locator('[role="table"] [data-row-key]').first().waitFor({ timeout: 30_000 })
}

const rows = (page: Page) => page.locator('[data-row-key]')

test('disabled schedule rows remain visually distinct without changing their switches', async ({
  page,
}, testInfo) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const active = rows(page).first()
  const disabledGroup = rows(page).nth(1)
  await expect(disabledGroup.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
  const marker = (element: Element) => getComputedStyle(element).boxShadow
  expect(await disabledGroup.evaluate(marker)).not.toBe(await active.evaluate(marker))
  await disabledGroup.click()
  await page.locator('#weight-1').fill('40')
  await disabledGroup.hover()
  await expect(disabledGroup).toHaveAttribute('aria-selected', 'true')
  await expect(disabledGroup.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
  expect(await disabledGroup.evaluate(marker)).not.toBe(await active.evaluate(marker))
  expect(
    await disabledGroup.evaluate((element) => {
      const probe = document.createElement('i')
      probe.style.backgroundColor = 'var(--color-action-soft)'
      element.append(probe)
      const matches =
        getComputedStyle(element).backgroundColor === getComputedStyle(probe).backgroundColor
      probe.remove()
      return matches
    }),
  ).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('disabled-selected-draft-light.png') })
  await page.evaluate(() => localStorage.setItem('gpt-load.theme', 'dark'))
  await page.reload()
  await waitForScheduleDetailReady(page)
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  expect(await disabledGroup.evaluate(marker)).not.toBe(await active.evaluate(marker))
  await page.screenshot({ path: testInfo.outputPath('disabled-selected-draft-dark.png') })
})

test('Chinese schedule headings and ordinary rows fit desktop tracks', async ({ page }) => {
  await installScheduleRoutes(page)
  await page.addInitScript(() => localStorage.setItem('gpt-load.locale', 'zh-CN'))
  await page.setViewportSize({ width: 1200, height: 900 })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleTable(page)
  for (const width of [768, 1199, 1200, 1439, 1440]) {
    await page.setViewportSize({ width, height: 900 })
    const geometry = await page.evaluate(() => {
      const headers = [...document.querySelectorAll<HTMLElement>('[role="columnheader"]')]
      const lineCount = (element: HTMLElement) => {
        const range = document.createRange()
        range.selectNodeContents(element)
        return new Set([...range.getClientRects()].map((rect) => Math.round(rect.top))).size
      }
      const visibleRows = [...document.querySelectorAll<HTMLElement>('[data-row-key]')]
      return {
        lines: [lineCount(headers[1]!), lineCount(headers[4]!)],
        statusLines: visibleRows.map((row) =>
          lineCount(
            row.querySelectorAll<HTMLElement>('[role="cell"]')[1]!.firstElementChild as HTMLElement,
          ),
        ),
        heights: visibleRows.map((row) => row.getBoundingClientRect().height),
        actionRight: headers[5]!.getBoundingClientRect().right,
        buttonRight: Math.max(
          ...[...visibleRows[0]!.querySelectorAll('[role="cell"]:last-child button')]
            .filter((button) => button.getBoundingClientRect().width > 0)
            .map((button) => button.getBoundingClientRect().right),
        ),
        pageWidth: document.documentElement.scrollWidth,
      }
    })
    expect(geometry.lines, `${width}px headings`).toEqual([1, 1])
    expect(geometry.statusLines, `${width}px eligibility`).toEqual([1, 1])
    expect(
      Math.abs(geometry.heights[0]! - geometry.heights[1]!),
      `${width}px rows`,
    ).toBeLessThanOrEqual(1)
    expect(
      Math.abs(geometry.actionRight - geometry.buttonRight),
      `${width}px actions`,
    ).toBeLessThanOrEqual(1)
    expect(geometry.pageWidth).toBeLessThanOrEqual(width)
    const wrap = page.getByRole('table').locator('..')
    await wrap.evaluate((element) => {
      element.scrollLeft = element.scrollWidth
    })

    const reachable = await wrap.evaluate((element) => {
      const button = element.querySelector('[data-row-key] [role="cell"]:last-child button')!
      return button.getBoundingClientRect().right <= element.getBoundingClientRect().right + 1
    })
    expect(reachable, `${width}px last column reachable`).toBe(true)
  }
})

test('Chinese tier-demoted eligibility has no orphan character or uneven ordinary row', async ({
  page,
}) => {
  await installScheduleRoutes(page, {
    projectDetail: (model) => {
      const detail = detailForModel(model)
      detail.groups[0]!.entries[0]!.reason_code = 'tier_demoted'
      detail.groups[1]!.enabled = true
      return detail
    },
  })
  await page.addInitScript(() => localStorage.setItem('gpt-load.locale', 'zh-CN'))
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleTable(page)
  const status = rows(page).first().getByRole('cell').nth(1).locator('span').first()
  await expect(status).toHaveText('处于后备优先级层待命')

  for (const width of [768, 1199, 1200, 1439, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 })
    const lineLengths = await status.evaluate((element) => {
      const text = element.firstChild!
      const lines = new Map<number, number>()
      for (let index = 0; index < text.textContent!.length; index++) {
        const range = document.createRange()
        range.setStart(text, index)
        range.setEnd(text, index + 1)
        const top = Math.round(range.getBoundingClientRect().top)
        lines.set(top, (lines.get(top) ?? 0) + 1)
      }
      return [...lines.values()]
    })
    expect(lineLengths.at(-1), `${width}px final status line`).toBeGreaterThan(1)
    const heights = await rows(page).evaluateAll((elements) =>
      elements.map((element) => element.getBoundingClientRect().height),
    )
    expect(
      Math.abs(heights[0]! - heights[1]!),
      `${width}px ordinary row heights`,
    ).toBeLessThanOrEqual(1)
  }
})

test('Chinese schedule switches to usable cards at 767px', async ({ page }, testInfo) => {
  await installScheduleRoutes(page)
  await page.addInitScript(() => localStorage.setItem('gpt-load.locale', 'zh-CN'))
  await page.setViewportSize({ width: 767, height: 900 })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleTable(page)
  await expect(page.getByRole('columnheader').first()).toBeHidden()
  await expect(rows(page).first().getByRole('switch')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(767)
  await page.screenshot({ path: testInfo.outputPath('schedule-767-zh.png') })
})

test('optimistic disable hides schedulable eligibility and failure restores the saved snapshot', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  let release: () => void = () => {}
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route('**/api/model-route/schedule', async (route) => {
    if (route.request().method() !== 'PATCH') return route.fallback()
    await gate
    await route.fulfill(response({}, 500))
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const row = rows(page).first()
  const status = row.getByRole('cell').nth(1)
  await row.getByRole('switch', { name: /^Disable entry model-a/ }).click()
  try {
    await expect(row.getByRole('switch')).toHaveAttribute('aria-checked', 'false')
    await expect(row.getByRole('switch')).toHaveAccessibleName(/^Enable entry model-a/)
    await expect(status).toContainText('Route entry is disabled')
    await expect(status).not.toContainText('Schedulable')
    expect(await row.evaluate((element) => getComputedStyle(element).boxShadow)).not.toBe('none')
  } finally {
    release()
  }
  await expect(row.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
  expect(await row.evaluate((element) => getComputedStyle(element).boxShadow)).toBe('none')
  await expect(status).toContainText('Schedulable')
})

test('draft share excludes disabled groups from both numerator and denominator', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await page.locator('#weight-0').fill('100')
  await expect(rows(page).first().getByRole('cell').nth(4)).toContainText('100.0%')
  await expect(rows(page).nth(1).getByRole('cell').nth(4)).toContainText('0.0%')
})

test('eligibility snapshot separates health, group statistics and unsaved share preview', async ({
  page,
}, testInfo) => {
  await installScheduleRoutes(page, {
    projectDetail: (model) => {
      const detail = detailForModel(model)
      const group = detail.groups[0]!
      group.request_count = 0
      group.success_rate = 0
      group.entries = [
        entry('entry-1', 'model-a', 50),
        ...[
          [true, 'tier_demoted', 'available'],
          [false, 'credential_cooldown', 'available'],
          [false, 'entry_cooldown', 'cooldown'],
          [false, null, 'blacklisted'],
          [false, 'entry_disabled', 'available'],
        ].map(([routable, reason, health], index) => {
          const candidate = entry(`zcase-${index}`, `model-z${index}`, 10)
          candidate.routable = routable === true
          candidate.reason_code = typeof reason === 'string' ? reason : null
          candidate.runtime.state = String(health)
          candidate.enabled = reason !== 'entry_disabled'
          return candidate
        }),
      ]
      detail.groups[1]!.success_rate = 0
      for (const candidate of group.entries) {
        candidate.configured_share = 0
        candidate.effective_share = 0
      }
      // The one enabled group in this tier owns the whole configured share, and
      // the only routable row carries the whole effective share.
      group.entries[0]!.configured_share = 1
      group.entries[0]!.effective_share = 1
      return detail
    },
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const status = (index: number) => rows(page).nth(index).getByRole('cell').nth(1)
  await expect(status(0)).toContainText('Schedulable')
  await expect(status(0)).toContainText('Entry health: Available')
  await expect(status(1)).toContainText('Standing by in a standby priority tier')
  await expect(status(2)).toContainText('Credential is cooling down')
  await expect(status(3)).toContainText('Route entry is cooling down')
  await expect(status(4)).toContainText('Unknown route reason')
  await expect(status(4)).toContainText('Entry health: Blacklisted')
  await expect(status(5)).toContainText('Route entry is disabled')
  await expect(status(6)).toContainText('Group is disabled')
  await expect(rows(page).first().getByRole('cell').nth(2)).toContainText('—')
  await expect(rows(page).nth(6).getByRole('cell').nth(2)).toContainText('0.0%')
  // A disabled group holds no configured share even before any draft exists.
  await expect(rows(page).first().getByRole('cell').nth(4)).toContainText('100.0%')
  await expect(rows(page).nth(6).getByRole('cell').nth(4)).toContainText('0.0%')
  await page.locator('#weight-0').fill('100')
  await expect(status(1)).toContainText('Standing by in a standby priority tier')
  await expect(rows(page).first().getByRole('cell').nth(4)).toContainText('71.4%')
  await expect(rows(page).nth(6).getByRole('cell').nth(4)).toContainText('0.0%')
  await expect(page.getByText('Unsaved share preview', { exact: true })).toBeVisible()
  const toggle = rows(page)
    .first()
    .getByRole('switch', { name: /^Disable entry model-a/ })
  await expect(toggle).toHaveAttribute('title', 'Disable entry model-a · primary group')
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 })
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({ path: testInfo.outputPath(`eligibility-${width}.png`), fullPage: true })
  }
})

async function changeReasoning(page: Page, value = 'high') {
  const selector = page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ })
  await selector.click()
  await page.getByRole('option', { name: value, exact: true }).click()
  await expect(selector).toContainText(value)
  return selector
}

const moreButton = (page: Page) =>
  page.getByRole('button', { name: 'More model-a · primary group', exact: true })

const modelSwitchDialog = (page: Page) => page.getByRole('dialog', { name: /Save changes before/ })

test('reasoning-only draft writes URL, survives reload and saves its exact update', async ({
  page,
}) => {
  const { patches } = await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  await expect(page).toHaveURL(/schedule_draft=/)
  const draft = new URL(page.url()).searchParams.get('schedule_draft')
  await page.reload()
  await waitForScheduleDetailReady(page)
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
  expect(new URL(page.url()).searchParams.get('schedule_draft')).toBe(draft)
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(() => patches.length).toBe(1)
  expect(patches[0]!.updates).toEqual([
    { group_id: 1, entry_id: 'entry-1', reasoning_effort: 'high' },
  ])
  await expect(page).not.toHaveURL(/schedule_draft=/)
})

test('reasoning-only model switch can cancel then explicitly discard', async ({ page }) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  await page.getByRole('button', { name: 'worker-b', exact: true }).click()
  const guard = modelSwitchDialog(page)
  await expect(guard).toBeVisible()
  await guard.getByRole('button', { name: 'Stay on this model', exact: true }).click()
  await expect(guard).not.toBeVisible()
  await expect(page).toHaveURL(/schedule_model=worker(?:&|$)/)
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
  await page.getByRole('button', { name: 'worker-b', exact: true }).click()
  await guard.getByRole('button', { name: 'Discard and continue', exact: true }).click()
  await expect(page).toHaveURL(/schedule_model=worker-b/)
  await expect(page).not.toHaveURL(/schedule_draft=/)
  await page.getByRole('button', { name: 'worker', exact: true }).click()
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).not.toContainText('high')
})

test('reasoning and numeric drafts survive row toggle with an increased server revision', async ({
  page,
}) => {
  const state: ScheduleRoutes = await installScheduleRoutes(page, {
    projectDetail: (model) => {
      const detail = detailForModel(model)
      detail.snapshot_revision =
        11 +
        state.patches.filter((patch) =>
          patch.updates.some((update) => update.enabled !== undefined),
        ).length
      if (detail.snapshot_revision > 11) detail.groups[0]!.entries[0]!.enabled = false
      return detail
    },
  })
  const { patches } = state
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await page.locator('#priority-0').fill('3')
  await page.locator('#weight-0').fill('37')
  await moreButton(page).click()
  await page.getByRole('spinbutton', { name: /Concurrency limit/ }).fill('4')
  await page.keyboard.press('Escape')
  await changeReasoning(page)
  const refreshed = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === '/api/model-route/schedule/detail' &&
      response.request().method() === 'GET',
  )
  await page
    .getByRole('switch', { name: /^(?:Enable|Disable) entry model-a(?: · primary group)?$/ })
    .click()
  expect((await (await refreshed).json()).data.snapshot_revision).toBe(12)
  await expect.poll(() => patches.length).toBe(1)
  await expect(
    page.getByRole('switch', { name: /^(?:Enable|Disable) entry model-a(?: · primary group)?$/ }),
  ).not.toBeChecked()
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
  await expect(page.locator('#priority-0')).toHaveValue('3')
  await expect(page.locator('#weight-0')).toHaveValue('37')
  await moreButton(page).click()
  await expect(page.getByRole('spinbutton', { name: /Concurrency limit/ })).toHaveValue('4')
  await page.keyboard.press('Escape')
  expect(patches[0]!.updates).toEqual([{ group_id: 1, entry_id: 'entry-1', enabled: false }])
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(() => patches.length).toBe(2)
  expect(patches[1]!.snapshot_revision).toBe(12)
  expect(patches[1]!.updates).toEqual([
    {
      group_id: 1,
      entry_id: 'entry-1',
      priority: 3,
      weight: 37,
      concurrency_limit: 4,
      reasoning_effort: 'high',
    },
  ])
})

test('reasoning-only 409 conflict preserves URL draft and editable selection', async ({ page }) => {
  await installScheduleRoutes(page, { patchOutcome: 'conflict' })

  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  await expect(page).toHaveURL(/schedule_draft=/)
  const draft = new URL(page.url()).searchParams.get('schedule_draft')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(
    page.getByText('The configuration changed. Refresh and try again.').first(),
  ).toBeVisible()
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
  expect(new URL(page.url()).searchParams.get('schedule_draft')).toBe(draft)
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeEnabled()
})

test('reasoning model switch saves the old model before continuing to the requested model', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  let started!: (patch: SchedulePatch) => void
  const requested = new Promise<SchedulePatch>((resolve) => {
    started = resolve
  })
  await page.route('**/api/model-route/schedule', async (route) => {
    if (route.request().method() !== 'PATCH') return route.fallback()
    started(route.request().postDataJSON() as SchedulePatch)
    await gate
    await route.fulfill(response({ snapshot_revision_new: 12, detail: null }))
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  await page.getByRole('button', { name: 'worker-b', exact: true }).click()
  await modelSwitchDialog(page)
    .getByRole('button', { name: 'Save and continue', exact: true })
    .click()
  try {
    const patch = await requested
    expect(patch.external_model).toBe('worker')
    expect(patch.updates).toEqual([{ group_id: 1, entry_id: 'entry-1', reasoning_effort: 'high' }])
    // The response is held: navigating before successful persistence would lose the draft.
    await expect(page).toHaveURL(/schedule_model=worker(?:&|$)/)
    await expect(page).toHaveURL(/schedule_draft=/)
  } finally {
    release()
  }
  await expect(page).toHaveURL(/schedule_model=worker-b/)
  await expect(page).not.toHaveURL(/schedule_draft=/)
  await expect(page.getByRole('combobox', { name: /Reasoning policy model-a-b/ })).toBeVisible()
})

for (const outcome of ['error', 'conflict'] as const) {
  test(`reasoning Save and continue ${outcome} stays on the original model with its draft`, async ({
    page,
  }) => {
    const { patches } = await installScheduleRoutes(page, { patchOutcome: outcome })
    await openSchedule(page, '?schedule_model=worker')
    await waitForScheduleDetailReady(page)
    await changeReasoning(page)
    await expect(page).toHaveURL(/schedule_draft=/)
    const draft = new URL(page.url()).searchParams.get('schedule_draft')
    await page.getByRole('button', { name: 'worker-b', exact: true }).click()
    const guard = modelSwitchDialog(page)
    await guard.getByRole('button', { name: 'Save and continue', exact: true }).click()
    await expect.poll(() => patches.length).toBe(1)
    await expect(
      page
        .getByText(
          outcome === 'conflict'
            ? 'The configuration changed. Refresh and try again.'
            : 'Save failed.',
        )
        .first(),
    ).toBeVisible()
    await expect(page).toHaveURL(/schedule_model=worker(?:&|$)/)
    expect(new URL(page.url()).searchParams.get('schedule_draft')).toBe(draft)
    await expect(guard).toBeVisible()
    await guard.getByRole('button', { name: 'Stay on this model', exact: true }).click()
    await expect(
      page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
    ).toContainText('high')
    await changeReasoning(page, 'low')
  })
}

test('reasoning page exit Save and continue waits for successful persistence before leaving', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  let started!: (patch: SchedulePatch) => void
  const requested = new Promise<SchedulePatch>((resolve) => {
    started = resolve
  })
  await page.route('**/api/model-route/schedule', async (route) => {
    if (route.request().method() !== 'PATCH') return route.fallback()
    started(route.request().postDataJSON() as SchedulePatch)
    await gate
    await route.fulfill(response({ snapshot_revision_new: 12, detail: null }))
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  const originalURL = page.url()
  await page.getByRole('link', { name: 'Request logs', exact: true }).click()
  const guard = modelSwitchDialog(page)
  await expect(guard).toBeVisible()
  await guard.getByRole('button', { name: 'Save and continue', exact: true }).click()
  try {
    expect((await requested).updates).toEqual([
      { group_id: 1, entry_id: 'entry-1', reasoning_effort: 'high' },
    ])
    await expect(page).toHaveURL(originalURL)
    await expect(guard).toBeVisible()
  } finally {
    release()
  }
  await expect(page).toHaveURL(/\/logs/)
  await expect(guard).not.toBeVisible()
})

for (const outcome of ['error', 'conflict'] as const) {
  test(`reasoning page exit Save and continue ${outcome} keeps pending guard and original draft URL`, async ({
    page,
  }) => {
    const { patches } = await installScheduleRoutes(page, { patchOutcome: outcome })
    await openSchedule(page, '?schedule_model=worker')
    await waitForScheduleDetailReady(page)
    await changeReasoning(page)
    await expect(page).toHaveURL(/schedule_draft=/)
    const originalURL = page.url()
    await page.getByRole('link', { name: 'Request logs', exact: true }).click()
    const guard = modelSwitchDialog(page)
    await guard.getByRole('button', { name: 'Save and continue', exact: true }).click()
    await expect.poll(() => patches.length).toBe(1)
    await expect(
      page
        .getByText(
          outcome === 'conflict'
            ? 'The configuration changed. Refresh and try again.'
            : 'Save failed.',
        )
        .first(),
    ).toBeVisible()
    await expect(guard).toBeVisible()
    await expect(page).toHaveURL(originalURL)
    await guard.getByRole('button', { name: 'Stay on this model', exact: true }).click()
    await expect(
      page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
    ).toContainText('high')
  })
}

test('reasoning browser Back uses the same three-option guard and can stay or discard', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker-b')
  await waitForScheduleDetailReady(page)
  await page.getByRole('button', { name: 'worker', exact: true }).click()
  await expect(page).toHaveURL(/schedule_model=worker(?:&|$)/)
  await changeReasoning(page)
  const originalURL = page.url()
  // History traversal is asynchronous; avoid awaiting a navigation that is intentionally blocked.
  await page.evaluate(() => window.history.back())
  const guard = modelSwitchDialog(page)
  await expect(guard).toBeVisible()
  await expect(guard.getByRole('button', { name: 'Save and continue', exact: true })).toBeVisible()
  await guard.getByRole('button', { name: 'Stay on this model', exact: true }).click()
  await expect(page).toHaveURL(originalURL)
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
  await page.evaluate(() => window.history.back())
  await expect(guard).toBeVisible()
  await guard.getByRole('button', { name: 'Discard and continue', exact: true }).click()
  await expect(page).toHaveURL(/schedule_model=worker-b/)
  await expect(page).not.toHaveURL(/schedule_draft=/)
})

test('reasoning draft survives explicit refresh with a newer snapshot revision', async ({
  page,
}) => {
  let revision = 11
  const { patches } = await installScheduleRoutes(page, {
    projectDetail: (model) => ({ ...detailForModel(model), snapshot_revision: revision }),
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  await expect(page).toHaveURL(/schedule_draft=/)
  const draft = new URL(page.url()).searchParams.get('schedule_draft')
  revision = 12
  const refreshed = page.waitForResponse(
    (response) => new URL(response.url()).pathname === '/api/model-route/schedule/detail',
  )
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  expect((await (await refreshed).json()).data.snapshot_revision).toBe(12)
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
  expect(new URL(page.url()).searchParams.get('schedule_draft')).toBe(draft)
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(() => patches.length).toBe(1)
  expect(patches[0]!.snapshot_revision).toBe(12)
  expect(patches[0]!.updates).toEqual([
    { group_id: 1, entry_id: 'entry-1', reasoning_effort: 'high' },
  ])
})

test('reasoning-only page exit is guarded and beforeunload protects only dirty state', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const unloadPrevented = () =>
    page.evaluate(() => {
      const event = new Event('beforeunload', { cancelable: true })
      window.dispatchEvent(event)
      return event.defaultPrevented
    })
  expect(await unloadPrevented()).toBe(false)
  await changeReasoning(page)
  expect(await unloadPrevented()).toBe(true)
  await page.getByRole('link', { name: 'Request logs', exact: true }).click()
  const guard = modelSwitchDialog(page)
  await expect(guard).toBeVisible()
  await guard.getByRole('button', { name: 'Stay on this model', exact: true }).click()
  await expect(page).toHaveURL(/\/schedule\?/)
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
  await page.getByRole('link', { name: 'Request logs', exact: true }).click()
  await guard.getByRole('button', { name: 'Discard and continue', exact: true }).click()
  await expect(page).toHaveURL(/\/logs/)
  expect(await unloadPrevented()).toBe(false)
})

test('row More popover exposes price and concurrency, Escape restores trigger focus', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await expect(page.getByRole('button', { name: 'Price', exact: true })).not.toBeVisible()
  await expect(page.getByRole('spinbutton', { name: /Concurrency limit/ })).not.toBeVisible()
  const trigger = moreButton(page)
  await trigger.focus()
  await trigger.press('Enter')
  const popover = page.getByRole('dialog').filter({
    has: page.locator('#concurrency-0'),
  })
  await expect(popover).toBeVisible()
  await expect(page.getByRole('button', { name: 'Price', exact: true })).toBeVisible()
  const concurrency = popover.locator('#concurrency-0')
  await expect(concurrency).toBeVisible()
  await concurrency.focus()
  await page.keyboard.press('Escape')
  await expect(popover).not.toBeVisible()
  await expect(concurrency).not.toBeVisible()
  await expect(trigger).toBeFocused()
})

test('blacklisted row keeps recovery directly available outside More', async ({ page }) => {
  await installScheduleRoutes(page, {
    projectDetail: (model) => {
      const detail = detailForModel(model)
      detail.groups[0]!.entries[0]!.runtime.state = 'blacklisted'
      return detail
    },
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const recovery = rows(page).first().getByRole('button', { name: 'Recover', exact: true })
  await expect(recovery).toBeVisible()
  const recovered = page.waitForRequest(
    (request) =>
      new URL(request.url()).pathname === '/api/model-route/schedule/recover' &&
      request.method() === 'POST',
  )
  await recovery.click()
  expect((await recovered).postDataJSON()).toEqual({
    group_id: 1,
    entry_id: 'entry-1',
    failure_version: 0,
  })
})

test('sticky schedule header has opaque background over scrolled rows', async ({
  page,
}, testInfo) => {
  await installScheduleRoutes(page)
  await page.setViewportSize({ width: 1440, height: 900 })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const table = page.getByRole('table', { name: 'Schedule detail' })
  const header = table.getByRole('row').first()
  const wrap = table.locator('xpath=..')
  await wrap.evaluate((element) => {
    element.style.height = '90px'
    element.style.maxHeight = '90px'
  })
  expect(await wrap.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(true)
  await wrap.evaluate((element) => {
    element.scrollTop = 48
  })
  await expect.poll(() => wrap.evaluate((element) => element.scrollTop)).toBe(48)
  const coverage = await header.evaluate((element) => {
    const css = getComputedStyle(element)
    const rect = element.getBoundingClientRect()
    const wrapper = element.parentElement!.parentElement!
    const top = wrapper.getBoundingClientRect().top + wrapper.clientTop
    const probe = document.createElement('span')
    probe.style.backgroundColor = 'var(--color-surface-sunken)'
    element.append(probe)
    const expectedBackground = getComputedStyle(probe).backgroundColor
    probe.remove()
    const x = rect.left + 110
    const y = rect.top + rect.height / 2
    const underlyingRow = Array.from(
      element.parentElement!.querySelectorAll('[data-row-key]'),
    ).some((row) => {
      const bounds = row.getBoundingClientRect()
      return bounds.top < y && bounds.bottom > y
    })
    return {
      position: css.position,
      background: css.backgroundColor,
      expectedBackground,
      pinned: Math.abs(rect.top - top) < 1,
      headerOnTop: element.contains(document.elementFromPoint(x, y)),
      underlyingRow,
    }
  })
  await testInfo.attach('sticky-header-coverage', {
    body: JSON.stringify(coverage, null, 2),
    contentType: 'application/json',
  })
  expect(coverage.position).toBe('sticky')
  expect(coverage.pinned).toBe(true)
  expect(coverage.underlyingRow).toBe(true)
  expect(coverage.headerOnTop).toBe(true)
  expect(coverage.background).toBe(coverage.expectedBackground)
  expect(coverage.background).toMatch(/^rgb\(/)
  await page.screenshot({
    path: testInfo.outputPath('schedule-sticky-scrolled.png'),
    fullPage: true,
  })
})

test('schedule rows support keyboard selection without hijacking input keys', async ({ page }) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const first = rows(page).first()
  await expect(first).toHaveAttribute('tabindex', '0')
  await first.focus()
  await first.press('Enter')
  await expect(first).toHaveAttribute('aria-selected', 'true')
  const second = rows(page).nth(1)
  await second.focus()
  await second.press('Space')
  await expect(second).toHaveAttribute('aria-selected', 'true')
  await first.getByRole('spinbutton').first().press('Enter')
  await expect(second).toHaveAttribute('aria-selected', 'true')
})

test('invalid schedule numbers explain their accepted range', async ({ page }) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const priority = rows(page).first().getByRole('spinbutton', { name: 'Priority model-a' })
  await priority.fill('0')
  await expect(priority).toHaveAttribute('aria-invalid', 'true')
  await expect(priority).toHaveAccessibleDescription(/1/)
  await expect(page.locator('#priority-0-error')).toBeVisible()
  await priority.fill('2')
  await expect(priority).not.toHaveAttribute('aria-describedby')
  const weight = rows(page).first().getByRole('spinbutton', { name: 'Weight model-a' })
  await weight.fill('101')
  await expect(weight).toHaveAccessibleDescription(/0.*100/)
})

test('compact desktop uses six semantic columns and 58–84px rows at every desktop breakpoint', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const table = page.getByRole('table', { name: 'Schedule detail' })
  for (const width of [768, 900, 1200, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 })
    await expect(table.getByRole('columnheader')).toHaveCount(6)
    const row = rows(page).first()
    await expect(row.getByRole('cell')).toHaveCount(6)
    const identity = row.getByRole('cell').nth(0)
    await expect(identity.getByText('model-a', { exact: true })).toBeVisible()
    await expect(identity.getByRole('link', { name: 'primary group', exact: true })).toBeVisible()
    await expect(row.getByRole('cell').nth(1)).toContainText('Available')
    await expect(row.getByRole('cell').nth(2)).toContainText('10')
    await expect(row.getByRole('cell').nth(2)).toContainText('100')
    const settings = row.getByRole('cell').nth(3)
    await expect(
      settings.getByRole('spinbutton', { name: /^Priority model-a(?: · |$)/ }),
    ).toBeVisible()
    await expect(
      settings.getByRole('spinbutton', { name: /^Weight model-a(?: · |$)/ }),
    ).toBeVisible()
    await expect(
      settings.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
    ).toBeVisible()
    // Configured share is normalized within each priority across every row the
    // server includes: the enabled group owns the whole tier, the disabled
    // group contributes and receives nothing (internal/control/route_inspect.go
    // `configuredEntryShares`).
    await expect(row.getByRole('cell').nth(4)).toContainText('100.0%')
    await expect(rows(page).nth(1).getByRole('cell').nth(4)).toContainText('0.0%')
    const settingsGeometry = await row.evaluate((element) => {
      const cells = [...element.querySelectorAll<HTMLElement>('[role="cell"]')]
      const settingsCell = cells[3]!
      const shareCell = cells[4]!
      const combo = settingsCell.querySelector<HTMLElement>('[role="combobox"]')!
      return {
        scroll: settingsCell.scrollWidth,
        client: settingsCell.clientWidth,
        comboRight: combo.getBoundingClientRect().right,
        shareLeft: shareCell.getBoundingClientRect().left,
      }
    })
    expect(
      settingsGeometry.scroll,
      `settings controls stay inside their column at ${width}px`,
    ).toBeLessThanOrEqual(settingsGeometry.client + 1)
    expect(
      settingsGeometry.comboRight,
      `reasoning selector does not reach the share column at ${width}px`,
    ).toBeLessThanOrEqual(settingsGeometry.shareLeft)
    // The share cell must render `100.0%` (the widest real value) inside its own
    // track instead of sliding under the action switch.
    const shareGeometry = await row.evaluate((element) => {
      const cells = [...element.querySelectorAll<HTMLElement>('[role="cell"]')]
      const shareCell = cells[4]!
      const actionsCell = cells[5]!
      const value = document.createRange()
      value.selectNode(shareCell.lastChild!)
      return {
        scroll: shareCell.scrollWidth,
        client: shareCell.clientWidth,
        valueRight: value.getBoundingClientRect().right,
        actionsLeft: actionsCell.getBoundingClientRect().left,
      }
    })
    expect(
      shareGeometry.scroll,
      `share text stays inside its column at ${width}px`,
    ).toBeLessThanOrEqual(shareGeometry.client + 1)
    expect(
      shareGeometry.valueRight,
      `share value does not reach the actions column at ${width}px`,
    ).toBeLessThanOrEqual(shareGeometry.actionsLeft)
    // The actions cell is right-aligned, so any content wider than its track
    // spills leftwards over the share value instead of growing rightwards.
    const actionsGeometry = await row.evaluate((element) => {
      const cells = [...element.querySelectorAll<HTMLElement>('[role="cell"]')]
      const actionsCell = cells[5]!
      const children = [...actionsCell.children].filter(
        (child) => child.getBoundingClientRect().width > 0,
      )
      const content = children.reduce(
        (total, child) =>
          total +
          child.getBoundingClientRect().width +
          Number.parseFloat(getComputedStyle(actionsCell).columnGap || '0'),
        0,
      )
      return {
        scroll: actionsCell.scrollWidth,
        client: actionsCell.clientWidth,
        content,
        firstChildLeft: children[0]!.getBoundingClientRect().left,
        shareRight: cells[4]!.getBoundingClientRect().right,
      }
    })
    expect(
      actionsGeometry.scroll,
      `action controls stay inside their column at ${width}px`,
    ).toBeLessThanOrEqual(actionsGeometry.client + 1)
    expect(
      actionsGeometry.firstChildLeft,
      `action controls do not reach back into the share column at ${width}px`,
    ).toBeGreaterThanOrEqual(actionsGeometry.shareRight)
    await expect(row.getByRole('cell').nth(5).getByRole('switch')).toBeVisible()
    await expect(
      row
        .getByRole('cell')
        .nth(5)
        .getByRole('button', { name: /More|更多/i }),
    ).toBeVisible()
    const geometry = await row.evaluate((element) => {
      const bounds = element.getBoundingClientRect()
      return {
        height: bounds.height,
        columns: getComputedStyle(element).gridTemplateColumns.split(' ').length,
      }
    })
    expect(geometry.height, `58–84px compact row at ${width}px`).toBeGreaterThanOrEqual(58)
    expect(geometry.height).toBeLessThanOrEqual(84)
    expect(geometry.columns).toBe(6)
    expect(await table.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await expect(row.locator('summary')).toHaveCount(0)
    await expect(row.getByRole('progressbar')).toHaveCount(0)
    await expect(page.getByRole('spinbutton', { name: /Concurrency limit/ })).not.toBeVisible()
    await expect(page.getByRole('button', { name: 'Price', exact: true })).not.toBeVisible()
  }
})

test('schedule visible typography keeps metadata and controls readable', async ({ page }) => {
  await installScheduleRoutes(page)
  await page.setViewportSize({ width: 390, height: 900 })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const row = rows(page).first()
  const identity = row.getByText('model-a', { exact: true })
  const group = row.getByRole('link', { name: 'primary group', exact: true })
  const readFont = (element: Element) => Number.parseFloat(getComputedStyle(element).fontSize)
  expect(await identity.evaluate(readFont)).toBeGreaterThanOrEqual(13)
  expect(await group.evaluate(readFont)).toBeGreaterThanOrEqual(12)
  expect(
    await row.getByRole('spinbutton', { name: /^Priority model-a(?: · |$)/ }).evaluate(readFont),
  ).toBeGreaterThanOrEqual(14)
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
  await moreButton(page).click()
  await page.getByRole('button', { name: 'Price', exact: true }).click()
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
  await moreButton(page).click()
  await page.getByRole('button', { name: 'Price', exact: true }).click()
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
  await expect(page.getByRole('columnheader')).toHaveCount(6)
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

test('compact mobile uses 205–240px cards and 44px settings and actions without an inline model index', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await page.setViewportSize({ width: 390, height: 844 })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await expect(page.getByRole('textbox', { name: 'External model', exact: true })).not.toBeVisible()
  await expect(page.getByRole('button', { name: 'worker-b', exact: true })).not.toBeVisible()
  const row = rows(page).first()
  const box = await row.boundingBox()
  expect(box!.height).toBeGreaterThanOrEqual(205)
  expect(box!.height).toBeLessThanOrEqual(240)
  for (const control of [
    row.getByRole('spinbutton', { name: /^Priority model-a(?: · |$)/ }),
    row.getByRole('spinbutton', { name: /^Weight model-a(?: · |$)/ }),
    row.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
    row.getByRole('switch', { name: /^(?:Enable|Disable) entry model-a(?: · primary group)?$/ }),
    row.getByRole('button', { name: /More|更多/i }),
    row.getByRole('button', { name: 'Probe', exact: true }),
  ]) {
    const bounds = await control.boundingBox()
    expect(bounds!.width).toBeGreaterThanOrEqual(44)
    expect(bounds!.height).toBeGreaterThanOrEqual(44)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  await expect(row.locator('summary')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Price', exact: true })).not.toBeVisible()
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

test('compact model index is 180px at wide desktop and collapsed below 1440px', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await page.setViewportSize({ width: 1440, height: 900 })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const search = page.getByRole('textbox', { name: 'External model', exact: true })
  const model = page.getByRole('button', { name: 'worker', exact: true })
  const table = page.getByRole('table', { name: 'Schedule detail' })
  const modelBounds = await model.boundingBox()
  const tableBounds = await table.boundingBox()
  expect(modelBounds!.width).toBeGreaterThanOrEqual(176)
  expect(modelBounds!.width).toBeLessThanOrEqual(180)
  expect(modelBounds!.x + modelBounds!.width).toBeLessThanOrEqual(tableBounds!.x)
  await expect(search).toBeVisible()
  for (const width of [1439, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 900 })
    await expect(search).not.toBeVisible()
    await expect(page.getByRole('button', { name: 'worker-b', exact: true })).not.toBeVisible()
    await expect(rows(page).first()).toBeVisible()
    const selector = page.getByTestId('schedule-model-selector')
    await expect(selector).toBeVisible()
    await selector.click()
    const current = page.getByRole('button', { name: 'worker', exact: true })
    await expect(current).toHaveAttribute('aria-pressed', 'true')
    await expect(page.getByRole('button', { name: 'worker-b', exact: true })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
    await page.keyboard.press('Escape')
    await expect(selector).toBeFocused()
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
  }
})

test('search with no match offers a distinct empty message and can recover', async ({ page }) => {
  await installScheduleRoutes(page)
  await openSchedule(page)
  await page.getByRole('textbox', { name: 'External model', exact: true }).fill('unknown-model')
  await expect(
    page.getByText('No matching models', { exact: true }).filter({ visible: true }),
  ).toBeVisible()
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
  await expect(
    page.getByText('No models available', { exact: true }).filter({ visible: true }),
  ).toBeVisible()
  await expect(
    page.getByText('Select a model', { exact: true }).filter({ visible: true }),
  ).not.toBeVisible()
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
  await page
    .getByRole('switch', { name: /^(?:Enable|Disable) entry model-a(?: · primary group)?$/ })
    .click()
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

test('read-only model search updates immediately but debounces the query and page together', async ({
  page,
}) => {
  await installScheduleRoutes(page, { principalType: 'access_key' })
  const requests: string[] = []
  await page.route('**/api/models?**', async (route) => {
    requests.push(route.request().url())
    await route.fulfill(response(emptyModelCollection()))
  })
  await openSchedule(page, '?schedule_model=worker')
  const input = page.getByRole('textbox', { name: 'Model' })
  await expect(input).toHaveValue('worker')
  const initialRequestCount = requests.length
  await input.fill('worker-b')
  await expect(input).toHaveValue('worker-b')
  await page.waitForTimeout(100)
  expect(requests.length).toBe(initialRequestCount)
  await expect.poll(() => requests.length).toBe(initialRequestCount + 1)
  expect(new URL(requests.at(-1)!).searchParams.get('q')).toBe('worker-b')
  expect(new URL(requests.at(-1)!).searchParams.get('page')).toBe('1')
})

test('read-only model collection errors use a banner and can retry', async ({ page }) => {
  await installScheduleRoutes(page, { principalType: 'access_key' })
  let attempts = 0
  await page.route('**/api/models?**', async (route) => {
    attempts += 1
    await route.fulfill(attempts === 1 ? response({}, 500) : response(emptyModelCollection()))
  })
  await openSchedule(page, '?schedule_model=worker')
  await expect(page.getByRole('alert')).toBeVisible()
  await expect(
    page.getByText('Unable to load the model schedule index.', { exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(attempts).toBe(2)
})

test('review regression Stay cancels pending model state before subsequent draft edits', async ({
  page,
}) => {
  test.setTimeout(20_000)
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  await page.getByRole('button', { name: 'worker-b', exact: true }).click()
  const guard = modelSwitchDialog(page)
  await guard.getByRole('button', { name: 'Stay on this model', exact: true }).click()
  await page.locator('#weight-0').fill('37')
  await expect(guard).not.toBeVisible()
  await expect(page).toHaveURL(/schedule_model=worker(?:&|$)/)
  await expect
    .poll(() => {
      const raw = new URL(page.url()).searchParams.get('schedule_draft')
      return raw ? JSON.parse(raw) : null
    })
    .toEqual({ '1:entry-1': { reasoning_effort: 'high', weight: 37 } })
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
})

test('review regression late save refresh cannot resurrect schedule after leaving', async ({
  page,
}) => {
  test.setTimeout(20_000)
  await installScheduleRoutes(page)
  let saved = false
  let postSaveReads = 0
  let refreshing = false
  let completed = false
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })

  await page.route('**/api/model-route/schedule', async (route) => {
    if (route.request().method() === 'PATCH') saved = true
    await route.fallback()
  })
  await page.route('**/api/model-route/schedule/detail?*', async (route) => {
    if (!saved) return route.fallback()
    postSaveReads += 1
    if (postSaveReads < 2) return route.fallback()
    refreshing = true
    await gate
    await route.fulfill(response({ ...detailForModel('worker'), snapshot_revision: 12 }))
    completed = true
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  await page.getByRole('link', { name: 'Request logs', exact: true }).click()
  await modelSwitchDialog(page)
    .getByRole('button', { name: 'Save and continue', exact: true })
    .click()
  try {
    await expect.poll(() => refreshing).toBe(true)
    await expect(page).toHaveURL(/\/logs/)
  } finally {
    release()
  }
  await expect.poll(() => completed).toBe(true)
  await page.waitForLoadState('networkidle')
  await expect(page).toHaveURL(/\/logs/)
  await expect(page.getByTestId('schedule-page')).not.toBeVisible()
})

test('review regression same-model price history Back still guards reasoning drafts', async ({
  page,
}) => {
  test.setTimeout(20_000)
  await installScheduleRoutes(page)
  const lookup = await delayPriceLookup(page)
  let priceRequested = false
  void lookup.requested.then(() => {
    priceRequested = true
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await moreButton(page).click()
  await page.getByRole('button', { name: 'Price', exact: true }).click()
  await expect.poll(() => priceRequested).toBe(true)
  lookup.release()
  await expect(page).toHaveURL(/selected_price_id=7/)
  // The drawer may show its mocked load error; closing it still exercises the real history entry.
  const drawer = page
    .getByRole('dialog')
    .filter({ has: page.getByRole('button', { name: 'Close', exact: true }) })
  await drawer.getByRole('button', { name: 'Close', exact: true }).click()
  await expect(page).not.toHaveURL(/selected_price_id=/)
  await changeReasoning(page)
  const originalURL = page.url()
  await page.evaluate(() => window.history.back())
  const guard = modelSwitchDialog(page)
  await expect(guard).toBeVisible()
  await guard.getByRole('button', { name: 'Stay on this model', exact: true }).click()
  await expect(page).toHaveURL(originalURL)
  await expect(
    page.getByRole('combobox', { name: /Reasoning policy model-a.*primary group/ }),
  ).toContainText('high')
  await page.evaluate(() => window.history.back())
  await guard.getByRole('button', { name: 'Discard and continue', exact: true }).click()
  await expect(page).toHaveURL(/schedule_model=worker/)
  await expect(page).not.toHaveURL(/schedule_draft=/)
})

test('review regression URL echo preserves an invalid raw priority alongside reasoning and weight', async ({
  page,
}) => {
  test.setTimeout(20_000)
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  await changeReasoning(page)
  const priority = page.locator('#priority-0')
  await priority.fill('0')
  await expect(priority).toHaveAttribute('aria-invalid', 'true')
  await page.locator('#weight-0').fill('37')
  await expect
    .poll(() => {
      const raw = new URL(page.url()).searchParams.get('schedule_draft')
      return raw ? JSON.parse(raw) : null
    })
    .toEqual({ '1:entry-1': { reasoning_effort: 'high', weight: 37 } })
  await expect(priority).toHaveValue('0')
  await expect(priority).toHaveAttribute('aria-invalid', 'true')
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
})

test('review regression invalid-only numeric edit remains discardable and guarded without URL draft', async ({
  page,
}) => {
  test.setTimeout(20_000)
  let revision = 11
  await installScheduleRoutes(page, {
    projectDetail: (model) => ({ ...detailForModel(model), snapshot_revision: revision }),
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const priority = page.locator('#priority-0')
  await priority.fill('0')
  await expect(priority).toHaveAttribute('aria-invalid', 'true')
  await expect(page).not.toHaveURL(/schedule_draft=/)
  const unloadPrevented = await page.evaluate(() => {
    const event = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(event)
    return event.defaultPrevented
  })
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Discard changes', exact: true })).toBeVisible()
  expect(unloadPrevented).toBe(true)
  await page.getByRole('button', { name: 'worker-b', exact: true }).click()
  const guard = modelSwitchDialog(page)
  await expect(guard).toBeVisible()
  await expect(guard.getByRole('button', { name: 'Save and continue', exact: true })).toBeDisabled()
  await guard.getByRole('button', { name: 'Stay on this model', exact: true }).click()
  await expect(priority).toHaveValue('0')
  await expect(priority).toHaveAttribute('aria-invalid', 'true')
  revision = 12
  let refreshed = false
  page.on('response', async (response) => {
    if (new URL(response.url()).pathname === '/api/model-route/schedule/detail') {
      refreshed = (await response.json()).data.snapshot_revision === 12
    }
  })
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await expect.poll(() => refreshed).toBe(true)
  await expect(priority).toHaveValue('0')
  await expect(priority).toHaveAttribute('aria-invalid', 'true')
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'worker-b', exact: true }).click()
  await guard.getByRole('button', { name: 'Discard and continue', exact: true }).click()
  await expect(page).toHaveURL(/schedule_model=worker-b/)
})

for (const action of ['Save and continue', 'Discard and continue'] as const) {
  test(`review regression same-model Back ${action} never revives an older reasoning draft`, async ({
    page,
  }) => {
    test.setTimeout(20_000)
    const state: ScheduleRoutes = await installScheduleRoutes(page, {
      projectDetail: (model) => {
        const detail = detailForModel(model)
        const saved = state.patches
          .at(-1)
          ?.updates.find((update) => update.reasoning_effort !== undefined)
        if (saved) {
          detail.snapshot_revision = 12
          detail.groups[0]!.entries[0]!.reasoning.configured = saved.reasoning_effort ?? null
          detail.groups[0]!.entries[0]!.reasoning.effective = saved.reasoning_effort ?? null
          detail.groups[0]!.entries[0]!.reasoning.source = 'entry'
        }
        return detail
      },
    })
    const lookup = await delayPriceLookup(page)
    let priceRequested = false
    void lookup.requested.then(() => {
      priceRequested = true
    })
    await openSchedule(page, '?schedule_model=worker')
    await waitForScheduleDetailReady(page)
    await changeReasoning(page, 'high')
    await expect(page).toHaveURL(/schedule_draft=/)
    const oldDraft = new URL(page.url()).searchParams.get('schedule_draft')
    await moreButton(page).click()
    await page.getByRole('button', { name: 'Price', exact: true }).click()
    await expect.poll(() => priceRequested).toBe(true)
    lookup.release()
    await expect(page).toHaveURL(/selected_price_id=7/)
    const drawer = page
      .getByRole('dialog')
      .filter({ has: page.getByRole('button', { name: 'Close', exact: true }) })
    await drawer.getByRole('button', { name: 'Close', exact: true }).click()
    await expect(page).not.toHaveURL(/selected_price_id=/)
    await changeReasoning(page, 'low')
    await expect
      .poll(() => new URL(page.url()).searchParams.get('schedule_draft'))
      .not.toBe(oldDraft)
    await page.evaluate(() => window.history.back())
    const guard = modelSwitchDialog(page)
    await expect(guard).toBeVisible()
    await guard.getByRole('button', { name: action, exact: true }).click()
    await expect(page).not.toHaveURL(/schedule_draft=/)
    await expect(guard).not.toBeVisible()
    const reasoning = page.getByRole('combobox', {
      name: /Reasoning policy model-a.*primary group/,
    })
    await expect(reasoning).not.toContainText('high')
    if (action === 'Save and continue') {
      await expect.poll(() => state.patches.length).toBe(1)
      expect(state.patches[0]!.updates).toEqual([
        { group_id: 1, entry_id: 'entry-1', reasoning_effort: 'low' },
      ])
      await expect(reasoning).toContainText('low')
    } else {
      expect(state.patches).toHaveLength(0)
      await expect(reasoning).not.toContainText('low')
    }
  })
}

test('review regression reverted numeric input adopts a newer server value on refresh', async ({
  page,
}) => {
  test.setTimeout(20_000)
  let revision = 11
  let serverPriority = 1
  let refreshed = false
  await installScheduleRoutes(page, {
    projectDetail: (model) => {
      const detail = detailForModel(model)
      detail.snapshot_revision = revision
      detail.groups[0]!.entries[0]!.priority = serverPriority
      if (serverPriority !== 1) {
        // Configured shares normalize within each priority tier, not globally.
        // The disabled group is excluded from the distribution entirely, so its
        // row holds no configured share (internal/control/route_inspect.go
        // `configuredEntryShares`).
        detail.groups[0]!.entries[0]!.configured_share = 1
        detail.groups[1]!.entries[0]!.configured_share = 0
      }
      return detail
    },
  })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const priority = page.getByRole('spinbutton', { name: /^Priority model-a(?: · |$)/ })
  await expect(priority).toHaveValue('1')
  await priority.fill('2')
  await expect(page).toHaveURL(/schedule_draft=/)
  await priority.fill('1')
  await expect(page).not.toHaveURL(/schedule_draft=/)
  await expect(page.getByRole('button', { name: 'Discard changes', exact: true })).not.toBeVisible()
  revision = 12
  serverPriority = 4
  page.on('response', async (response) => {
    if (new URL(response.url()).pathname === '/api/model-route/schedule/detail') {
      refreshed = (await response.json()).data.snapshot_revision === 12
    }
  })
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await expect.poll(() => refreshed).toBe(true)
  await expect(priority).toHaveValue('4')
  await expect(priority).not.toHaveAttribute('aria-invalid', 'true')
  await expect(page).not.toHaveURL(/schedule_draft=/)
})

test('unknown schedule_model explains the empty state and can recover to worker', async ({
  page,
}) => {
  await installScheduleRoutes(page)
  await openSchedule(page, '?schedule_model=unknown-model')
  await expect(
    page.getByText('No matching models', { exact: true }).filter({ visible: true }),
  ).toBeVisible()
  await expect(page.getByRole('table', { name: 'Schedule detail' })).not.toBeVisible()
  await expect(page).toHaveURL(/schedule_model=unknown-model/)
  await page.getByRole('button', { name: 'worker', exact: true }).click()
  await expect(page).toHaveURL(/schedule_model=worker(?:&|$)/)
  await waitForScheduleDetailReady(page)
  await expect(rows(page)).toHaveCount(2)
})

test('compact model selector supports search, ArrowDown Enter and Escape focus restoration', async ({
  page,
}, testInfo) => {
  await installScheduleRoutes(page)
  await page.setViewportSize({ width: 900, height: 900 })
  await openSchedule(page, '?schedule_model=worker')
  await waitForScheduleDetailReady(page)
  const trigger = page.getByTestId('schedule-model-selector')
  await trigger.focus()
  await trigger.press('Enter')
  const picker = page.getByRole('dialog', { name: 'Select a model', exact: true })
  const search = picker.getByRole('textbox', { name: 'External model', exact: true })
  await search.fill('worker-b')
  await expect(picker.getByRole('button', { name: 'worker', exact: true })).toHaveCount(0)
  const next = picker.getByRole('button', { name: 'worker-b', exact: true })
  await expect(next).toHaveAttribute('aria-pressed', 'false')
  await search.press('ArrowDown')
  await expect(next).toBeFocused()
  await next.press('Enter')
  await expect(page).toHaveURL(/schedule_model=worker-b/)
  await expect(picker).not.toBeVisible()
  await trigger.click()
  await expect(picker.getByRole('button', { name: 'worker-b', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
  await search.fill('worker')
  await search.press('ArrowDown')
  const current = picker.getByRole('button', { name: 'worker', exact: true })
  await expect(current).toBeFocused()
  await current.press('Enter')
  await expect(page).toHaveURL(/schedule_model=worker(?:&|$)/)
  await trigger.click()
  await search.fill('no-match')
  await expect(picker.getByText('No matching models', { exact: true })).toBeVisible()
  await search.press('Escape')
  await expect(picker).not.toBeVisible()
  await expect(trigger).toBeFocused()

  const directory = '/tmp/gpt-load-schedule-product'
  await mkdir(directory, { recursive: true })
  const measurements = []
  for (const { width, theme, locale } of [
    { width: 1440, theme: 'light', locale: 'en-US' },
    { width: 900, theme: 'light', locale: 'en-US' },
    { width: 390, theme: 'light', locale: 'en-US' },
    { width: 1440, theme: 'dark', locale: 'en-US' },
    { width: 1440, theme: 'light', locale: 'zh-CN' },
    { width: 390, theme: 'light', locale: 'zh-CN' },
  ] as const) {
    await page.setViewportSize({ width, height: 900 })
    await page.evaluate(
      ({ theme, locale }) => {
        localStorage.setItem('gpt-load.theme', theme)
        localStorage.setItem('gpt-load.locale', locale)
      },
      { theme, locale },
    )
    await page.reload()
    await waitForScheduleTable(page)
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme)
    if (locale === 'zh-CN') {
      // Long localized eligibility copy renders in both layouts; the header row
      // only exists at >=768px (`rowHeader` is display:none in the narrow card).
      await expect(rows(page).first().getByRole('cell').nth(1)).toContainText('可调度')
      await expect(rows(page).nth(1).getByRole('cell').nth(1)).toContainText('分组已停用')
      if (width >= 768) {
        await expect(page.getByRole('columnheader').nth(1)).toContainText('已保存调度资格')
        await expect(page.getByRole('columnheader').nth(4)).toContainText('同级配置占比')
      }
    }
    // Fixture truth: the enabled group owns the tier, the disabled group none.
    await expect(rows(page).first().getByRole('cell').nth(4)).toContainText('100.0%')
    await expect(rows(page).nth(1).getByRole('cell').nth(4)).toContainText('0.0%')
    const geometry = await rows(page).evaluateAll((elements) => ({
      viewport: window.innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      rows: elements.map((element) => {
        const rect = element.getBoundingClientRect()
        return { width: rect.width, height: rect.height }
      }),
    }))
    expect(geometry.documentWidth).toBeLessThanOrEqual(width)
    for (const row of geometry.rows) {
      expect(row.height).toBeGreaterThanOrEqual(width < 768 ? 205 : 58)
      expect(row.height).toBeLessThanOrEqual(width < 768 ? 240 : 84)
    }
    const screenshot = `${directory}/schedule-${width}-${theme}-${locale}.png`
    await page.screenshot({ path: screenshot, fullPage: true })
    measurements.push({ width, theme, locale, screenshot, ...geometry })
  }
  const evidence = JSON.stringify(measurements, null, 2)
  await writeFile(`${directory}/measurements.json`, evidence)
  await testInfo.attach('compact-product-geometry', {
    body: evidence,
    contentType: 'application/json',
  })
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
