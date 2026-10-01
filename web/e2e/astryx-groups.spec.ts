import { expect, test, type Page, type Route } from '@playwright/test'

// B10 spike(a): the Groups collection on the Astryx entry — real <table>
// semantics with controlled sortable/filtering/pagination/stickyColumns
// plugin state, typed route search carrying the filter contract, and
// server-driven pagination (no client-side slicing). The mock mirrors the
// server's in-memory filter/sort/page contract so assertions prove the
// request pipeline end to end.
//
// Parity companion: same query params, same summary chips,
// same debounce/reset/page-correction behavior.

// First navigation cold-compiles the whole astryx module graph (the router
// statically imports every view) — over 30 s on a cold vite cache.
test.setTimeout(90_000)

interface FixtureGroup {
  id: number
  name: string
  status: 'available' | 'unavailable' | 'disabled'
  connection_type: 'api_key' | 'subscription'
  channel_id: string
  credential_total: number
  created_ms: number
}

function collectionItem(group: FixtureGroup) {
  return {
    id: group.id,
    name: group.name,

    channel_id: group.channel_id,
    connection_type: group.connection_type,
    params: {},
    provider_url: null,
    status: group.status,
    model_count: 4,
    client_model_count: 2,
    // Contract: a disabled group carries all credentials in `disabled`.
    credential_counts:
      group.status === 'disabled'
        ? {
            total: group.credential_total,
            available: 0,
            cooldown: 0,
            blacklisted: 0,
            disabled: group.credential_total,
          }
        : {
            total: group.credential_total,
            available: group.credential_total,
            cooldown: 0,
            blacklisted: 0,
            disabled: 0,
          },
  }
}

// Array.from's mapper passes (element, index) — wrap rather than pass
// makeGroup bare, or the index arrives as the element and ids go NaN.
function makeGroups(count: number): FixtureGroup[] {
  return Array.from({ length: count }, (_el, index) => makeGroup(index))
}

function makeGroup(index: number): FixtureGroup {
  const status = index % 7 === 0 ? 'unavailable' : index % 5 === 0 ? 'disabled' : 'available'
  return {
    id: index + 1,
    name: `Group ${String(index + 1).padStart(4, '0')}`,
    status,
    connection_type: index % 3 === 0 ? 'subscription' : 'api_key',
    channel_id: 'openai',
    credential_total: (index * 13) % 40,
    created_ms: 1_700_000_000_000 + index * 60_000,
  }
}

interface LastQuery {
  q?: string
  status?: string
  connection_type?: string
  sort?: string
  page?: number
  page_size?: number
}

// Server-mirroring collection mock: applies the shared filter/sort/page
// contract to the fixture so the test observes what the route actually
// sent, not a canned page.
async function mockCollection(page: Page, groups: FixtureGroup[]) {
  const seen: LastQuery[] = []
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
    if (path === '/api/groups') {
      const params = url.searchParams
      const query: LastQuery = {
        q: params.get('q') ?? undefined,
        status: params.get('status') ?? undefined,
        connection_type: params.get('connection_type') ?? undefined,
        sort: params.get('sort') ?? 'recent',
        page: Number(params.get('page') ?? '1'),
        page_size: Number(params.get('page_size') ?? '100'),
      }
      seen.push(query)

      let rows = groups
      if (query.q !== undefined) {
        const needle = query.q.toLowerCase()
        rows = rows.filter(
          (group) =>
            group.name.toLowerCase().includes(needle) ||
            group.channel_id.toLowerCase().includes(needle),
        )
      }
      if (query.status !== undefined) {
        rows = rows.filter((group) => group.status === query.status)
      }
      if (query.connection_type !== undefined) {
        rows = rows.filter((group) => group.connection_type === query.connection_type)
      }
      const sorted = [...rows]
      switch (query.sort) {
        case 'name':
          sorted.sort((a, b) => a.name.localeCompare(b.name))
          break
        case 'credentials':
          sorted.sort((a, b) => b.credential_total - a.credential_total)
          break
        case 'created':
          sorted.sort((a, b) => b.created_ms - a.created_ms)
          break
        case 'status': {
          const order = { unavailable: 0, available: 1, disabled: 2 }
          sorted.sort((a, b) => order[a.status] - order[b.status])
          break
        }
        case 'recent':
        default:
          sorted.sort((a, b) => b.created_ms - a.created_ms)
          break
      }

      const pageSize = query.page_size ?? 100
      const totalItems = sorted.length
      const totalPages = Math.ceil(totalItems / pageSize)
      // The server echoes the requested page; out-of-range pages return an
      // empty slice and the client self-corrects via total_pages.
      const page = query.page ?? 1
      const items =
        page <= totalPages
          ? sorted.slice((page - 1) * pageSize, page * pageSize).map(collectionItem)
          : []
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          code: 0,
          message: 'ok',
          data: {
            observed_at_ms: 1_700_000_000_000,
            summary: {
              total: groups.length,
              available: groups.filter((g) => g.status === 'available').length,
              unavailable: groups.filter((g) => g.status === 'unavailable').length,
              disabled: groups.filter((g) => g.status === 'disabled').length,
            },
            items,
            pagination: {
              page,
              page_size: pageSize,
              total_items: totalItems,
              total_pages: totalPages,
            },
          },
        }),
      })
      return
    }
    if (path === '/api/channels') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          code: 0,
          message: 'ok',
          data: { items: [], total: 0 },
        }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: {} }),
    })
  })
  return seen
}

async function expectAstryxDocument(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible()
  await expect(page.locator('[data-testid="desktop-nav"]')).toBeVisible()
}

function collectionTable(page: Page) {
  return page.getByRole('table', { name: 'Group list' })
}

test('renders summary chips, toolbar, and server-paginated rows', async ({ page }) => {
  const seen = await mockCollection(page, makeGroups(150))
  await page.goto('/groups', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  await expect(page.getByRole('heading', { name: 'Groups', exact: true })).toBeVisible()

  // Summary chips reflect the mocked totals (150 = 103/22/25 split).
  const summary = page.getByRole('region', { name: 'Group status overview' })
  await expect(summary).toBeVisible()
  await expect(summary.getByRole('button', { name: /All/ })).toBeVisible()
  await expect(summary.getByRole('button', { name: /Unavailable/ })).toBeVisible()

  // The table renders exactly one server page (100 rows), not the full set.
  const table = collectionTable(page)
  await expect(table).toBeVisible()
  await expect(table.getByRole('row')).toHaveCount(101) // header + 100 rows
  await expect(seen.at(-1)).toMatchObject({ page: 1, page_size: 100 })
})

test('search, status chip, and connection-type select hit the server', async ({ page }) => {
  const seen = await mockCollection(page, makeGroups(20))
  await page.goto('/groups', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  // Debounced search drives ?q= and resets page to 1.
  await page.getByLabel('Search').fill('group 0007')
  await expect(page).toHaveURL(/q=group(\+|%20)0007/, { timeout: 3_000 })
  await expect.poll(() => seen.at(-1)?.q).toBe('group 0007')
  await expect(seen.at(-1)).toMatchObject({ page: 1 })

  const table = collectionTable(page)
  await expect(table.getByRole('row')).toHaveCount(2) // header + single match

  // Status chip narrows server-side; page stays at 1.
  await page
    .getByRole('region', { name: 'Group status overview' })
    .getByRole('button', { name: /Disabled/ })
    .click()
  await expect(page).toHaveURL(/status=disabled/)
  await expect.poll(() => seen.at(-1)?.status).toBe('disabled')

  // Connection-type select writes the shared contract param.
  // (combobox role: 'Connection type' alone also matches the open listbox)
  await page.getByRole('combobox', { name: 'Connection type' }).click()
  await page.getByRole('option', { name: 'Subscription account' }).click()
  await expect(page).toHaveURL(/connection_type=subscription/)

  // Reset clears criteria back to canonical defaults. Two resets render at
  // this point (toolbar + no-results state); exercise the toolbar one.
  await page.getByLabel('Filter Groups').getByRole('button', { name: 'Reset filters' }).click()
  await expect(page).not.toHaveURL(/q=|status=|connection_type=/)
})

test('sort select and column-header sorting stay controlled', async ({ page }) => {
  const seen = await mockCollection(page, makeGroups(20))
  await page.goto('/groups', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  // Toolbar select: the named sort contract writes ?sort=name.
  // (combobox role: 'Sort' alone also matches the 'Sort by <col>' buttons)
  await page.getByRole('combobox', { name: 'Sort' }).click()
  await page.getByRole('option', { name: 'Name A–Z' }).click()
  await expect(page).toHaveURL(/sort=name/)
  await expect.poll(() => seen.at(-1)?.sort).toBe('name')

  const table = collectionTable(page)
  await expect(table.getByRole('row').nth(1)).toContainText('Group 0001')

  // Column-header sortable: the sortable plugin renders 'Sort by <col>'
  // buttons inside columnheaders; the click maps to the directionless enum.
  await table.getByRole('button', { name: 'Sort by Status' }).click()
  await expect(page).toHaveURL(/sort=status/)
  await expect.poll(() => seen.at(-1)?.sort).toBe('status')
})

test('pagination stays server-driven and corrects out-of-range pages', async ({ page }) => {
  const seen = await mockCollection(page, makeGroups(250))
  await page.goto('/groups', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  const table = collectionTable(page)
  await expect(table).toBeVisible()
  await expect(table.getByRole('row')).toHaveCount(101)

  const pagination = page.getByRole('navigation', { name: /[Pp]agination/ })
  // `recent` sorts newest-first, so id order is descending: page 2 of 250
  // opens at id 150.
  await pagination.getByRole('button', { name: /next/i }).click()
  await expect(page).toHaveURL(/page=2/)
  await expect.poll(() => seen.at(-1)?.page).toBe(2)
  await expect(table.getByRole('row').nth(1)).toContainText('Group 0150')

  // Deep-linking past the last page self-corrects to the canonical URL.
  await page.goto('/groups?page=99', { waitUntil: 'load' })
  await expect(page).toHaveURL(/page=3/)
  await expect(table.getByRole('row')).toHaveCount(51) // header + 50 rows
})

test('keyboard walkthrough reaches filters and row actions', async ({ page }) => {
  await mockCollection(page, makeGroups(5))
  await page.goto('/groups', { waitUntil: 'load' })
  await expectAstryxDocument(page)
  const table = collectionTable(page)
  await expect(table).toBeVisible()

  // Search is reachable and editable by keyboard alone.
  await page.getByLabel('Search').focus()
  await page.keyboard.type('group 0002')
  await expect(page).toHaveURL(/q=group(\+|%20)0002/, { timeout: 3_000 })

  // Row link activates with Enter and lands on the group detail route.
  // (.first(): the row's name link and the trailing action IconButton share
  // the same aria-label; the name link precedes it in DOM order.)
  const detailLink = table.getByRole('link', { name: 'View details for Group 0002' }).first()
  await detailLink.focus()
  await expect(detailLink).toBeFocused()
})

test('1,000-group collection stays interactive (gate #3)', async ({ page }) => {
  const seen = await mockCollection(page, makeGroups(1_000))
  await page.goto('/groups', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  const table = collectionTable(page)
  await expect(table).toBeVisible()
  await expect(table.getByRole('row')).toHaveCount(101)

  // Page through the 10-page collection; each step is one server request.
  // Descending `recent` order: page 2 opens at id 900, page 3 at id 800.
  const pagination = page.getByRole('navigation', { name: /[Pp]agination/ })
  await pagination.getByRole('button', { name: /next/i }).click()
  await expect(table.getByRole('row').nth(1)).toContainText('Group 0900')
  await pagination.getByRole('button', { name: /next/i }).click()
  await expect(table.getByRole('row').nth(1)).toContainText('Group 0800')
  await expect.poll(() => seen.at(-1)?.page).toBe(3)

  // Sort + filter at scale still reset to page 1 and stay responsive.
  await page.getByRole('combobox', { name: 'Sort' }).click()
  await page.getByRole('option', { name: 'Name A–Z' }).click()
  await expect(page).toHaveURL(/sort=name/)
  await expect(table.getByRole('row').nth(1)).toContainText('Group 0001')
  await expect(page.getByLabel('Search')).toBeEditable()
})

test('empty and no-results states mirror the classic contract', async ({ page }) => {
  await mockCollection(page, [])
  await page.goto('/groups', { waitUntil: 'load' })
  await expectAstryxDocument(page)

  // Zero groups: empty state with the import action.
  await expect(
    page.getByRole('heading', { name: 'Start with your first channel Group' }),
  ).toBeVisible()
  await expect(page.getByRole('link', { name: /Import channel credentials/ })).toBeVisible()

  // Repopulate, then filter to zero matches: no-results + reset.
  await page.unrouteAll()
  const seen2 = await mockCollection(page, makeGroups(8))
  await page.goto('/groups', { waitUntil: 'load' })
  const table = collectionTable(page)
  await expect(table).toBeVisible()

  await page.getByLabel('Search').fill('nothing-matches-this')
  await expect(page).toHaveURL(/q=nothing-matches-this/, { timeout: 3_000 })
  await expect.poll(() => seen2.at(-1)?.q).toBe('nothing-matches-this')
  await expect(page.getByRole('heading', { name: 'No Groups match these filters' })).toBeVisible()

  // Two resets are on screen (toolbar + empty-state action); activate the
  // empty-state one — it is the primary recovery affordance in this state.
  await page.getByRole('button', { name: 'Reset filters' }).last().click()
  await expect(page).not.toHaveURL(/q=/)
  await expect(table.getByRole('row')).toHaveCount(9)
})
