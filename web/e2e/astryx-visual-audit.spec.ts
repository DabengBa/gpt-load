import { mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test, type Page, type Route } from '@playwright/test'

import { installRequestLogDisplayRoutes } from './fixtures/request-log-display'

// TEMPORARY review-stage audit harness (final-review frontend pass).
// Captures light/dark screenshots and runs DOM-level audit checks for the
// three Astryx-flagged routes against both frontends (same paths; the
// selection cookie decides which document the dev server returns).
// Results land in <tmpdir>/astryx-audit/ — this spec is deleted after the
// review evidence is recorded.

const OUT = join(tmpdir(), 'astryx-audit')
mkdirSync(OUT, { recursive: true })

const report: Record<string, unknown> = {}

function groupsMock(authed: boolean) {
  return (route: Route) => {
    const path = new URL(route.request().url()).pathname
    const fulfill = (data: unknown) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data }),
      })
    if (path === '/api/auth/session') {
      return fulfill(authed ? { authenticated: true, principal_type: 'admin' } : { authenticated: false })
    }
  if (path === '/api/groups') {
    return fulfill({
      observed_at_ms: 1_700_000_000_000,
      summary: { total: 3, available: 2, unavailable: 1, disabled: 0 },
      items: [
        {
          id: 1,
          name: 'alpha',
          price_multiplier: '1',
          channel_id: 'openai',
          connection_type: 'api_key',
          params: {},
          provider_url: 'https://alpha.example/v1',
          status: 'available',
          model_count: 4,
          client_model_count: 2,
          credential_counts: {
            total: 2,
            available: 2,
            cooldown: 0,
            blacklisted: 0,
            disabled: 0,
          },
        },
        {
          id: 2,
          name: 'beta',
          price_multiplier: '1.5',
          channel_id: 'gemini',
          connection_type: 'subscription',
          params: {},
          provider_url: null,
          status: 'unavailable',
          model_count: 7,
          client_model_count: 7,
          credential_counts: {
            total: 5,
            available: 3,
            cooldown: 1,
            blacklisted: 1,
            disabled: 0,
          },
        },
        {
          id: 3,
          name: 'gamma-long-name-for-overflow-check',
          price_multiplier: '0.8',
          channel_id: 'anthropic',
          connection_type: 'api_key',
          params: {},
          provider_url: null,
          status: 'available',
          model_count: 1,
          client_model_count: 1,
          credential_counts: {
            total: 1,
            available: 1,
            cooldown: 0,
            blacklisted: 0,
            disabled: 0,
          },
        },
      ],
      pagination: { page: 1, page_size: 100, total_items: 3, total_pages: 1 },
    })
  }
  if (path === '/api/channels') {
    return fulfill({ items: [], total: 0 })
  }
  if (path === '/api/groups/options') {
    return fulfill([{ id: 1, name: 'alpha' }])
  }
  if (path === '/api/access-keys/options') {
    return fulfill([{ id: 7, name: 'e2e access key', status: 'active' }])
  }
  return fulfill({})
  }
}

async function installBaseMocks(page: Page, authed = true) {
  if (authed) {
    await page.addInitScript((key) => {
      window.localStorage.setItem('gpt-load.auth-key', key)
    }, 'e2e-auth-key')
  }
  await page.unrouteAll()
  await page.route('**/api/**', groupsMock(authed))
}

// WCAG contrast: relative luminance per sRGB, ratio of two rgb() strings.
const auditScript = `(() => {
  const rgb = (s) => {
    const m = s.match(/rgba?\\(([^)]+)\\)/)
    if (!m) return null
    const p = m[1].split(',').map((v) => parseFloat(v))
    return p.length >= 4 && p[3] === 0 ? null : p.slice(0, 3)
  }
  const lum = ([r, g, b]) => {
    const f = (c) => {
      c /= 255
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
    }
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
  }
  const ratio = (a, b) => {
    const [la, lb] = [lum(a), lum(b)].sort((x, y) => y - x)
    return (la + 0.05) / (lb + 0.05)
  }
  const bgOf = (el) => {
    let n = el
    while (n) {
      const c = rgb(getComputedStyle(n).backgroundColor)
      if (c) return c
      n = n.parentElement
    }
    return rgb(getComputedStyle(document.documentElement).backgroundColor) ?? [255, 255, 255]
  }
  const visible = (el) => {
    const r = el.getBoundingClientRect()
    const s = getComputedStyle(el)
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'
  }
  const label = (el) => {
    const direct = (el.getAttribute('aria-label') ||
      el.getAttribute('aria-labelledby') ||
      el.getAttribute('title') ||
      el.getAttribute('alt') ||
      el.innerText ||
      '').trim()
    if (direct) return direct
    // Implicit label association — visually-hidden Field labels still name
    // the control in the accessibility tree.
    const id = el.getAttribute('id')
    if (id && document.querySelector('label[for="' + id + '"]')) return 'via-label'
    const wrappingLabel = el.closest('label')
    if (wrappingLabel && wrappingLabel.innerText.trim()) return 'via-label'
    return ''
  }

  const interactive = [...document.querySelectorAll(
    'button, a[href], input, select, textarea, [role="button"], [role="link"], [role="tab"], summary',
  )].filter(visible)

  const unnamed = interactive
    .filter((el) => label(el) === '')
    .map((el) => el.outerHTML.slice(0, 90))

  const docEl = document.documentElement
  const overflowX = docEl.scrollWidth - docEl.clientWidth

  // Interactive elements fully under the 44x44 touch floor.
  const smallTargetsV2 = interactive
    .filter((el) => {
      const r = el.getBoundingClientRect()
      return r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth
    })
    .map((el) => {
      const r = el.getBoundingClientRect()
      return { tag: el.tagName.toLowerCase(), name: label(el).slice(0, 30), w: Math.round(r.width), h: Math.round(r.height) }
    })
    .filter((t) => t.w < 44 || t.h < 44)

  // Contrast on representative text nodes.
  const textEls = [...document.querySelectorAll('h1, h2, h3, p, span, td, th, label, a, button, [class*="title"], [class*="label"]')]
    .filter((el) => visible(el) && el.childNodes.length > 0 &&
      [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim().length > 0))
  const lowContrast = []
  for (const el of textEls.slice(0, 400)) {
    const s = getComputedStyle(el)
    const fg = rgb(s.color)
    if (!fg) continue
    const r = ratio(fg, bgOf(el))
    const large = parseFloat(s.fontSize) >= 18 || (parseFloat(s.fontSize) >= 14 && parseInt(s.fontWeight) >= 700)
    if (r < (large ? 3 : 4.5)) {
      lowContrast.push({ tag: el.tagName.toLowerCase(), cls: (el.className || '').toString().slice(0, 40), text: el.innerText.slice(0, 30), ratio: Math.round(r * 100) / 100, size: s.fontSize })
    }
  }

  const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].map((h) => h.tagName)
  const landmarks = {
    main: !!document.querySelector('main,[role="main"]'),
    banner: !!document.querySelector('header,[role="banner"]'),
    nav: !!document.querySelector('nav,[role="navigation"]'),
  }
  const imgsNoAlt = [...document.querySelectorAll('img')].filter((i) => !i.hasAttribute('alt')).length
  const divButtons = [...document.querySelectorAll('div[onclick], span[onclick]')].length

  return {
    lang: document.documentElement.lang || null,
    title: document.title,
    interactiveCount: interactive.length,
    unnamedInteractive: unnamed,
    overflowX,
    smallTargets: smallTargetsV2.slice(0, 30),
    lowContrast: lowContrast.slice(0, 30),
    lowContrastCount: lowContrast.length,
    headings,
    landmarks,
    imgsNoAlt,
    divButtons,
    domNodes: document.getElementsByTagName('*').length,
  }
})()`

async function auditAndShoot(
  page: Page,
  frontend: 'classic' | 'astryx',
  route: string,
  width: number,
  theme: 'light' | 'dark',
) {
  await page.setViewportSize({ width, height: 800 })
  await page.addInitScript(
    ([k, v]) => window.localStorage.setItem(k, v),
    ['gpt-load.theme', theme],
  )
  await page.goto(route, { waitUntil: 'networkidle' })
  // give React/classic paint a beat beyond networkidle
  await page.waitForTimeout(400)
  const key = `${frontend}${route.replace(/\//g, '_')}-${width}-${theme}`
  await page.screenshot({ path: join(OUT, `${key}.png`), fullPage: false })
  report[key] = await page.evaluate(auditScript)
}

test('visual parity + a11y audit across frontends', async ({ page, context, baseURL }) => {
  test.setTimeout(180_000)
  const astryxCookie = {
    name: 'gpt-load.frontend',
    value: 'astryx',
    url: baseURL as string,
    sameSite: 'Strict' as const,
  }

  // Phase A — login is an unauthenticated surface for both frontends.
  await installBaseMocks(page, false)
  for (const theme of ['light', 'dark'] as const) {
    await auditAndShoot(page, 'astryx', '/login', 1280, theme)
  }
  await context.clearCookies()
  for (const theme of ['light', 'dark'] as const) {
    await auditAndShoot(page, 'classic', '/login', 1280, theme)
  }

  // Phase B — authenticated collection surface.
  await installBaseMocks(page, true)
  await context.addCookies([astryxCookie])
  for (const theme of ['light', 'dark'] as const) {
    await auditAndShoot(page, 'astryx', '/groups', 1280, theme)
  }
  await auditAndShoot(page, 'astryx', '/groups', 390, 'light')
  await context.clearCookies()
  for (const theme of ['light', 'dark'] as const) {
    await auditAndShoot(page, 'classic', '/groups', 1280, theme)
  }
  await auditAndShoot(page, 'classic', '/groups', 390, 'light')

  // Phase C — logs surface incl. the detail overlay (B11/B12 ground).
  await installRequestLogDisplayRoutes(page)
  await context.addCookies([astryxCookie])
  await auditAndShoot(page, 'astryx', '/logs', 1280, 'light')
  await page.goto(`/logs?selected_request_id=${'aaaaaaaa-1111-4111-8111-111111111111'}`, {
    waitUntil: 'networkidle',
  })
  await page.waitForTimeout(400)
  await page.screenshot({ path: join(OUT, 'astryx_logs-detail-1280-light.png') })
  report['astryx_logs-detail-1280-light'] = await page.evaluate(auditScript)
  await context.clearCookies()
  await auditAndShoot(page, 'classic', '/logs', 1280, 'light')

  writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 1))
  expect(true).toBe(true)
})
