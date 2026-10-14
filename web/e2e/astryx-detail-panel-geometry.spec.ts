import { expect, test, type Page } from '@playwright/test'

import { installRequestLogDisplayRoutes, requestIDs } from './fixtures/request-log-display'

// U005 visual-polish geometry contract for the shared DetailPanel:
//   - header / footer edges align exactly with the padded content box
//     (header and footer carry no own padding; the content gutter is the
//     single alignment source)
//   - desktop gutter is 24px; <=520px full-bleed drawers use 18px
//   - complex drawers opt into the wider 640px surface via `width`;
//     simple drawers stay at the compact 520px default
//   - the sticky context notice paints an explicit surface background so
//     scrolling body content never shows through it
//   - a long title next to the close button truncates inside the header
//     instead of clipping (no horizontal overflow in the header band)

const dialog = (page: Page) => page.getByRole('dialog', { name: 'Request log details' })

interface PanelGeometry {
  panelWidth: number
  header: { left: number; right: number; padL: number; padR: number; overflowX: number }
  content: { left: number; right: number; padL: number; padR: number; overflowX: number }
  footer: { left: number; right: number; padL: number; padR: number; overflowX: number } | null
  noticeBg: string | null
  noticeZ: string | null
}

async function panelGeometry(page: Page): Promise<PanelGeometry> {
  return page.getByRole('dialog').evaluate((node): PanelGeometry => {
    const headerEl = node.querySelector('.astryx-layout-header') as HTMLElement | null
    const footerEl = node.querySelector('.astryx-layout-footer') as HTMLElement | null
    const contentEl = node.querySelector('.astryx-layout-content') as HTMLElement | null
    const notice = node.querySelector('[data-detail-panel-context-notice]') as HTMLElement | null
    const band = (el: HTMLElement | null) => {
      if (!el) return null
      const cs = getComputedStyle(el)
      const r = el.getBoundingClientRect()
      return {
        left: Math.round(r.left),
        right: Math.round(r.right),
        padL: parseFloat(cs.paddingLeft),
        padR: parseFloat(cs.paddingRight),
        overflowX: el.scrollWidth - el.clientWidth,
      }
    }
    if (!headerEl) throw new Error('no header band found')
    const header = band(headerEl)!
    return {
      panelWidth: Math.round(node.getBoundingClientRect().width),
      header,
      footer: band(footerEl),
      content: band(contentEl) ?? { left: 0, right: 0, padL: 0, padR: 0, overflowX: 0 },
      noticeBg: notice ? getComputedStyle(notice).backgroundColor : null,
      noticeZ: notice ? getComputedStyle(notice).zIndex : null,
    }
  })
}

test.describe(() => {
  test.describe.configure({ mode: 'serial' })

  let page: Page

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage()
    await installRequestLogDisplayRoutes(page, undefined, undefined, 'admin')
    await page.goto(`/logs?selected_request_id=${requestIDs.mapped}`, { waitUntil: 'commit' })
    await expect(page.locator('[data-testid="astryx-shell"]')).toBeVisible({ timeout: 60_000 })
    await expect(dialog(page)).toBeVisible()
  })

  test.afterAll(async () => {
    await page.close()
  })

  test('desktop: 640px wide drawer, 24px gutter, aligned header/footer, opaque sticky notice', async () => {
    await page.setViewportSize({ width: 1280, height: 900 })
    await expect(dialog(page)).toBeVisible()
    const geo = await panelGeometry(page)

    expect(geo.panelWidth).toBe(640)
    // Alignment: header and footer edges sit exactly on the padded content
    // box (content pad 24 on both sides).
    expect(geo.content.padL).toBe(24)
    expect(geo.content.padR).toBe(24)
    // Header and content share the drawer's left edge; the content band
    // carries the shared 24px gutter (the header has none of its own).
    expect(geo.header.padL).toBe(0)
    // When a context notice renders, it must carry an explicit opaque
    // surface background above the scrolling body. The log detail drawer
    // has none (its subtitle carries the scope line); the notice background
    // contract is pinned on the model upstream drawer journey.
    if (geo.noticeBg !== null) {
      expect(geo.noticeBg).not.toBe('rgba(0, 0, 0, 0)')
    }
    // No horizontal clipping anywhere.
    expect(geo.header.overflowX).toBe(0)
    expect(geo.content.overflowX).toBe(0)
  })

  test('mobile 375px: full-bleed drawer, 18px gutter, footer does not overflow', async () => {
    await page.setViewportSize({ width: 375, height: 812 })
    await expect(dialog(page)).toBeVisible()
    const geo = await panelGeometry(page)

    expect(geo.panelWidth).toBe(375)
    expect(geo.content.padL).toBe(18)
    expect(geo.content.padR).toBe(18)
    expect(geo.footer === null || geo.footer!.overflowX === 0).toBe(true)
    expect(geo.header.overflowX).toBe(0)
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      ),
    ).toBe(0)
  })

  test('mobile 320px: long title truncates instead of clipping the header', async () => {
    await page.setViewportSize({ width: 320, height: 700 })
    await expect(dialog(page)).toBeVisible()
    const geo = await panelGeometry(page)

    expect(geo.panelWidth).toBe(320)
    expect(geo.content.padL).toBe(18)
    expect(geo.header.overflowX).toBe(0)
    expect(geo.content.overflowX).toBe(0)
    // The close button's -8px bleed intentionally tucks into the header
    // padding; the real contract is that the button stays fully inside the
    // drawer edge and the title block clips the long title (single-line
    // ellipsis) instead of pushing the button out.
    const geometry = await dialog(page)
      .locator('.astryx-dialog-header-title-block')
      .first()
      .evaluate((el) => {
        const heading = el.querySelector('.astryx-heading') as HTMLElement
        const close = el.parentElement!.querySelector('button') as HTMLElement | null
        const drawer = el.closest('dialog') as HTMLElement
        // The heading clips: its rendered text box fits inside the clipped
        // title block (scrollWidth no longer exceeds clientWidth once the
        // theme overflow rule applies).
        return {
          titleClips: el.scrollWidth <= el.clientWidth,
          closeRight: close ? Math.round(close.getBoundingClientRect().right) : null,
          drawerRight: Math.round(drawer.getBoundingClientRect().right),
          headingText: heading.textContent,
        }
      })
    expect(geometry.headingText).toBe('Request log details')
    expect(geometry.titleClips).toBe(true)
    expect(geometry.closeRight).not.toBeNull()
    expect(geometry.closeRight!).toBeLessThanOrEqual(geometry.drawerRight)
  })

  test('short viewport: title and subtitle keep vertical breathing room inside the header band', async () => {
    // Defect regression (log drawer, 584px viewport): the zero-padding
    // dialog surface collapsed --layout-padding-outer-y/inner-y to 0, and
    // DialogHeader's negative title margin (≈ -7px) pushed the title above
    // the drawer's top edge while the subtitle sat flush against it.
    await page.setViewportSize({ width: 1280, height: 584 })
    await expect(dialog(page)).toBeVisible()
    const geometry = await dialog(page).evaluate((node) => {
      const layoutEl = node.querySelector('.astryx-layout') as HTMLElement
      const cs = getComputedStyle(layoutEl)
      const headerEl = node.querySelector('.astryx-layout-header') as HTMLElement
      const headerInner = headerEl.firstElementChild as HTMLElement
      const titleBlock = node.querySelector('.astryx-dialog-header-title-block') as HTMLElement
      const heading = titleBlock.querySelector('.astryx-heading') as HTMLElement
      const subtitle = titleBlock.querySelector(
        'p, [data-theme~="text"], span, div:not(:has(*))',
      ) as HTMLElement | null
      const subtitleEl =
        subtitle && subtitle !== heading
          ? subtitle
          : (titleBlock.children[1] as HTMLElement | undefined)
      const contentEl = node.querySelector('.astryx-layout-content') as HTMLElement
      const headerRect = headerEl.getBoundingClientRect()
      const titleRect = heading.getBoundingClientRect()
      const subtitleRect = subtitleEl?.getBoundingClientRect()
      return {
        outerY: cs.getPropertyValue('--layout-padding-outer-y').trim(),
        headerPadTop: parseFloat(getComputedStyle(headerInner).paddingTop),
        headerTop: headerRect.top,
        titleTop: titleRect.top,
        subtitleTop: subtitleRect?.top ?? null,
        subtitleBottom: subtitleRect ? subtitleRect.bottom : null,
        headerBottom: headerRect.bottom,
        contentOutline: getComputedStyle(contentEl).outlineStyle,
      }
    })

    // The Layout must carry its own vertical padding inside the drawer so the
    // header's negative-margin alignment contract has room to breathe.
    expect(geometry.outerY).not.toBe('0px')
    expect(geometry.headerPadTop).toBeGreaterThan(0)
    // Title fully inside the drawer (was y = -7.2 before the fix).
    expect(geometry.titleTop).toBeGreaterThanOrEqual(geometry.headerTop)
    // Subtitle below the title and fully above the header divider.
    expect(geometry.subtitleTop).not.toBeNull()
    expect(geometry.subtitleTop!).toBeGreaterThanOrEqual(geometry.titleTop)
    expect(geometry.subtitleBottom!).toBeLessThanOrEqual(geometry.headerBottom)
    // The programmatic content focus anchor must not paint the UA outline
    // around the whole scrollport (interactive controls keep their own
    // focus-visible indicators).
    expect(geometry.contentOutline).toBe('none')
  })

  test('subtitle is not clipped by any overflow ancestor (no inner header clip)', async () => {
    // Defect regression: a previous fix added `overflow: hidden` on the
    // inner .astryx-dialog-header. DialogHeader's title block carries a
    // negative margin-block, so the inner container's border box ends
    // before the subtitle's second half and the inner clip amputated it.
    // The outer header band already has the vertical padding that absorbs
    // the close button bleed; horizontal truncation is owned by the title
    // block itself.
    await page.setViewportSize({ width: 1280, height: 584 })
    await expect(dialog(page)).toBeVisible()
    const clipped = await dialog(page).evaluate((node) => {
      const titleBlock = node.querySelector('.astryx-dialog-header-title-block') as HTMLElement
      const heading = titleBlock.querySelector('.astryx-heading') as HTMLElement
      const subtitle = titleBlock.lastElementChild as HTMLElement
      if (subtitle === heading) throw new Error('no subtitle element found')
      const sub = subtitle.getBoundingClientRect()
      const clips: Array<{
        label: string
        clipTop: number
        clipBottom: number
        subTop: number
        subBottom: number
      }> = []
      let ancestor = subtitle.parentElement
      while (ancestor && ancestor !== node.parentElement) {
        const cs = getComputedStyle(ancestor)
        if (/(hidden|clip|auto|scroll)/.test(cs.overflowY) || /(hidden|clip)/.test(cs.overflowX)) {
          const r = ancestor.getBoundingClientRect()
          if (sub.top < r.top - 0.5 || sub.bottom > r.bottom + 0.5) {
            clips.push({
              label: `${ancestor.tagName}.${(ancestor.className as string).split(' ')[0] ?? ''}`,
              clipTop: r.top,
              clipBottom: r.bottom,
              subTop: sub.top,
              subBottom: sub.bottom,
            })
          }
        }
        ancestor = ancestor.parentElement
      }
      return clips
    })
    expect(clipped).toEqual([])
  })

  test('zh-CN: subtitle renders fully inside the header on a short viewport', async () => {
    // Independent Chinese-locale coverage: the default spec page runs
    // en-US; this journey drives the real zh-CN route (请求日志详情) so the
    // subtitle line's legibility is proven in the shipped locale, not just
    // geometry-neutral English copy.
    const zhPage = await page.context().browser()!.newPage({ locale: 'zh-CN' })
    try {
      await installRequestLogDisplayRoutes(zhPage, undefined, undefined, 'admin')
      await zhPage.goto(`/logs?selected_request_id=${requestIDs.mapped}`, { waitUntil: 'commit' })
      await expect(zhPage.locator('[data-testid="astryx-shell"]')).toBeVisible({ timeout: 60_000 })
      await zhPage.setViewportSize({ width: 1280, height: 584 })
      const zhDialog = zhPage.getByRole('dialog', { name: '请求日志详情' })
      await expect(zhDialog).toBeVisible()
      const result = await zhDialog.evaluate((node) => {
        const titleBlock = node.querySelector('.astryx-dialog-header-title-block') as HTMLElement
        const heading = titleBlock.querySelector('.astryx-heading') as HTMLElement
        const subtitle = titleBlock.lastElementChild as HTMLElement
        const sub = subtitle.getBoundingClientRect()
        const head = heading.getBoundingClientRect()
        const drawer = node.getBoundingClientRect()
        const clippingAncestors: string[] = []
        for (let ancestor = subtitle.parentElement; ancestor; ancestor = ancestor.parentElement) {
          const cs = getComputedStyle(ancestor)
          if (/(hidden|clip|auto|scroll)/.test(cs.overflowY)) {
            const rect = ancestor.getBoundingClientRect()
            if (sub.top < rect.top - 0.5 || sub.bottom > rect.bottom + 0.5) {
              clippingAncestors.push(ancestor.className)
            }
          }
        }
        return {
          clippingAncestors,
          titleText: heading.textContent,
          subtitleText: subtitle.textContent,
          titleTop: head.top,
          subTop: sub.top,
          subBottom: sub.bottom,
          drawerTop: drawer.top,
        }
      })
      expect(result.titleText).toBe('请求日志详情')
      expect(result.clippingAncestors).toEqual([])
      expect(result.subtitleText!.length).toBeGreaterThan(0)
      expect(result.titleTop).toBeGreaterThanOrEqual(result.drawerTop)
      expect(result.subTop).toBeGreaterThanOrEqual(result.titleTop)
      expect(result.subBottom).toBeLessThanOrEqual(result.drawerTop + 584)
    } finally {
      await zhPage.close()
    }
  })
})
