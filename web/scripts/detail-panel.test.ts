import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const detailPanelPath = fileURLToPath(
  new URL('../src/frontends/astryx/components/DetailPanel.tsx', import.meta.url),
)

test('DetailPanel exposes an optional, non-focusable context notice below the header', async () => {
  const source = await readFile(detailPanelPath, 'utf8')

  assert.match(source, /contextNotice\?: ReactNode/)
  assert.match(source, /contextNotice,/)
  assert.match(source, /contextNotice !== undefined && contextNotice !== null/)
  assert.match(source, /data-detail-panel-context-notice/)
  assert.match(source, /tabIndex=\{-1\}/)

  // The notice renders inside the LayoutContent slot, which Layout places
  // after the header slot; assert the JSX ordering directly.
  const header = source.indexOf('<DialogHeader')
  const content = source.indexOf('<LayoutContent')
  const notice = source.indexOf('data-detail-panel-context-notice')
  assert.ok(header >= 0 && content > header, 'content slot must come after the header slot')
  assert.ok(notice > content, 'notice must render inside the content slot')
})

test('DetailPanel context notice remains width-safe on narrow panels', async () => {
  const source = await readFile(detailPanelPath, 'utf8')

  assert.match(source, /contextNotice:\s*\{[^}]*maxWidth:\s*'100%'/s)
  assert.match(source, /overflowWrap:\s*'anywhere'/)
})
