import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

import { channelIconImage } from '../src/frontends/astryx/components/channel-icon-image.ts'

test('U006 preset SVG assets use the existing native image rendering path', async () => {
  for (const name of [
    'cerebras',
    'huggingface',
    'mistral',
    'nebius',
    'parasail',
    'wafer',
    'opencode',
    'cohere',
  ]) {
    const markup = await readFile(
      new URL(`../src/shared/assets/channels/${name}.svg`, import.meta.url),
      'utf8',
    )
    assert.match(markup, /<svg/u)
    assert.ok(['mask', 'img'].includes(channelIconImage(markup).mode))
  }
})

const componentSourceURL = new URL(
  '../src/frontends/astryx/components/ChannelIcon.tsx',
  import.meta.url,
)

test('ChannelIcon renders vendored SVG through native image/mask, never raw HTML', async () => {
  const source = await readFile(componentSourceURL, 'utf8')
  assert.doesNotMatch(source, /dangerouslySetInnerHTML/u)
  assert.doesNotMatch(source, /\.innerHTML\s*=/u)
  assert.match(source, /image\.mode === 'mask'/u)
  assert.match(source, /<img/u)
})

test('monochrome currentColor icons become masks so they keep inheriting text color', () => {
  const plan = channelIconImage('<svg fill="currentColor"><path d="M0 0"/></svg>')
  assert.equal(plan.mode, 'mask')
})

test('multicolor icons keep their own colors as images', () => {
  const plan = channelIconImage('<svg><path fill="#3186FF" d="M0 0"/></svg>')
  assert.equal(plan.mode, 'img')
})

test('image source round-trips the vendored markup', () => {
  const markup = '<svg viewBox="0 0 24 24"><path d="M0 0h24v24H0z"/></svg>'
  const plan = channelIconImage(markup)
  assert.match(plan.src, /^data:image\/svg\+xml;charset=utf-8,/u)
  assert.equal(decodeURIComponent(plan.src.slice(plan.src.indexOf(',') + 1)), markup)
})
