import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

// U003 contract: both ModelDiscoveryDrawer call scenarios (GroupModelsTab and
// NewGroupImport) must wire the evidence-legend and scoped-selection labels,
// and ModelUpstreamDrawer must surface the save/reset scope through the frozen
// DetailPanel contextNotice.

const read = (path: string) => readFile(fileURLToPath(new URL(path, import.meta.url)), 'utf8')

test('discovery drawer renders the legend from labels without tooltip-only evidence', async () => {
  const source = await read('../src/frontends/astryx/features/models/ModelDiscoveryDrawer.tsx')

  assert.match(source, /labels\.legendLabel/)
  assert.match(source, /labels\.pricingLegend/)
  assert.match(source, /labels\.selectedDetail\(/)
  assert.match(source, /aria-label=\{labels\.legendLabel\}/)
})

test('the labels type carries the legend and scoped-selection contract', async () => {
  const source = await read('../src/shared/domain/models/model-draft.ts')

  assert.match(source, /legendLabel: string/)
  assert.match(source, /pricingLegend: string/)
  assert.match(source, /selectedDetail: \(selected: number, available: number\) => string/)
})

for (const caller of [
  '../src/frontends/astryx/features/groups/models/GroupModelsTab.tsx',
  '../src/frontends/astryx/features/import/NewGroupImport.tsx',
]) {
  test(`${caller.split('/').at(-1)} wires the discovery legend labels`, async () => {
    const source = await read(caller)

    assert.match(source, /legendLabel: t\(/)
    assert.match(source, /pricingLegend: t\(/)
    assert.match(source, /selectedDetail: \(selected, available\) =>/)
  })
}

test('upstream model drawer keeps the save/reset scope in the context notice', async () => {
  const source = await read('../src/frontends/astryx/features/models/ModelUpstreamDrawer.tsx')

  assert.match(source, /contextNotice=/)
  assert.match(source, /models\.drawer\.contextImpact/)
  assert.match(source, /models\.drawer\.saveScope/)
})
