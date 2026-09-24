import assert from 'node:assert/strict'
import { createServer } from 'vite'
import { fileURLToPath } from 'node:url'

const webRoot = fileURLToPath(new URL('..', import.meta.url))
const server = await createServer({
  root: webRoot,
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})

try {
  const groups = await server.ssrLoadModule('/src/frontends/classic/app/resources/groups.ts')
  const modelDiff = await server.ssrLoadModule(
    '/src/frontends/classic/features/groups/models/model-diff.ts',
  )
  const payload = {
    items: [
      {
        id: 'provider-model',
        alias: '',
        alias_enabled: false,
        client_model: 'provider-model',
        test_alias: 'a4g233',
        entry_id: 'entry-1',
        weight: null,
        priority: null,
        pricing_status: 'configured',
      },
    ],
    total: 1,
    pending: 0,
  }

  const projected = groups.projectGroupModels(payload)
  assert.equal(projected.items[0].test_alias, 'a4g233')

  for (const testAlias of [undefined, '', 'A4g233', 'a4g23', 'a4g2333']) {
    const invalid = structuredClone(payload)
    if (testAlias === undefined) delete invalid.items[0].test_alias
    else invalid.items[0].test_alias = testAlias
    assert.throws(
      () => groups.projectGroupModels(invalid),
      `accepted invalid test_alias ${testAlias}`,
    )
  }

  const secretField = structuredClone(payload)
  secretField.items[0].token = 1
  assert.throws(() => groups.projectGroupModels(secretField), 'accepted secret-like field')

  const draft = modelDiff.createModelDraft(projected.items)
  assert.equal(draft[0].test_alias, 'a4g233')
  const normalized = modelDiff.normalizedModels(draft)
  assert.equal(normalized[0].entry_id, 'entry-1')
  assert.equal(Object.hasOwn(normalized[0], 'test_alias'), false)

  const changedServerAlias = modelDiff.createModelDraft([
    { ...projected.items[0], test_alias: 'b5h244' },
  ])
  assert.equal(modelDiff.sameModels(draft, changedServerAlias), true)
  console.log('model-test-alias contract: PASS')
} finally {
  await server.close()
}
