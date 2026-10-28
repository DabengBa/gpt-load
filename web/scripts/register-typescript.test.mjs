import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import test from 'node:test'

test('Node loads TypeScript tests that use transform-only syntax', () => {
  const result = spawnSync(
    process.execPath,
    [
      '--import',
      new URL('./register-typescript.mjs', import.meta.url).pathname,
      '--input-type=module',
      '--eval',
      `
        import assert from 'node:assert/strict'
        import { Status, Example } from './typescript-fixture.ts'
        assert.equal(new Example(42).value, 42)
        assert.equal(Status.Ready, 'ready')
      `,
    ],
    { cwd: new URL('.', import.meta.url), encoding: 'utf8' },
  )

  assert.equal(result.status, 0, result.stderr)
})
