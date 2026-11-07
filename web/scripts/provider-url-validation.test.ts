import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import {
  extractSiteRootUrl,
  isValidProviderUrlForWrite,
} from '../src/shared/domain/groups/settings/group-settings-patch.ts'

const cases = JSON.parse(
  readFileSync(
    new URL('../../internal/channel/spec/testdata/provider_url_cases.json', import.meta.url),
    'utf8',
  ),
) as { input: string; valid: boolean }[]
for (const { input, valid } of cases) {
  test(`provider URL ${input}`, () => {
    assert.equal(isValidProviderUrlForWrite(input), valid)
  })
}

test('historical oversized provider ports remain readable', () => {
  assert.equal(extractSiteRootUrl('https://example.com:99999/path'), 'https://example.com:99999')
})
