import assert from 'node:assert/strict'
import test from 'node:test'

import { readSingleCredential } from '../src/frontends/astryx/features/import/single-credential-input.ts'

test('one key and one structured object are accepted', () => {
  assert.equal(readSingleCredential('  sk-provider  '), 'sk-provider')
  assert.equal(readSingleCredential('{\n "api_key": "value"\n}'), '{\n "api_key": "value"\n}')
})

test('multiple values including repeated keys are rejected without deduplication', () => {
  for (const input of ['first\nsecond', 'same\nsame', '[{"api_key":"a"},{"api_key":"b"}]', '']) {
    assert.equal(readSingleCredential(input), null)
  }
})
