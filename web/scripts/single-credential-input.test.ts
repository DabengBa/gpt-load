import assert from 'node:assert/strict'
import test from 'node:test'

import { readSingleCredential } from '../src/frontends/astryx/features/import/single-credential-input.ts'

test('one key and one structured object are accepted', () => {
  assert.deepEqual(readSingleCredential('  sk-provider  '), {
    credential: 'sk-provider',
    error: null,
  })
  assert.deepEqual(readSingleCredential('{\n "api_key": "value"\n}'), {
    credential: '{\n "api_key": "value"\n}',
    error: null,
  })
})

test('multiple values including repeated keys are rejected without deduplication', () => {
  for (const input of ['first\nsecond', 'same\nsame', 'first\r\nsecond']) {
    assert.deepEqual(readSingleCredential(input), { credential: null, error: 'tooMany' })
  }
})

test('empty input is distinguished from malformed JSON and JSON arrays', () => {
  for (const input of ['', ' \n ']) {
    assert.deepEqual(readSingleCredential(input), { credential: null, error: 'required' })
  }
  for (const input of ['{"api_key":', '[{"api_key":"a"},{"api_key":"b"}]', '[]']) {
    assert.deepEqual(readSingleCredential(input), { credential: null, error: 'invalidJSON' })
  }
})
