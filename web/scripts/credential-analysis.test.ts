import assert from 'node:assert/strict'
import test from 'node:test'

import { analyzeCredentials } from '../src/shared/domain/import/credential-analysis.ts'

test('one complete JSON credential counts once without channel restrictions', () => {
  const raw = '  {\r\n  "api_key": "test-placeholder",\r\n  "nested": {"value": "test"}\r\n}  '
  const analysis = analyzeCredentials(raw)
  assert.equal(analysis.raw, raw)
  assert.equal(analysis.nonEmptyCount, 1)
  assert.equal(analysis.tooManyCredentials, false)
  assert.equal(analysis.emptyLineCount, 0)
})

test('subscription JSON remains one object rather than token lines', () => {
  assert.equal(
    analyzeCredentials('{\n"access_token":"test",\n"refresh_token":"test"\n}').tooManyCredentials,
    false,
  )
})

test('incomplete JSON, arrays, and multiple objects are not treated as one credential', () => {
  for (const raw of [
    '{\n"api_key":"test"',
    '[\n{"api_key":"test"}\n]',
    '{"api_key":"first"}\n{"api_key":"second"}',
  ]) {
    assert.equal(analyzeCredentials(raw).tooManyCredentials, true)
  }
})

test('plain distinct keys, duplicate lines, empty lines, and access keys retain their semantics', () => {
  assert.equal(analyzeCredentials('first\nsecond').tooManyCredentials, true)
  assert.deepEqual(analyzeCredentials('sk-gl-test\n\nsk-gl-test'), {
    raw: 'sk-gl-test\n\nsk-gl-test',
    nonEmptyCount: 2,
    emptyLineCount: 1,
    duplicateCount: 1,
    likelyAccessKeyCount: 2,
    tooManyCredentials: false,
  })
})
