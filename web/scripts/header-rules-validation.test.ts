import assert from 'node:assert/strict'
import test from 'node:test'

import {
  validateHeaderRuleRows,
  type HeaderRuleInput,
} from '../src/shared/domain/settings/header-rules-validation.ts'

// Contract tests for the shared header-rules validator — both the settings
// editors (request + response policies) and the group-settings editor consume
// this, so the error codes are the cross-frontend API surface.

function row(
  rowKey: number,
  action: HeaderRuleInput['action'],
  name: string,
  value = '',
): HeaderRuleInput {
  return { rowKey, action, name, value }
}

function codes(errors: ReturnType<typeof validateHeaderRuleRows>): string[] {
  return errors.map((error) => error.code)
}

test('valid set/remove rows produce no errors', () => {
  assert.deepEqual(
    validateHeaderRuleRows([
      row(1, 'set', 'x-custom', 'value'),
      row(2, 'remove', 'x-legacy'),
    ]),
    [],
  )
})

test('missing and malformed names are reported per row', () => {
  const errors = validateHeaderRuleRows([
    row(1, 'set', '', 'value'),
    row(2, 'set', 'bad name!', 'v'),
    row(3, 'set', 'ok-name', 'v'),
  ])
  assert.deepEqual(errors, [
    { code: 'required', rowKey: 1 },
    { code: 'invalid_name', rowKey: 2 },
  ])
})

test('duplicate names match case-insensitively and flag both rows', () => {
  const errors = validateHeaderRuleRows([
    row(1, 'set', 'X-Thing', 'a'),
    row(2, 'remove', 'x-thing'),
    row(3, 'set', 'x-other', 'b'),
  ])
  assert.deepEqual(codes(errors), ['duplicate_name', 'duplicate_name'])
  assert.deepEqual(
    errors.map((error) => error.rowKey).sort(),
    [1, 2],
  )
})

test('credential and hop-by-hop names are forbidden for request rules', () => {
  for (const name of ['authorization', 'X-API-Key', 'Cookie', 'connection', 'transfer-encoding']) {
    assert.deepEqual(codes(validateHeaderRuleRows([row(1, 'set', name, 'v')])), [
      'forbidden_set_name',
    ])
  }
  assert.deepEqual(
    codes(validateHeaderRuleRows([row(1, 'set', 'proxy-whatever', 'v')])),
    ['forbidden_set_name'],
  )
})

test('the response policy additionally forbids protocol-managed headers', () => {
  const extra = ['content-type', 'set-cookie', 'access-control-allow-origin', 'vary']
  for (const name of extra) {
    assert.deepEqual(
      codes(validateHeaderRuleRows([row(1, 'set', name, 'v')], 'response')),
      ['forbidden_set_name'],
      name,
    )
    // The same names are legal for request rules.
    assert.deepEqual(validateHeaderRuleRows([row(1, 'set', name, 'v')], 'request'), [])
  }
  assert.deepEqual(
    codes(validateHeaderRuleRows([row(1, 'set', 'x-gptload-internal', 'v')], 'response')),
    ['forbidden_set_name'],
  )
})

test('remove rules skip value validation; set rules reject control bytes', () => {
  assert.deepEqual(validateHeaderRuleRows([row(1, 'remove', 'x-any', '\u0007')]), [])
  assert.deepEqual(codes(validateHeaderRuleRows([row(1, 'set', 'x-any', 'a\u0007b')])), [
    'invalid_value',
  ])
  assert.deepEqual(validateHeaderRuleRows([row(1, 'set', 'x-any', 'a\tb')]), [])
})
