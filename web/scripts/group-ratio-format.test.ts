import assert from 'node:assert/strict'
import test from 'node:test'
import { formatGroupRatio } from '../src/shared/domain/groups/settings/group-settings-patch.ts'

test('group ratios keep their real precision instead of collapsing to zero', () => {
  assert.equal(formatGroupRatio(0.04), '0.04')
  assert.equal(formatGroupRatio(0.004), '0.004')
  assert.equal(formatGroupRatio(0.125), '0.125')
  assert.equal(formatGroupRatio(0.123456), '0.123456')
  assert.equal(formatGroupRatio(1.5), '1.5')
  assert.equal(formatGroupRatio(2.0001), '2.0001')
})

test('whole-number ratios still render with one decimal place', () => {
  assert.equal(formatGroupRatio(0), '0.0')
  assert.equal(formatGroupRatio(1), '1.0')
  assert.equal(formatGroupRatio(12), '12.0')
})
