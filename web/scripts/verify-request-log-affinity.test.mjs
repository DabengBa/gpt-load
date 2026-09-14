import assert from 'node:assert/strict'

export function runRequestLogAffinityContractTests({
  REQUEST_LOG_AFFINITY_KEY_PATTERN,
  isValidRequestLogAffinityKey,
  parseRequestLogAffinityKey,
  projectRequestLogAffinityKey,
  serializeRequestLogAffinityKey,
}) {
  const valid = '0123456789abcdef****fedcba9876543210'
  assert.equal(REQUEST_LOG_AFFINITY_KEY_PATTERN.test(valid), true)
  assert.equal(isValidRequestLogAffinityKey(valid), true)
  assert.deepEqual(parseRequestLogAffinityKey(valid), { kind: 'valid', value: valid })
  assert.equal(projectRequestLogAffinityKey(valid), valid)
  assert.equal(serializeRequestLogAffinityKey(valid), valid)

  for (const invalid of [
    '',
    '0123456789ABCDE****fedcba9876543210',
    '0123456789abcdef***fedcba9876543210',
    '0123456789abcdef*****fedcba9876543210',
    '0123456789abcdef****fedcba987654321',
    '0123456789abcdef****fedcba9876543210\n',
    '0123456789abcdef****fedcba98765432x0',
  ]) {
    assert.equal(isValidRequestLogAffinityKey(invalid), false, invalid)
    assert.deepEqual(parseRequestLogAffinityKey(invalid), { kind: 'invalid', raw: invalid })
    assert.equal(projectRequestLogAffinityKey(invalid), null)
    assert.throws(() => serializeRequestLogAffinityKey(invalid), /invalid affinity key/u)
  }

  assert.deepEqual(parseRequestLogAffinityKey(undefined), { kind: 'missing' })
  assert.deepEqual(parseRequestLogAffinityKey(['bad']), { kind: 'invalid', raw: ['bad'] })
  assert.equal(projectRequestLogAffinityKey(null), null)
}
