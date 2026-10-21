import assert from 'node:assert/strict'
import test from 'node:test'
import { createImportRecoveryService } from '../src/shared/controllers/import-recovery.ts'

const draft = () => ({
  mode: 'new',
  channel_id: 'openai',
  connection_type: 'api_key',
  params: {},
  proxy: { mode: 'inherit', url: '' },
  name: 'x',

  provider_url: '',
  credentials: '',

  models: [],
})

function roundTrip(value: unknown) {
  const storage = new Map<string, string>()
  const controller = createImportRecoveryService({
    storage: {
      getItem: (key) => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: (key) => storage.delete(key),
      get length() {
        return storage.size
      },
      clear() {
        storage.clear()
      },
      key() {
        return null
      },
    } as Storage,
    now: () => 1000,
    setTimer: () => 0 as ReturnType<typeof setTimeout>,
    clearTimer: () => {},
  })

  controller.register(() => value as never)
  assert.equal(controller.captureForUnauthorized(), 'stored')
  return controller.consume()
}

test('API-only recovery preserves API/cloud/external gateway drafts without staging', () => {
  for (const channel_id of ['openai', 'anthropic', 'azure', 'vertex', 'bedrock', 'cliproxyapi']) {
    const value = { ...draft(), channel_id, credentials: '{"api_key":"test-only"}' }
    assert.deepEqual(roundTrip(value), value)
  }
  const existing = { mode: 'existing', group_id: 9, credentials: 'test-only' }
  assert.deepEqual(roundTrip(existing), existing)
})

test('API-only recovery rejects retired connection kinds and staging fields', () => {
  assert.equal(
    roundTrip({ ...draft(), connection_type: 'subscription', staged_credential: null }),
    null,
  )
  assert.equal(roundTrip({ ...draft(), staged_credential: null }), null)
  assert.equal(roundTrip({ ...draft(), staged_credentials: [] }), null)
  assert.equal(
    roundTrip({ mode: 'existing', group_id: 9, credentials: '', staged_credential: null }),
    null,
  )
})
