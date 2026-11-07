import assert from 'node:assert/strict'
import test from 'node:test'
import { createImportRecoveryService } from '../src/shared/controllers/import-recovery.ts'

const draft = () => ({
  mode: 'new',
  channel_id: 'openai',
  params: {},
  proxy: { mode: 'inherit', url: '' },
  name: 'x',

  provider_url: '',
  site_user_id: '42',
  credentials: '',

  models: [],
})

/** v10 record written before the site access fields existed. */
const legacyDraft = () => {
  const value: Record<string, unknown> = { ...draft() }
  delete value.site_user_id
  return value
}

function attempt(value: unknown) {
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
  const result = controller.captureForUnauthorized()
  return { result, consumed: controller.consume(), stored: storage.size > 0 }
}

function roundTrip(value: unknown) {
  const { result, consumed } = attempt(value)
  assert.equal(result, 'stored')
  return consumed
}

function rejects(value: unknown) {
  const { result, consumed, stored } = attempt(value)
  assert.equal(result, 'no-active-draft')
  assert.equal(consumed, null)
  assert.equal(stored, false)
}

test('recovery preserves channel drafts without staging fields', () => {
  for (const channel_id of ['openai', 'anthropic', 'azure', 'vertex', 'bedrock', 'cliproxyapi']) {
    const value = { ...draft(), channel_id, credentials: '{"api_key":"test-only"}' }
    assert.deepEqual(roundTrip(value), value)
  }
  const existing = { mode: 'existing', group_id: 9, credentials: 'test-only' }
  assert.deepEqual(roundTrip(existing), existing)
  const channelNotSelected = { ...draft(), channel_id: '', credentials: 'key-before-login' }
  assert.deepEqual(roundTrip(channelNotSelected), channelNotSelected)
})

test('recovery keeps drafts written before the site user id field existed', () => {
  for (const channel_id of ['openai', 'anthropic', 'azure', 'vertex', 'bedrock', 'cliproxyapi']) {
    const value = { ...legacyDraft(), channel_id, credentials: '{"api_key":"test-only"}' }
    assert.deepEqual(roundTrip(value), { ...value, site_user_id: '' })
  }
  const legacyNotSelected = { ...legacyDraft(), channel_id: '', credentials: 'key-before-login' }
  assert.deepEqual(roundTrip(legacyNotSelected), { ...legacyNotSelected, site_user_id: '' })
})

test('recovery rejects drafts that carry site access secrets', () => {
  rejects({ ...draft(), site_access_token: 'not-persisted' })
  rejects({ ...draft(), access_token: 'not-persisted' })
  rejects({ ...draft(), encrypted_access_token: 'not-persisted' })
  rejects({ ...legacyDraft(), site_access_token: 'not-persisted' })
})

test('recovery rejects retired connection kinds and staging fields', () => {
  rejects({ ...draft(), connection_type: 'api_key' })
  rejects({ ...draft(), staged_credential: null })
  rejects({ ...draft(), staged_credentials: [] })
  rejects({ mode: 'existing', group_id: 9, credentials: '', staged_credential: null })
})
