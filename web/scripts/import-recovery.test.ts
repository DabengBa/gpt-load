import assert from 'node:assert/strict'
import test from 'node:test'
import { createImportRecoveryService } from '../src/shared/controllers/import-recovery.ts'
import { projectCredentialStage } from '../src/shared/control/resources/credential-stages.ts'

const draft = (staged_credential: unknown) => ({
  mode: 'new',
  channel_id: 'openai',
  connection_type: 'subscription',
  params: {},
  proxy: { mode: 'inherit', url: '' },
  name: 'x',

  provider_url: '',
  credentials: '',
  staged_credential,
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

test('recovery persists and restores nullable single stage', () => {
  assert.deepEqual(roundTrip(draft(null)), draft(null))
  const stage = { stage_id: 'stage_one', status: 'ready', account: {}, expires_at_ms: 100000 }
  assert.deepEqual(roundTrip(draft(stage)), draft(stage))
})

test('recovery rejects arrays, malformed stages and retired plural fields', () => {
  assert.equal(roundTrip(draft([])), null)
  assert.equal(roundTrip(draft({ stage_id: 'invalid' })), null)
  assert.equal(roundTrip({ ...draft(null), staged_credentials: [] }), null)
  const value = draft(null)
  const { staged_credential: omitted, ...missing } = value
  assert.equal(omitted, null)
  assert.equal(roundTrip(missing), null)
})

for (const authorization_method of ['browser_oauth', 'device_oauth'] as const) {
  test(`real ${authorization_method} projector output survives unauthorized recovery`, () => {
    const stage = projectCredentialStage({
      stage_id: 'stage_real',
      status: 'pending_authorization',
      authorization_method,
      authorization_url: 'https://auth.example.com/authorize',
      redirect_uri: 'http://localhost/callback',
      ...(authorization_method === 'device_oauth'
        ? { user_code: 'ABCD-EFGH', next_poll_at_ms: 5000 }
        : {}),
      account: { email_mask: 'a***@example.com', expires_at_ms: 90000, last_refresh_at_ms: 1000 },
      expires_at_ms: 100000,
    })
    assert.deepEqual(roundTrip(draft(stage)), draft(stage))
    const existing = { mode: 'existing', group_id: 9, credentials: '', staged_credential: stage }
    assert.deepEqual(roundTrip(existing), existing)
  })
}

test('recovery validates authorization fields and rejects secret additions', () => {
  const stage = { stage_id: 'stage_one', status: 'ready', account: {}, expires_at_ms: 100000 }
  for (const fields of [
    { authorization_method: 'invalid' },
    { user_code: '\nsecret' },
    { next_poll_at_ms: -1 },
    { authorization_url: 'javascript:alert(1)' },
    { access_token: 'not-a-real-secret' },
    { duplicate: 'yes' },
  ])
    assert.equal(roundTrip(draft({ ...stage, ...fields })), null)
})
