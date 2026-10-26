import assert from 'node:assert/strict'
import test from 'node:test'

import {
  consumeScriptcatImportPayload,
  scriptcatImportPayloadStorageKey,
  scriptcatImportPayloadTtlMs,
} from '../src/shared/domain/import/scriptcat-payload.ts'

function memoryStorage(): Storage {
  const values = new Map<string, string>()
  return {
    get length() {
      return values.size
    },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    removeItem: (key) => values.delete(key),
    setItem: (key, value) => values.set(key, value),
  }
}

test('stores and consumes group details and credentials once', () => {
  const storage = memoryStorage()
  storage.setItem(
    scriptcatImportPayloadStorageKey,
    JSON.stringify({
      version: 2,
      createdAt: Date.now(),
      apiKey: 'sk-test',
      baseURL: 'https://api.example.test/v1',
      groupName: 'Example provider',
      providerUrl: 'https://provider.example.test',
    }),
  )
  assert.deepEqual(consumeScriptcatImportPayload(storage), {
    apiKey: 'sk-test',
    baseURL: 'https://api.example.test/v1',
    groupName: 'Example provider',
    providerUrl: 'https://provider.example.test',
  })
  assert.equal(storage.getItem(scriptcatImportPayloadStorageKey), null)
  assert.equal(consumeScriptcatImportPayload(storage), null)
})

test('rejects malformed, expired, and oversized payloads and removes them', () => {
  const storage = memoryStorage()
  const validPayload = (overrides: Record<string, unknown> = {}) => ({
    version: 2,
    createdAt: Date.now(),
    apiKey: 'k',
    baseURL: 'https://x.test',
    groupName: 'Example provider',
    providerUrl: 'https://provider.example.test',
    ...overrides,
  })
  const values = [
    '{',
    JSON.stringify(validPayload({ createdAt: Date.now() - scriptcatImportPayloadTtlMs - 1 })),
    JSON.stringify(validPayload({ apiKey: '' })),
    JSON.stringify(validPayload({ groupName: '  ' })),
    JSON.stringify(validPayload({ providerUrl: 'javascript:alert(1)' })),
    JSON.stringify(validPayload({ apiKey: 'k'.repeat(20_000) })),
    JSON.stringify(validPayload({ extra: true })),
  ]

  for (const value of values) {
    storage.setItem(scriptcatImportPayloadStorageKey, value)
    assert.equal(consumeScriptcatImportPayload(storage), null)
    assert.equal(storage.getItem(scriptcatImportPayloadStorageKey), null)
  }
})
