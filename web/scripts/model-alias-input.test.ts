import assert from 'node:assert/strict'
import test from 'node:test'

import {
  clientModel,
  modelDraftValidity,
  normalizeModel,
  sameModels,
} from '../src/shared/domain/models/model-draft.ts'

test('alias text alone determines routing, normalized writes and dirty state', () => {
  const empty = { id: 'upstream', alias: '', alias_enabled: false }
  const typed = { ...empty, alias: ' public ', alias_enabled: false }
  const spaces = { ...empty, alias: '   ', alias_enabled: true }
  assert.deepEqual(normalizeModel(typed), {
    id: 'upstream',
    alias: 'public',
    alias_enabled: true,
    weight: null,
    priority: null,
  })
  assert.equal(clientModel(typed), 'public')
  assert.deepEqual(normalizeModel(spaces), {
    id: 'upstream',
    alias: '',
    alias_enabled: false,
    weight: null,
    priority: null,
  })
  assert.equal(clientModel(spaces), 'upstream')
  assert.equal(modelDraftValidity([spaces]).invalidIndexes.size, 0)
  assert.equal(sameModels([empty], [spaces]), true)
  assert.equal(sameModels([empty], [typed]), false)
  assert.equal(sameModels([typed], [{ ...typed, alias: '' }]), false)
})
