import assert from 'node:assert/strict'
import test from 'node:test'

import {
  channelOperations,
  sameStringMembers,
} from '../src/frontends/classic/app/resources/channel-contract.ts'

test('channel protocol membership does not depend on route declaration order', () => {
  assert.equal(
    sameStringMembers(
      ['openai-completions', 'openai-responses', 'openai-images', 'openai-embeddings'],
      ['openai-completions', 'openai-embeddings', 'openai-images', 'openai-responses'],
    ),
    true,
  )
})

test('channel operation contract includes Codex web search routes', () => {
  assert.equal((channelOperations as readonly string[]).includes('web_search'), true)
})
