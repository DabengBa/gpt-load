import assert from 'node:assert/strict'
import test from 'node:test'

import {
  resolveSchedulePriceID,
  type SchedulePricePage,
  type SchedulePriceTarget,
} from '../src/shared/domain/monitor/schedule-price.ts'

// Focused proof for the schedule → price identity mapping: pages of fuzzy
// client-model matches must be narrowed to an exact client model, then the
// price is picked by channel_id + upstream model_id (never invented).

function upstream(modelID: string, channelID: string, priceID: number) {
  return { model_id: modelID, price: { id: priceID, channel_id: channelID } }
}

function item(clientModel: string, upstreams: ReturnType<typeof upstream>[]) {
  return { client_model: clientModel, upstream_models: upstreams }
}

function page(pageno: number, total: number, items: SchedulePricePage['items']): SchedulePricePage {
  return { items, pagination: { page: pageno, total_pages: total } }
}

function recorder(pages: Record<number, SchedulePricePage>) {
  const requested: number[] = []
  const fetchPage = async (pageno: number): Promise<SchedulePricePage> => {
    requested.push(pageno)
    const found = pages[pageno]
    if (found === undefined) throw new Error(`unexpected page ${pageno}`)
    return found
  }
  return { fetchPage, requested }
}

test('crosses pages to exact-match the 11th client model', async () => {
  const noise = Array.from({ length: 10 }, (_, index) =>
    item(`gpt-5-mini-${index}`, [upstream(`gpt-5-mini-${index}`, 'ch-1', 100 + index)]),
  )
  const { fetchPage, requested } = recorder({
    1: page(1, 2, noise),
    2: page(2, 2, [item('gpt-5', [upstream('gpt-5', 'ch-1', 4242)])]),
  })
  const target: SchedulePriceTarget = {
    clientModel: 'gpt-5',
    channelID: 'ch-1',
    upstreamModelID: 'gpt-5',
  }
  assert.equal(await resolveSchedulePriceID(target, fetchPage), 4242)
  assert.deepEqual(requested, [1, 2])
})

test('fuzzy same-name matches never override the exact client model', async () => {
  const target: SchedulePriceTarget = {
    clientModel: 'gpt-5',
    channelID: 'ch-2',
    upstreamModelID: 'gpt-5-alias',
  }
  // 'gpt-5' sorts after its fuzzy siblings; its upstream lives on ch-2 only.
  const { fetchPage } = recorder({
    1: page(1, 1, [
      item('gpt-5-mini', [upstream('gpt-5', 'ch-2', 11)]),
      item('gpt-5-turbo', [upstream('gpt-5', 'ch-2', 12)]),
      item('gpt-5', [upstream('gpt-5-alias', 'ch-2', 77), upstream('gpt-5-alias', 'ch-1', 78)]),
    ]),
  })
  // Exact client model on the page picks the ch-2 price, not ch-1's.
  assert.equal(await resolveSchedulePriceID(target, fetchPage), 77)
})

test('resolves prices on disabled route groups without status filtering', async () => {
  const { fetchPage, requested } = recorder({
    1: page(1, 1, [item('gpt-5', [upstream('gpt-5', 'ch-off', 909)])]),
  })
  const target: SchedulePriceTarget = {
    clientModel: 'gpt-5',
    channelID: 'ch-off',
    upstreamModelID: 'gpt-5',
  }
  assert.equal(await resolveSchedulePriceID(target, fetchPage), 909)
  assert.deepEqual(requested, [1])
})

test('returns undefined instead of inventing an identity', async () => {
  const target: SchedulePriceTarget = {
    clientModel: 'gpt-5',
    channelID: 'ch-x',
    upstreamModelID: 'gpt-5',
  }
  // Exact client model exists but the channel does not belong to it.
  const wrongChannel = recorder({
    1: page(1, 1, [item('gpt-5', [upstream('gpt-5', 'ch-1', 5)])]),
  })
  assert.equal(await resolveSchedulePriceID(target, wrongChannel.fetchPage), undefined)
  // No exact match at all (only fuzzy siblings) → undefined, not the sibling.
  const noExact = recorder({
    1: page(1, 2, [item('gpt-5-mini', [upstream('gpt-5', 'ch-x', 6)])]),
    2: page(2, 2, [item('gpt-5-turbo', [upstream('gpt-5', 'ch-x', 7)])]),
  })
  assert.equal(await resolveSchedulePriceID(target, noExact.fetchPage), undefined)
  assert.deepEqual(noExact.requested, [1, 2])
})
