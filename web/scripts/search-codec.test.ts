import assert from 'node:assert/strict'
import test from 'node:test'

import {
  parseSharedRouteSearch,
  stringifySharedRouteSearch,
} from '../src/frontends/astryx/app/search-codec.ts'

// Codec parity with vue-router's parseQuery/stringifyQuery — the astryx
// frontend must read and write the exact same URL query shape classic does.

// deepStrictEqual distinguishes prototypes; the codec's null-prototype result
// spreads to a plain object for shape comparisons.
function plain(query: ReturnType<typeof parseSharedRouteSearch>): Record<string, unknown> {
  return { ...query }
}

test('basic pairs and leading ? are parsed', () => {
  assert.deepEqual(plain(parseSharedRouteSearch('?a=1&b=2')), { a: '1', b: '2' })
  assert.deepEqual(plain(parseSharedRouteSearch('a=1')), { a: '1' })
  assert.deepEqual(plain(parseSharedRouteSearch('')), {})
  assert.deepEqual(plain(parseSharedRouteSearch('?')), {})
})

test('bare keys map to null; explicit empty stays empty string', () => {
  assert.deepEqual(plain(parseSharedRouteSearch('?key')), { key: null })
  assert.deepEqual(plain(parseSharedRouteSearch('?key=')), { key: '' })
})

test('+ maps to space before the = split, on both sides', () => {
  assert.deepEqual(plain(parseSharedRouteSearch('?q=a+b')), { q: 'a b' })
  assert.deepEqual(plain(parseSharedRouteSearch('?a+b=c')), { 'a b': 'c' })
})

test('key and value decode independently; malformed side keeps raw', () => {
  assert.deepEqual(plain(parseSharedRouteSearch('?a=%E0%A4%A')), { a: '%E0%A4%A' })
  assert.deepEqual(plain(parseSharedRouteSearch('?%E0%A4%A=x')), { '%E0%A4%A': 'x' })
  assert.deepEqual(plain(parseSharedRouteSearch('?k=%25')), { k: '%' })
})

test('duplicate keys collect into arrays in order', () => {
  assert.deepEqual(plain(parseSharedRouteSearch('?x=1&x=2&x=3')), { x: ['1', '2', '3'] })
  assert.deepEqual(plain(parseSharedRouteSearch('?x&x=1')), { x: [null, '1'] })
})

test('only the first = splits key from value', () => {
  assert.deepEqual(plain(parseSharedRouteSearch('?a=b=c')), { a: 'b=c' })
})

test('__proto__ stays an own property and never mutates the prototype', () => {
  const parsed = parseSharedRouteSearch('?__proto__=x&safe=1')
  assert.equal(Object.getPrototypeOf(parsed), null)
  assert.equal(parsed.__proto__, 'x')
  assert.equal(({} as Record<string, unknown>).safe === undefined, true)
})

test('empty pairs are skipped', () => {
  assert.deepEqual(plain(parseSharedRouteSearch('?a=1&&b=2')), { a: '1', b: '2' })
})

test('stringify: null emits a bare key, undefined drops it', () => {
  assert.equal(
    stringifySharedRouteSearch({ a: null, b: undefined, c: 'x' }),
    '?a&c=x',
  )
})

test('stringify: arrays repeat the key; numbers serialize', () => {
  assert.equal(
    stringifySharedRouteSearch({ page: 2, tag: ['a', 'b'] }),
    '?page=2&tag=a&tag=b',
  )
})

test('stringify encodes both key and value', () => {
  assert.equal(stringifySharedRouteSearch({ 'a b': 'c/d' }), '?a%20b=c%2Fd')
})

test('parse → stringify round-trip preserves the query', () => {
  assert.equal(
    stringifySharedRouteSearch(parseSharedRouteSearch('?q=a+b&x&x=2')),
    '?q=a%20b&x&x=2',
  )
})
