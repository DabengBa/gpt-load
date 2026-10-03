import assert from 'node:assert/strict'
import test from 'node:test'

import { pagePath } from '../src/shared/routing/page-routes.ts'
import { sharedPageRouteNames } from '../src/shared/routing/route-names.ts'
import {
  decodedPathSegments,
  safeRedirectTarget,
  type SafeRedirectResolution,
} from '../src/shared/routing/safe-redirect.ts'

const home = pagePath(sharedPageRouteNames.home)
const blockedRouteNames: readonly string[] = [sharedPageRouteNames.login, 'not-found']

function resolution(overrides: Partial<SafeRedirectResolution> = {}): SafeRedirectResolution {
  return {
    matched: [{}],
    name: sharedPageRouteNames.logs,
    path: '/logs',
    fullPath: '/logs',
    meta: { requiresAuth: true },
    ...overrides,
  }
}

test('malformed percent-encoding is rejected before resolution and falls back home', () => {
  const resolvedInputs: string[] = []
  const target = safeRedirectTarget(
    '/%E0%A4%A',
    (raw) => {
      resolvedInputs.push(raw)
      return resolution({ path: raw, fullPath: raw })
    },
    blockedRouteNames,
  )
  assert.equal(target, home)
  assert.deepEqual(resolvedInputs, [])
})

test('a valid authenticated route resolves to its full path', () => {
  const target = safeRedirectTarget(
    '/logs',
    (raw) => resolution({ path: raw, fullPath: raw }),
    blockedRouteNames,
  )
  assert.equal(target, '/logs')
})

test('protocol-relative and encoded double-slash targets fall back home', () => {
  const resolve = (raw: string): SafeRedirectResolution => resolution({ path: raw, fullPath: raw })
  assert.equal(safeRedirectTarget('//evil.example', resolve, blockedRouteNames), home)
  assert.equal(safeRedirectTarget('/%2F%2Fevil.example', resolve, blockedRouteNames), home)
})

test('blocked and unauthenticated routes fall back home', () => {
  const resolve = (raw: string): SafeRedirectResolution => resolution({ path: raw, fullPath: raw })
  assert.equal(safeRedirectTarget('/login', resolve, blockedRouteNames), home)
  assert.equal(
    safeRedirectTarget(
      '/logs',
      (raw) => resolution({ path: raw, fullPath: raw, meta: { requiresAuth: false } }),
      blockedRouteNames,
    ),
    home,
  )
  assert.equal(
    safeRedirectTarget(
      '/logs',
      (raw) => resolution({ path: raw, fullPath: raw, matched: [] }),
      blockedRouteNames,
    ),
    home,
  )
})

test('non-string and backslash targets fall back home without resolution', () => {
  const resolvedInputs: string[] = []
  const resolve = (raw: string): SafeRedirectResolution => {
    resolvedInputs.push(raw)
    return resolution({ path: raw, fullPath: raw })
  }
  assert.equal(safeRedirectTarget('https://evil.example', resolve, blockedRouteNames), home)
  assert.equal(safeRedirectTarget('/\\evil.example', resolve, blockedRouteNames), home)
  assert.deepEqual(resolvedInputs, [])
})

test('undecodable path segments become an explicit invalid-path marker', () => {
  assert.deepEqual(decodedPathSegments('/%E0%A4%A'), ['invalid-path'])
  assert.deepEqual(decodedPathSegments('/logs/42'), ['logs', '42'])
})
