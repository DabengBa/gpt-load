import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import {
  astryxRoutePaths,
  toTanStackPath,
} from '../src/frontends/astryx/app/route-adapter.ts'
import { pageRouteMeta } from '../src/shared/routing/route-meta.ts'

const manifestPath = fileURLToPath(
  new URL('../../internal/webui/page_routes.json', import.meta.url),
)
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
  routes: Array<{ name: string; path: string; astryx?: boolean }>
}

test('manifest adapter covers every route name exactly once', () => {
  const descriptors = astryxRoutePaths(manifest.routes)
  assert.equal(descriptors.length, manifest.routes.length)
  assert.deepEqual(
    descriptors.map((descriptor) => descriptor.name).sort(),
    manifest.routes.map((route) => route.name).sort(),
  )
})

test('manifest adapter converts vue :param segments to tanstack $param', () => {
  assert.equal(toTanStackPath('/groups/:id'), '/groups/$id')
  assert.equal(toTanStackPath('/'), '/')
  assert.equal(toTanStackPath('/settings'), '/settings')
  const groupDetail = astryxRoutePaths(manifest.routes).find(
    (entry) => entry.name === 'group-detail',
  )
  assert.equal(groupDetail?.path, '/groups/$id')
})

test('every manifest route has shared page meta', () => {
  for (const route of manifest.routes) {
    assert.notEqual(
      pageRouteMeta[route.name],
      undefined,
      `missing pageRouteMeta for "${route.name}"`,
    )
  }
})

test('auth meta mirrors the classic route contract', () => {
  assert.equal(pageRouteMeta.login.requiresAuth, undefined)
  assert.equal(pageRouteMeta.settings.requiresAuth, true)
  assert.equal(pageRouteMeta.settings.adminOnly, true)
  assert.equal(pageRouteMeta.monitor.requiresAuth, true)
  assert.equal(pageRouteMeta.monitor.adminOnly, undefined)
})
