import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import { astryxRoutePaths, toTanStackPath } from '../src/frontends/astryx/app/route-adapter.ts'
import { pageRouteMeta } from '../src/shared/routing/route-meta.ts'

const manifestPath = fileURLToPath(
  new URL('../../internal/webui/page_routes.json', import.meta.url),
)
const shellPath = fileURLToPath(
  new URL('../src/frontends/astryx/app/shell/Shells.tsx', import.meta.url),
)
const groupModelsTabPath = fileURLToPath(
  new URL('../src/frontends/astryx/features/groups/models/GroupModelsTab.tsx', import.meta.url),
)
const routerPath = fileURLToPath(new URL('../src/frontends/astryx/app/router.tsx', import.meta.url))
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

// U001: /schedule is the single model entry. The retired /models page must
// disappear from the manifest outright — no compatibility redirect.
test('models page route is retired from the manifest and the route meta', () => {
  const modelRoutes = manifest.routes.filter(
    (route) => route.name === 'models' || route.path === '/models',
  )
  assert.deepEqual(modelRoutes, [], 'manifest still declares a models page route')
  assert.equal(pageRouteMeta.models, undefined, 'route meta still declares a models route')
})

// schedule stays auth-gated but is no longer adminOnly, so an access_key
// principal reaches it too, and it loads the namespaces the price entry needs.
test('schedule opens to both principals and loads the model price namespaces', () => {
  const schedule = pageRouteMeta.schedule
  assert.equal(schedule.requiresAuth, true)
  assert.equal(schedule.adminOnly, undefined)
  assert.equal(schedule.primaryNav, 'schedule')
  assert.ok(
    schedule.messageNamespaces?.includes('models') === true,
    'schedule meta must load the models namespace',
  )
  assert.ok(
    schedule.messageNamespaces?.includes('model-prices') === true,
    'schedule meta must load the model-prices namespace',
  )
})

// Source-level routing contract: the shell nav exposes schedule for both
// principals and no longer links the retired models page. Runtime nav proof
// belongs to the Playwright shell journey (U004); this is the fast-lane guard.
test('shell nav links schedule and drops the models link', () => {
  const shell = readFileSync(shellPath, 'utf8')
  assert.ok(shell.includes("pagePath('schedule')"), 'shell nav has no schedule link')
  assert.ok(!shell.includes("pagePath('models')"), 'shell nav still links the models page')
})

// Price deep link moves to schedule with the selected_price_id the schedule
// query state consumes.
test('price entry jumps to schedule with selected_price_id', () => {
  const groupModelsTab = readFileSync(groupModelsTabPath, 'utf8')
  assert.ok(
    groupModelsTab.includes("pagePath('schedule')"),
    'price entry does not target the schedule page',
  )
  assert.ok(groupModelsTab.includes('selected_price_id'), 'price entry dropped selected_price_id')
  assert.ok(!groupModelsTab.includes("pagePath('models')"), 'price entry still targets /models')
})

// The router must not build a models route any more; otherwise pagePath
// ('models') would throw and ModelsView would stay reachable.
test('router no longer wires the models route', () => {
  const router = readFileSync(routerPath, 'utf8')
  assert.ok(!router.includes('models-route'), 'router still imports the models route codec')
  assert.ok(!router.includes('ModelsView'), 'router still mounts ModelsView')
  assert.ok(
    !router.includes('sharedPageRouteNames.models'),
    'router still registers the models route name',
  )
})
