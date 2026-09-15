import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const WEB_ROOT = fileURLToPath(new URL('..', import.meta.url))

const baseSettings = {
  name: 'Reasoning group',
  price_multiplier: '1',
  channel_id: 'openai',
  connection_type: 'api_key',
  params: {},
  provider_url: null,
  enabled: true,
  overrides: {
    parameter_overrides: [{ match: { model: 'public-*' }, set: { temperature: 0.4 } }],
    reasoning_effort_overrides: {
      'upstream-b': 'high',
      'upstream-a': 'low',
    },
  },
  effective: {
    first_byte_timeout: 120,
    request_timeout: 600,
    stream_idle_timeout: 300,
    blacklist_threshold: 3,
    header_rules: { set: {}, remove: [] },
    affinity_enabled: true,
    responses_websocket_enabled: true,
  },
  proxy: {
    configured_mode: 'inherit',
    effective_mode: 'direct',
    effective_source: 'default',
    has_auth: false,
  },
}

async function main() {
  const server = await createServer({
    root: WEB_ROOT,
    server: { middlewareMode: true },
    appType: 'custom',
    logLevel: 'error',
  })

  try {
    const groups = await server.ssrLoadModule('/src/app/resources/groups.ts')
    const patching = await server.ssrLoadModule(
      '/src/features/groups/settings/group-settings-patch.ts',
    )
    const projected = groups.projectGroupSettings(baseSettings)
    assert.deepEqual(projected.overrides.reasoning_effort_overrides, {
      'upstream-a': 'low',
      'upstream-b': 'high',
    })
    assert.throws(() =>
      groups.projectGroupSettings({
        ...baseSettings,
        overrides: { reasoning_effort_overrides: { 'upstream-a': 'unsupported' } },
      }),
    )

    const changed = patching.createGroupSettingsDraft(projected)
    changed.overrides.reasoning_effort_overrides = {
      ...changed.overrides.reasoning_effort_overrides,
      'upstream-a': 'xhigh',
    }
    const changedPatch = patching.buildGroupSettingsPatch(projected, changed)
    assert.deepEqual(changedPatch.overrides, {
      parameter_overrides: baseSettings.overrides.parameter_overrides,
      reasoning_effort_overrides: {
        'upstream-a': 'xhigh',
        'upstream-b': 'high',
      },
    })

    const removed = patching.createGroupSettingsDraft(projected)
    delete removed.overrides.reasoning_effort_overrides
    const removedPatch = patching.buildGroupSettingsPatch(projected, removed)
    assert.deepEqual(removedPatch.overrides, {
      parameter_overrides: baseSettings.overrides.parameter_overrides,
    })
    console.log('PASS  group reasoning effort projection and patch preservation')
  } finally {
    await server.close()
  }
}

try {
  await main()
} catch (error) {
  console.error(`FAIL  group reasoning effort projection and patch preservation: ${error.message}`)
  process.exitCode = 1
}
