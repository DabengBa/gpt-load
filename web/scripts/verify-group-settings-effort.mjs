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
  },
  effective: {
    first_byte_timeout: 120,
    request_timeout: 600,
    stream_idle_timeout: 300,
    blacklist_threshold: 3,
    header_rules: { set: {}, remove: [] },
    affinity_enabled: true,
    responses_websocket_enabled: true,
    responses_reasoning_status_filter_enabled: true,
  },
  proxy: {
    configured_mode: 'inherit',
    effective_mode: 'direct',
    effective_source: 'default',
    has_auth: false,
  },
}

const baseSchedule = {
  observed_at_ms: 1_800_000_000_000,
  snapshot_revision: 7,
  external_model: 'reasoning-model',
  protocol: 'openai-completions',
  operation: 'chat_completion',
  route_requirement: 'any',
  access_key: { id: 1, name: 'default', status: 'active' },
  routable: true,
  reason_code: null,
  groups: [
    {
      group_id: 4,
      group_name: 'OpenAI primary',
      channel_id: 'openai',
      enabled: true,
      reasoning_effort_default: 'medium',
      request_count: 10,
      success_rate: 0.9,
      entries: [
        {
          entry_id: 'entry-1',
          model_id: 'o3',
          alias: 'reasoning-model',
          weight: 100,
          priority: 1,
          fallback: false,
          circuit_breaker: {
            configured: { blacklist_threshold: null, cooldown_seconds: null },
            effective: { blacklist_threshold: null, cooldown_seconds: 3600 },
            sources: { blacklist_threshold: 'default', cooldown_seconds: 'default' },
          },
          reasoning: {
            configured: 'high',
            effective: 'high',
            source: 'entry',
          },
          runtime: {
            state: 'available',
            cooldown_until_ms: null,
            blacklist_release_at_ms: null,
            failure_count: 0,
            failure_version: 0,
          },
          included: true,
          routable: true,
          reason_code: null,
          configured_share: 1,
          effective_share: 1,
          credentials: [],
        },
      ],
    },
  ],
}

for (const group of baseSchedule.groups) {
  group.reasoning_entries = group.entries.map(({ entry_id, model_id, reasoning }) => ({
    entry_id,
    model_id,
    reasoning,
  }))
}

async function main() {
  const server = await createServer({
    root: WEB_ROOT,
    server: { middlewareMode: true },
    appType: 'custom',
    logLevel: 'error',
  })

  try {
    const groups = await server.ssrLoadModule('/src/frontends/classic/app/resources/groups.ts')
    const patching = await server.ssrLoadModule(
      '/src/frontends/classic/features/groups/settings/group-settings-patch.ts',
    )
    const schedule = await server.ssrLoadModule(
      '/src/frontends/classic/app/resources/model-route-schedule.ts',
    )

    const projected = groups.projectGroupSettings(baseSettings)
    const changed = patching.createGroupSettingsDraft(projected)
    changed.channel_id = 'anthropic'
    assert.deepEqual(patching.buildGroupSettingsPatch(projected, changed), {
      channel_id: 'anthropic',
    })
    assert.throws(() =>
      groups.projectGroupSettings({
        ...baseSettings,
        overrides: { reasoning_effort_overrides: { 'upstream-a': 'high' } },
      }),
    )

    const detail = schedule.projectModelRouteScheduleDetail(baseSchedule)
    assert.deepEqual(detail.groups[0].reasoning_entries, baseSchedule.groups[0].reasoning_entries)
    const missingGroupModels = structuredClone(baseSchedule)
    delete missingGroupModels.groups[0].reasoning_entries
    assert.throws(() => schedule.projectModelRouteScheduleDetail(missingGroupModels))
    assert.equal(detail.groups[0].reasoning_effort_default, 'medium')
    assert.deepEqual(
      detail.groups[0].entries[0].reasoning,
      baseSchedule.groups[0].entries[0].reasoning,
    )
    assert.throws(() =>
      schedule.projectModelRouteScheduleDetail({
        ...baseSchedule,
        groups: [
          {
            ...baseSchedule.groups[0],
            entries: [
              {
                ...baseSchedule.groups[0].entries[0],
                reasoning: {
                  ...baseSchedule.groups[0].entries[0].reasoning,
                  source: 'normalized',
                },
              },
            ],
          },
        ],
      }),
    )
    for (const effort of schedule.reasoningEffortValues) {
      const candidate = structuredClone(baseSchedule)
      candidate.groups[0].reasoning_effort_default = effort
      candidate.groups[0].entries[0].reasoning = {
        configured: effort,
        effective: effort,
        source: 'entry',
      }
      candidate.groups[0].reasoning_entries[0].reasoning = candidate.groups[0].entries[0].reasoning
      const projected = schedule.projectModelRouteScheduleDetail(candidate)
      assert.equal(projected.groups[0].entries[0].reasoning.effective, effort)
      assert.equal(projected.groups[0].reasoning_entries[0].reasoning.effective, effort)
    }
    const legacy = structuredClone(baseSchedule)
    legacy.groups[0].entries[0].reasoning.obsolete = true
    assert.throws(() => schedule.projectModelRouteScheduleDetail(legacy))
    const invalidEffort = structuredClone(baseSchedule)
    invalidEffort.groups[0].entries[0].reasoning.configured = 'unknown'
    assert.throws(() => schedule.projectModelRouteScheduleDetail(invalidEffort))

    console.log('PASS  retired group effort and dispatch reasoning projection contracts')
  } finally {
    await server.close()
  }
}

try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
