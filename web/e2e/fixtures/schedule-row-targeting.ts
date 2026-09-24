import type { Page } from '@playwright/test'

// Fixtures for U001: navigating from the group models editor to the schedule
// page with source-group row targeting (schedule_group / schedule_row).

export const alphaGroupId = 1

function envelope(data: unknown) {
  return { code: 0, message: 'OK', data }
}

function response(data: unknown, status = 200) {
  return {
    status,
    contentType: 'application/json',
    body: JSON.stringify(envelope(data)),
  }
}

export function scheduleEntry(
  entryID: string,
  modelID: string,
  weight: number,
  priority = 1,
  alias = '',
  share = 0.5,
  effectiveShare = share,
) {
  return {
    entry_id: entryID,
    model_id: modelID,
    alias,
    weight,
    priority,
    fallback: priority > 1,
    circuit_breaker: {
      configured: { blacklist_threshold: null, cooldown_seconds: null },
      effective: { blacklist_threshold: 3, cooldown_seconds: 60 },
      sources: { blacklist_threshold: 'default', cooldown_seconds: 'default' },
    },
    reasoning: {
      configured: null,
      effective: null,
      source: 'provider_default',
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
    configured_share: share,
    effective_share: effectiveShare,
    credentials: [],
  }
}

function scheduleGroup(
  groupID: number,
  groupName: string,
  entries: ReturnType<typeof scheduleEntry>[],
) {
  return {
    group_id: groupID,
    group_name: groupName,
    channel_id: `channel-${groupID}`,
    enabled: true,
    reasoning_effort_default: null,
    reasoning_entries: entries.map(({ entry_id, model_id, reasoning }) => ({
      entry_id,
      model_id,
      reasoning,
    })),
    request_count: 10,
    success_rate: 1,
    entries,
  }
}

// 'worker' carries enough rows to overflow the default viewport so the scroll
// behaviour of row targeting is observable.
function workerDetail(externalModel: string) {
  return {
    observed_at_ms: 1_700_000_000_000,
    snapshot_revision: 11,
    external_model: externalModel,
    protocol: 'openai-completions',
    operation: 'chat_completion',
    route_requirement: 'any',
    access_key: { id: 7, name: 'e2e access key', status: 'active' },
    routable: true,
    reason_code: null,
    groups: [
      scheduleGroup(1, 'alpha group', [
        scheduleEntry('entry-1', 'model-a', 10, 1, '', 0.1),
        scheduleEntry('entry-3', 'model-c', 10, 1, '', 0.1),
      ]),
      scheduleGroup(2, 'beta group', [
        scheduleEntry('entry-2', 'model-b', 10, 1, 'worker', 0.1),
        scheduleEntry('entry-6', 'model-f', 10, 1, '', 0.1),
        scheduleEntry('entry-7', 'model-g', 10, 1, '', 0.1),
        scheduleEntry('entry-8', 'model-h', 10, 1, '', 0.1),
        scheduleEntry('entry-9', 'model-i', 10, 1, '', 0.1),
        scheduleEntry('entry-10', 'model-j', 10, 1, '', 0.1),
        scheduleEntry('entry-11', 'model-k', 10, 1, '', 0.1),
        scheduleEntry('entry-12', 'model-l', 10, 1, '', 0.1),
      ]),
    ],
  }
}

function workerBDetail(externalModel: string) {
  return {
    observed_at_ms: 1_700_000_000_000,
    snapshot_revision: 21,
    external_model: externalModel,
    protocol: 'openai-completions',
    operation: 'chat_completion',
    route_requirement: 'any',
    access_key: { id: 7, name: 'e2e access key', status: 'active' },
    routable: true,
    reason_code: null,
    groups: [
      scheduleGroup(1, 'alpha group', [
        scheduleEntry('entry-2', 'model-b', 50, 2, 'worker-b', 0.5, 0.25),
        scheduleEntry('entry-4', 'model-d', 50, 1, '', 0.5, 0.25),
      ]),
      scheduleGroup(2, 'beta group', [
        scheduleEntry('entry-5', 'model-e', 50, 1, '', 0.5, 0.25),
        scheduleEntry('entry-6', 'model-f', 50, 2, '', 0.5, 0.25),
      ]),
    ],
  }
}

function groupSummary() {
  return {
    id: alphaGroupId,
    name: 'alpha group',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    price_multiplier: '1',
    service_status: 'available',
    service_status_reason: null,
    credential_count: 1,
    model_count: 2,
  }
}

function groupSettings() {
  return {
    name: 'alpha group',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    price_multiplier: '1',
    enabled: true,
    overrides: {},
    effective: {
      first_byte_timeout: 30,
      request_timeout: 600,
      stream_idle_timeout: 30,
      blacklist_threshold: 3,
      header_rules: { set: {}, remove: [] },
      affinity_enabled: false,
      responses_websocket_enabled: false,
      responses_reasoning_status_filter_enabled: false,
    },
    proxy: {
      configured_mode: 'inherit',
      effective_mode: 'direct',
      effective_source: 'default',
      has_auth: false,
    },
  }
}

function groupModels() {
  return {
    items: [
      {
        id: 'model-a',
        alias: '',
        alias_enabled: false,
        client_model: 'model-a',
        test_alias: 'aaaaaa',
        weight: 50,
        priority: 1,
        pricing_status: 'configured',
        entry_id: 'entry-1',
      },
      {
        id: 'model-b',
        alias: 'worker-b',
        alias_enabled: true,
        client_model: 'worker-b',
        test_alias: 'bbbbbb',
        weight: 50,
        priority: 2,
        pricing_status: 'configured',
        entry_id: 'entry-2',
      },
    ],
    total: 2,
    pending: 0,
  }
}

export function detailRowLocatorKey(groupID: number, entryID: string): string {
  return `${groupID}:${entryID}`
}

export async function installScheduleRowTargetingRoutes(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.localStorage.setItem('gpt-load.auth-key', 'e2e-admin-key')
  })

  await page.route(
    (url) => url.pathname === '/api' || url.pathname.startsWith('/api/'),
    async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      const path = url.pathname

      if (path === '/api/auth/session') {
        await route.fulfill(response({ authenticated: true, principal_type: 'admin' }))
        return
      }
      if (path === '/api/channels') {
        await route.fulfill(response({ items: [], total: 0 }))
        return
      }
      if (path === `/api/groups/${alphaGroupId}`) {
        await route.fulfill(response(groupSummary()))
        return
      }
      if (path === `/api/groups/${alphaGroupId}/settings`) {
        await route.fulfill(response(groupSettings()))
        return
      }
      if (path === `/api/groups/${alphaGroupId}/credentials`) {
        await route.fulfill(
          response({
            observed_at_ms: 1_700_000_000_000,
            stats_window_seconds: 86_400,
            summary: { total: 0, available: 0, cooldown: 0, blacklisted: 0, disabled: 0 },
            items: [],
            pagination: { page: 1, page_size: 20, total_items: 0, total_pages: 0 },
          }),
        )
        return
      }
      if (path === `/api/groups/${alphaGroupId}/models`) {
        await route.fulfill(response(groupModels()))
        return
      }
      if (path === '/api/groups/options') {
        await route.fulfill(response([]))
        return
      }
      if (path === '/api/access-keys/options') {
        await route.fulfill(response([{ id: 7, name: 'e2e access key', status: 'active' }]))
        return
      }
      if (path === '/api/model-route/schedule' && request.method() === 'GET') {
        await route.fulfill(
          response({
            items: [
              {
                external_model: 'worker',
                protocol: 'openai-completions',
                operation: 'chat_completion',
                candidate_count: 10,
                group_count: 2,
                has_fallback: false,
                cooled_candidates: 0,
                blacklisted_candidates: 0,
              },
              {
                external_model: 'worker-b',
                protocol: 'openai-completions',
                operation: 'chat_completion',
                candidate_count: 4,
                group_count: 2,
                has_fallback: false,
                cooled_candidates: 0,
                blacklisted_candidates: 0,
              },
            ],
          }),
        )
        return
      }
      if (path === '/api/model-route/schedule/detail') {
        const externalModel = url.searchParams.get('external_model') ?? 'worker'
        await route.fulfill(
          response(
            externalModel === 'worker-b'
              ? workerBDetail(externalModel)
              : workerDetail(externalModel),
          ),
        )
        return
      }

      await route.fulfill(response({}, 404))
    },
  )
}
