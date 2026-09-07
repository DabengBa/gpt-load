<script setup lang="ts">
import { ListFilter, RefreshCw } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import { lazySurface } from '@/app/async-surface'
import { monitorLocation } from '@/app/route-locations'
import { usageRanges } from '@/app/resources/usage'
import type { AccessProtocol } from '@/api/control/types'
import type { RouteInspectOperation } from '@/app/resources/route-inspection'
import LedgerSheet from '@/components/layout/LedgerSheet.vue'
import PageFrame from '@/components/layout/PageFrame.vue'
import AppButton from '@/components/ui/AppButton.vue'
import AppSelect from '@/components/ui/AppSelect.vue'
import AppTabs, { type AppTabItem } from '@/components/ui/AppTabs.vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import { isTimeRange } from '@/lib/time'
import { useAuthSession } from '@/features/auth/auth-session'

import HealthTab from './HealthTab.vue'
import {
  normalizeAccessKeyMonitorQuery,
  normalizeMonitorQuery,
  normalizeMonitorTab,
  parseScheduleMonitorState,
  parseUsageMonitorState,
  sameMonitorQuery,
  scheduleMonitorQuery,
  scopeAccessKeyUsageFilters,
} from './monitor-route'
import { usageMonitorQuery } from './monitor-route'
import type { SchedulePanelLabels } from './SchedulePanel.vue'
import { parseAppliedUsageFilters } from './usage-filters'

const InspectorTab = lazySurface(() => import('./InspectorTab.vue'))
const LogsTab = lazySurface(() => import('./LogsTab.vue'))
const UsageTab = lazySurface(() => import('./UsageTab.vue'))
const SchedulePanel = lazySurface(() => import('./SchedulePanel.vue'))

const route = useRoute()
const session = useAuthSession()
const router = useRouter()
const { locale, t } = useI18n()
const healthTab = ref<InstanceType<typeof HealthTab> | null>(null)
const usageTab = ref<{ openFilters: () => void; refresh: () => Promise<void> } | null>(null)
const healthRefreshPending = ref(false)
const usageRefreshPending = ref(false)
const isAccessKey = computed(() => session.state.principalType === 'access_key')
const isAdmin = computed(() => session.state.principalType === 'admin')
const canonicalQuery = computed(() => {
  if (isAdmin.value) return normalizeMonitorQuery(route.query)
  if (isAccessKey.value || normalizeMonitorTab(route.query.tab) === 'schedule') {
    return normalizeAccessKeyMonitorQuery(route.query)
  }
  return normalizeMonitorQuery(route.query)
})
const activeTab = computed(() => normalizeMonitorTab(canonicalQuery.value.tab))
const scheduleState = computed(() => parseScheduleMonitorState(route.query))
const scheduleAccessKeyID = computed(() => {
  const raw = scheduleState.value.accessKeyID
  if (raw === undefined) return undefined
  const value = Number(raw)
  return Number.isSafeInteger(value) && value > 0 ? value : undefined
})
const isCanonicalQuery = computed(() => sameMonitorQuery(route.query, canonicalQuery.value))
const items = computed<AppTabItem[]>(() => {
  const shared = [
    { value: 'usage', label: t('monitor.tabs.usage') },
    { value: 'logs', label: t('monitor.tabs.logs') },
  ]
  return isAdmin.value
    ? [
        { value: 'health', label: t('monitor.tabs.health') },
        ...shared,
        { value: 'inspector', label: t('monitor.tabs.inspector') },
        { value: 'schedule', label: t('monitor.tabs.schedule') },
      ]
    : shared
})
const usageFilters = computed(() => {
  const filters = parseAppliedUsageFilters(route.query)
  return isAccessKey.value ? scopeAccessKeyUsageFilters(filters) : filters
})
const usageRangeOptions = computed(() =>
  usageRanges.map((value) => ({
    value,
    label: t(`monitor.usage.filters.ranges.${value}`),
  })),
)
const usageFilterCount = computed(
  () =>
    Number(!isAccessKey.value && usageFilters.value.group_id !== undefined) +
    Number(!isAccessKey.value && usageFilters.value.channel_id !== undefined) +
    Number(!isAccessKey.value && usageFilters.value.credential_id !== undefined) +
    Number(usageFilters.value.upstream_model !== undefined),
)

watch(
  () => route.query,
  (query) => {
    const normalized = canonicalQuery.value
    if (!sameMonitorQuery(query, normalized)) {
      void router.replace(monitorLocation(normalized))
    }
  },
  { immediate: true },
)

function selectTab(value: string): void {
  const tab = normalizeMonitorTab(value)
  if (tab === activeTab.value) return
  if (tab === 'schedule') {
    if (!isAdmin.value) return
    void router.push(monitorLocation(scheduleMonitorQuery(scheduleState.value)))
    return
  }
  if (isAccessKey.value && tab !== 'usage' && tab !== 'logs') return
  void router.push(monitorLocation({ tab }))
}

async function refreshHealth(): Promise<void> {
  if (!healthTab.value || healthRefreshPending.value) return
  healthRefreshPending.value = true
  try {
    await healthTab.value.refresh()
  } finally {
    healthRefreshPending.value = false
  }
}

async function refreshUsage(): Promise<void> {
  if (!usageTab.value || usageRefreshPending.value) return
  usageRefreshPending.value = true
  try {
    await usageTab.value.refresh()
  } finally {
    usageRefreshPending.value = false
  }
}

function selectUsageRange(value: string): void {
  if (!isTimeRange(value)) return
  const state = parseUsageMonitorState(route.query)
  void router.push(
    monitorLocation(
      usageMonitorQuery(
        { ...usageFilters.value, range: value },
        {
          filtersOpen: false,
          seriesExpanded: false,
          metric: state.metric,
        },
      ),
    ),
  )
}
function updateScheduleContext(next: Partial<ReturnType<typeof parseScheduleMonitorState>>): void {
  void router.replace(monitorLocation(scheduleMonitorQuery({ ...scheduleState.value, ...next })))
}

function commitScheduleContext(context: {
  externalModel: string
  protocol: AccessProtocol
  operation?: RouteInspectOperation
  accessKeyId?: number
}): void {
  const next = {
    externalModel: context.externalModel,
    protocol: context.protocol,
    operation: context.operation,
    accessKeyID: context.accessKeyId === undefined ? undefined : String(context.accessKeyId),
  }
  if (sameMonitorQuery(route.query, scheduleMonitorQuery(next))) return
  void router.push(monitorLocation(scheduleMonitorQuery(next)))
}

function refreshScheduleRoute(): void {
  void router.replace(monitorLocation(scheduleMonitorQuery(scheduleState.value)))
}

const scheduleLabels = computed<SchedulePanelLabels>(() => ({
  model: t('monitor.schedule.panel.model'),
  protocol: t('monitor.schedule.panel.protocol'),
  operation: t('monitor.schedule.panel.operation'),
  accessKey: t('monitor.schedule.panel.accessKey'),
  selectModel: t('monitor.schedule.panel.selectModel'),
  selectProtocol: t('monitor.schedule.panel.selectProtocol'),
  selectAccessKey: t('monitor.schedule.panel.selectAccessKey'),
  loadingOptions: t('monitor.schedule.panel.loadingOptions'),
  optionsFailed: t('monitor.schedule.panel.optionsFailed'),
  contextRequired: t('monitor.schedule.panel.contextRequired'),
  kicker: t('monitor.schedule.panel.kicker'),
  contextReady: t('monitor.schedule.panel.contextReady'),
  context: t('monitor.schedule.panel.context'),
  disabled: t('monitor.schedule.panel.disabled'),
  retry: t('monitor.schedule.panel.retry'),
  indexFailed: t('monitor.schedule.panel.indexFailed'),
  detailFailed: t('monitor.schedule.panel.detailFailed'),
  protocolLabels: {
    'openai-completions': t('monitor.schedule.protocols.openai-completions'),
    'openai-responses': t('monitor.schedule.protocols.openai-responses'),
    'openai-images': t('monitor.schedule.protocols.openai-images'),
    'openai-embeddings': t('monitor.schedule.protocols.openai-embeddings'),
    anthropic: t('monitor.schedule.protocols.anthropic'),
    gemini: t('monitor.schedule.protocols.gemini'),
  },
  operationLabels: {
    chat_completion: t('monitor.schedule.operations.chat_completion'),
    responses_create: t('monitor.schedule.operations.responses_create'),
    responses_retrieve: t('monitor.schedule.operations.responses_retrieve'),
    responses_delete: t('monitor.schedule.operations.responses_delete'),
    responses_cancel: t('monitor.schedule.operations.responses_cancel'),
    responses_input_items: t('monitor.schedule.operations.responses_input_items'),
    responses_compact: t('monitor.schedule.operations.responses_compact'),
    responses_input_tokens: t('monitor.schedule.operations.responses_input_tokens'),
    count_tokens: t('monitor.schedule.operations.count_tokens'),
    responses_passthrough: t('monitor.schedule.operations.responses_passthrough'),
    images_generate: t('monitor.schedule.operations.images_generate'),
    images_edit: t('monitor.schedule.operations.images_edit'),
    embeddings_create: t('monitor.schedule.operations.embeddings_create'),
  },
  index: {
    title: t('monitor.schedule.index.title'),
    refresh: t('monitor.schedule.index.refresh'),
    loading: t('monitor.schedule.index.loading'),
    failed: t('monitor.schedule.index.failed'),
    empty: t('monitor.schedule.index.empty'),
    candidates: t('monitor.schedule.index.candidates'),
    groups: t('monitor.schedule.index.groups'),
    fallback: t('monitor.schedule.index.fallback'),
    cooldown: t('monitor.schedule.index.cooldown'),
    blacklist: t('monitor.schedule.index.blacklist'),
    select: t('monitor.schedule.index.select'),
    eyebrow: t('monitor.schedule.index.eyebrow'),
    routeExceptions: t('monitor.schedule.index.routeExceptions'),
  },
  detail: {
    title: t('monitor.schedule.detail.title'),
    loading: t('monitor.schedule.detail.loading'),
    failed: t('monitor.schedule.detail.failed'),
    refresh: t('monitor.schedule.detail.refresh'),
    observed: t('monitor.schedule.detail.observed'),
    stale: t('monitor.schedule.detail.stale'),
    routeUnavailable: t('monitor.schedule.detail.routeUnavailable'),
    groupWeight: t('monitor.schedule.detail.groupWeight'),
    channel: t('monitor.schedule.detail.channel'),
    entryId: t('monitor.schedule.detail.entryId'),
    weight: t('monitor.schedule.detail.weight'),
    priority: t('monitor.schedule.detail.priority'),
    fallback: t('monitor.schedule.detail.fallback'),
    share: t('monitor.schedule.detail.share'),
    reason: t('monitor.schedule.detail.reason'),
    runtime: t('monitor.schedule.detail.runtime'),
    available: t('monitor.schedule.detail.available'),
    cooldown: t('monitor.schedule.detail.cooldown'),
    blacklisted: t('monitor.schedule.detail.blacklisted'),
    failures: t('monitor.schedule.detail.failures'),
    recover: t('monitor.schedule.detail.recover'),
    credentials: t('monitor.schedule.detail.credentials'),
    breaker: t('monitor.schedule.detail.breaker'),
    threshold: t('monitor.schedule.detail.threshold'),
    cooldownSeconds: t('monitor.schedule.detail.cooldownSeconds'),
    effective: t('monitor.schedule.detail.effective'),
    configured: t('monitor.schedule.detail.configured'),
    inherited: t('monitor.schedule.detail.inherited'),
    clear: t('monitor.schedule.detail.clear'),
    invalidValue: t('monitor.schedule.detail.invalidValue'),
    derivedReadOnly: t('monitor.schedule.detail.derivedReadOnly'),
    save: t('monitor.schedule.detail.save'),
    discard: t('monitor.schedule.detail.discard'),
    unsaved: t('monitor.schedule.detail.unsaved'),
    saved: t('monitor.schedule.detail.saved'),
    saveFailed: t('monitor.schedule.detail.saveFailed'),
    conflict: t('monitor.schedule.detail.conflict'),
    refreshToResolve: t('monitor.schedule.detail.refreshToResolve'),
    recoverFailed: t('monitor.schedule.detail.recoverFailed'),
    noEntries: t('monitor.schedule.detail.noEntries'),
    unknownReason: t('monitor.schedule.detail.unknownReason'),
    unknownRouteStrategy: t('monitor.schedule.detail.unknownRouteStrategy'),
    unknownBreakerSource: t('monitor.schedule.detail.unknownBreakerSource'),
    nativeRoute: t('monitor.schedule.detail.nativeRoute'),
    draftPreview: t('monitor.schedule.detail.draftPreview'),
    observedShare: t('monitor.schedule.detail.observedShare'),
    reasonLabels: Object.fromEntries(
      [
        'access_key_disabled',
        'access_key_expired',
        'protocol_filtered',
        'model_filtered',
        'model_required_by_filter',
        'operation_unsupported',
        'native_route_required',
        'no_route_target',
        'group_disabled',
        'group_filtered',
        'no_available_group',
        'no_credentials',
        'group_weight_zero',
        'credential_disabled',
        'credential_blacklisted',
        'credential_cooldown',
        'credential_auth_unavailable',
        'credential_weight_zero',
        'credential_not_allowed',
        'no_available_credential',
        'entry_blacklisted',
        'entry_cooldown',
        'entry_weight_zero',
        'tier_demoted',
      ].map((code) => [code, t(`monitor.schedule.reasons.${code}`)]),
    ),
    routeStrategyLabels: {
      native_first: t('monitor.schedule.routeStrategies.native_first'),
      weighted_mix: t('monitor.schedule.routeStrategies.weighted_mix'),
    },
    sourceLabels: {
      default: t('monitor.schedule.sources.default'),
      entry: t('monitor.schedule.sources.entry'),
    },
  },
}))
</script>

<template>
  <PageFrame aria-labelledby="monitor-title">
    <LedgerSheet class="monitor-page">
      <PageHeader id="monitor-title" :title="t('monitor.title')" />
      <AppTabs
        class="monitor-tabs"
        :model-value="activeTab"
        :label="t('monitor.tabs.label')"
        :items="items"
        appearance="detail"
        @update:model-value="selectTab"
      >
        <template #actions>
          <AppButton
            v-if="activeTab === 'health'"
            class="monitor-refresh"
            variant="secondary"
            size="compact"
            :busy="healthRefreshPending"
            @click="refreshHealth"
          >
            <RefreshCw
              :class="{ 'monitor-refresh-icon--spinning': healthRefreshPending }"
              :size="14"
              aria-hidden="true"
            />
            {{ t('monitor.health.refresh') }}
          </AppButton>
          <div v-else-if="activeTab === 'usage'" class="monitor-usage-actions">
            <AppSelect
              :model-value="usageFilters.range"
              :label="t('monitor.usage.filters.range')"
              :options="usageRangeOptions"
              size="compact"
              @update:model-value="selectUsageRange"
            />
            <AppButton variant="secondary" size="compact" @click="usageTab?.openFilters()">
              <ListFilter :size="14" aria-hidden="true" />
              {{ t('monitor.usage.filters.button') }}
              <span v-if="usageFilterCount > 0" class="monitor-filter-count">
                {{ usageFilterCount }}
              </span>
            </AppButton>
            <AppButton
              class="monitor-refresh"
              variant="secondary"
              size="compact"
              :busy="usageRefreshPending"
              @click="refreshUsage"
            >
              <RefreshCw
                :class="{ 'monitor-refresh-icon--spinning': usageRefreshPending }"
                :size="14"
                aria-hidden="true"
              />
              {{ t('monitor.usage.filters.refresh') }}
            </AppButton>
          </div>
        </template>

        <template v-if="isCanonicalQuery">
          <div v-if="activeTab === 'health'" class="monitor-panel">
            <HealthTab ref="healthTab" />
          </div>
          <div v-else-if="activeTab === 'logs'" class="monitor-panel">
            <LogsTab />
          </div>
          <div v-else-if="activeTab === 'usage'" class="monitor-panel">
            <UsageTab ref="usageTab" />
          </div>
          <div v-else-if="activeTab === 'schedule' && isAdmin" class="monitor-panel">
            <SchedulePanel
              :external-model="scheduleState.externalModel"
              :protocol="scheduleState.protocol"
              :operation="scheduleState.operation"
              :access-key-id="scheduleAccessKeyID"
              :labels="scheduleLabels"
              :locale="locale"
              @update:external-model="updateScheduleContext({ externalModel: $event })"
              @update:protocol="updateScheduleContext({ protocol: $event })"
              @update:operation="updateScheduleContext({ operation: $event })"
              @update:access-key-id="
                updateScheduleContext({
                  accessKeyID: $event === undefined ? undefined : String($event),
                })
              "
              @change-context="commitScheduleContext"
              @saved="refreshScheduleRoute"
              @recovered="refreshScheduleRoute"
              @refresh="refreshScheduleRoute"
            />
          </div>
          <div v-else-if="activeTab === 'inspector'" class="monitor-panel">
            <InspectorTab />
          </div>
        </template>
      </AppTabs>
    </LedgerSheet>
  </PageFrame>
</template>

<style scoped>
.monitor-page {
  display: grid;
  min-height: 760px;
  min-width: 0;
  align-content: start;
  gap: 0;
}

.monitor-panel {
  min-width: 0;
  padding-top: var(--detail-panel-padding-top);
}

.monitor-tabs :deep(.app-tabs__bar) {
  border-top: 0;
}

.monitor-refresh-icon--spinning {
  animation: monitor-refresh-spin 800ms linear infinite;
}

.monitor-refresh[aria-busy='true'] {
  opacity: 1;
}

.monitor-usage-actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.monitor-filter-count {
  display: inline-grid;
  min-width: 17px;
  height: 17px;
  place-items: center;
  border-radius: 999px;
  background: var(--color-action-soft);
  color: var(--color-action);
  padding-inline: 4px;
  font-family: var(--font-mono);
  font-size: 10px;
}

@keyframes monitor-refresh-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 800px) {
  .monitor-page {
    min-height: 0;
  }

  .monitor-panel {
    padding-top: var(--detail-panel-padding-top-compact);
  }
}

@media (max-width: 560px) {
  .monitor-usage-actions {
    gap: var(--space-1);
  }
}

@media (prefers-reduced-motion: reduce) {
  .monitor-refresh-icon--spinning {
    animation: none;
  }
}
</style>
