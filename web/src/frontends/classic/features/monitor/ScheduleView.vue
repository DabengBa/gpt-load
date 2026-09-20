<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import { lazySurface } from '@/app/async-surface'
import { scheduleLocation } from '@/app/route-locations'
import LedgerSheet from '@/components/layout/LedgerSheet.vue'
import PageFrame from '@/components/layout/PageFrame.vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import { useAuthSession } from '@/features/auth/auth-session'

import type { ScheduleDrafts, ScheduleMode, ScheduleMonitorState } from './monitor-route'
import { parseScheduleMonitorState, sameMonitorQuery, scheduleMonitorQuery } from './monitor-route'
import type { SchedulePanelLabels } from './SchedulePanel.vue'

const SchedulePanel = lazySurface(() => import('./SchedulePanel.vue'))

const route = useRoute()
const session = useAuthSession()
const router = useRouter()
const { locale, t } = useI18n()

const isAdmin = computed(() => session.state.principalType === 'admin')
const scheduleState = computed(() => parseScheduleMonitorState(route.query))
const pendingScheduleState = ref<ScheduleMonitorState>()
const pendingNavigationGeneration = ref(0)
const renderedScheduleState = computed(
  () => pendingScheduleState.value ?? scheduleState.value,
)
const canonicalQuery = computed(() => scheduleMonitorQuery(scheduleState.value))
const isCanonicalQuery = computed(() => sameMonitorQuery(route.query, canonicalQuery.value))

watch(
  () => route.query,
  (query) => {
    const pending = pendingScheduleState.value
    if (pending) {
      pendingNavigationGeneration.value += 1
      pendingScheduleState.value = undefined
    }
    const normalized = canonicalQuery.value
    if (!sameMonitorQuery(query, normalized)) {
      void router.replace(scheduleLocation(normalized))
    }
  },
  { immediate: true },
)

async function navigateScheduleState(
  next: ScheduleMonitorState,
  method: 'push' | 'replace',
): Promise<void> {
  const generation = pendingNavigationGeneration.value + 1
  pendingNavigationGeneration.value = generation
  pendingScheduleState.value = next
  const target = scheduleLocation(scheduleMonitorQuery(next))
  const failure = await router[method](target)
  if (generation !== pendingNavigationGeneration.value) return
  pendingScheduleState.value = undefined
  if (failure) return
}

function replaceScheduleState(next: ScheduleMonitorState): void {
  void navigateScheduleState(next, 'replace')
}

function updateScheduleContext(next: Partial<ReturnType<typeof parseScheduleMonitorState>>): void {
  const current = renderedScheduleState.value
  const contextChanged =
    next.externalModel !== undefined && next.externalModel !== current.externalModel
  const nextState = contextChanged
    ? { ...current, ...next, selectedRow: undefined, drafts: {} }
    : { ...current, ...next }
  replaceScheduleState(nextState)
}

function commitScheduleContext(context: { externalModel?: string; mode: ScheduleMode }): void {
  const current = renderedScheduleState.value
  const externalModel = context.externalModel?.trim() || undefined
  const contextChanged = externalModel !== current.externalModel
  const next = contextChanged
    ? { ...current, ...context, externalModel, selectedRow: undefined, drafts: {} }
    : { ...current, ...context, externalModel }
  if (sameMonitorQuery(route.query, scheduleMonitorQuery(next))) return
  void navigateScheduleState(next, 'push')
}

function updateScheduleDrafts(drafts: ScheduleDrafts): void {
  updateScheduleContext({ drafts })
}

function updateScheduleRow(row: string | undefined): void {
  updateScheduleContext({ selectedRow: row })
}

function refreshScheduleRoute(): void {
  replaceScheduleState(renderedScheduleState.value)
}

const scheduleLabels = computed<SchedulePanelLabels>(() => ({
  model: t('monitor.schedule.panel.model'),
  mode: t('monitor.schedule.panel.mode'),
  selectModel: t('monitor.schedule.panel.selectModel'),
  loadingOptions: t('monitor.schedule.panel.loadingOptions'),
  optionsFailed: t('monitor.schedule.panel.optionsFailed'),
  contextRequired: t('monitor.schedule.panel.contextRequired'),
  kicker: t('monitor.schedule.panel.kicker'),
  contextReady: t('monitor.schedule.panel.contextReady'),
  context: t('monitor.schedule.panel.context'),
  retry: t('monitor.schedule.panel.retry'),
  indexFailed: t('monitor.schedule.panel.indexFailed'),
  detailFailed: t('monitor.schedule.panel.detailFailed'),
  modeLabels: {
    all: t('monitor.schedule.modes.all'),
    primary: t('monitor.schedule.modes.primary'),
    fallback: t('monitor.schedule.modes.fallback'),
  },
  detail: {
    title: t('monitor.schedule.detail.title'),
    loading: t('monitor.schedule.detail.loading'),
    refresh: t('monitor.schedule.detail.refresh'),
    stale: t('monitor.schedule.detail.stale'),
    routeUnavailable: t('monitor.schedule.detail.routeUnavailable'),
    group: t('monitor.schedule.detail.group'),
    upstreamModel: t('monitor.schedule.detail.upstreamModel'),
    weight: t('monitor.schedule.detail.weight'),
    priority: t('monitor.schedule.detail.priority'),
    share: t('monitor.schedule.detail.share'),
    status: t('monitor.schedule.detail.status'),
    available: t('monitor.schedule.detail.available'),
    cooldown: t('monitor.schedule.detail.cooldown'),
    blacklisted: t('monitor.schedule.detail.blacklisted'),
    failures: t('monitor.schedule.detail.failures'),
    recover: t('monitor.schedule.detail.recover'),
    breakerRecovery: t('monitor.schedule.detail.breakerRecovery'),
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
    draftPreview: t('monitor.schedule.detail.draftPreview'),
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
        'credential_blacklisted',
        'credential_cooldown',
        'credential_auth_unavailable',
        'credential_not_allowed',
        'no_available_credential',
        'entry_blacklisted',
        'entry_cooldown',
        'entry_weight_zero',
        'tier_demoted',
      ].map((code) => [code, t(`monitor.schedule.reasons.${code}`)]),
    ),
  },
}))
</script>

<template>
  <PageFrame aria-labelledby="schedule-title">
    <LedgerSheet class="schedule-page">
      <PageHeader id="schedule-title" :title="t('shell.schedule')" />
      <div v-if="isAdmin && isCanonicalQuery" class="schedule-page__panel">
        <SchedulePanel
          :external-model="renderedScheduleState.externalModel"
          :mode="renderedScheduleState.mode"
          :selected-row="renderedScheduleState.selectedRow"
          :drafts="renderedScheduleState.drafts"
          :labels="scheduleLabels"
          :locale="locale"
          @change-context="commitScheduleContext"
          @draft-change="updateScheduleDrafts"
          @row-change="updateScheduleRow"
          @saved="refreshScheduleRoute"
          @recovered="refreshScheduleRoute"
          @refresh="refreshScheduleRoute"
        />
      </div>
    </LedgerSheet>
  </PageFrame>
</template>

<style scoped>
.schedule-page {
  display: grid;
  min-height: 760px;
  min-width: 0;
  align-content: start;
  gap: 0;
}

.schedule-page__panel {
  min-width: 0;
  padding-top: var(--detail-panel-padding-top);
}

@media (max-width: 800px) {
  .schedule-page {
    min-height: 0;
  }

  .schedule-page__panel {
    padding-top: var(--detail-panel-padding-top-compact);
  }
}
</style>
