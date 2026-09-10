<script setup lang="ts">
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'

import { useApiClient } from '@/api/client-context'
import { useStableLoading } from '@/app/loading-state'
import {
  groupModelsQueryOptions,
  groupSettingsQueryOptions,
  groupSummaryQueryOptions,
} from '@/app/resources/groups'
import { credentialCollectionQueryOptions } from '@/app/resources/credentials'
import type { CredentialCollectionFilters, CredentialItemDto } from '@/api/control/types'
import { groupsLocation } from '@/app/route-locations'
import LedgerSheet from '@/components/layout/LedgerSheet.vue'
import PageFrame from '@/components/layout/PageFrame.vue'
import AsyncRefreshIndicator from '@/components/ui/AsyncRefreshIndicator.vue'
import AppButton from '@/components/ui/AppButton.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import SkeletonSurface from '@/components/ui/SkeletonSurface.vue'
import StickySaveBar from '@/components/ui/StickySaveBar.vue'

import GroupHeader from './GroupHeader.vue'
import GroupCredentialsTab from './credentials/GroupCredentialsTab.vue'
import GroupModelsTab from './models/GroupModelsTab.vue'
import GroupSettingsTab from './settings/GroupSettingsTab.vue'
import GroupDeleteDialog from './settings/GroupDeleteDialog.vue'
import { parseCredentialRouteQuery, parsePositiveId } from './group-route'
import StatusBadge from '@/components/ui/StatusBadge.vue'
import type { OperationalStatus } from '@/components/ui/status-presenter'

const route = useRoute()
const client = useApiClient()
const queryClient = useQueryClient()
const { t } = useI18n()
const groupId = computed(() => parsePositiveId(route.params.id))
const summaryQuery = useQuery(groupSummaryQueryOptions(client, groupId))
const credentialFilters: CredentialCollectionFilters = { page: 1, page_size: 20 }
const credentialsQuery = useQuery({
  ...credentialCollectionQueryOptions(client, () => groupId.value as number, credentialFilters),
  enabled: computed(() => groupId.value !== undefined),
})
const managementOpen = computed(() => route.query.tab === 'credentials')
const initialLoading = useStableLoading(
  () => summaryQuery.isPending.value && summaryQuery.data.value === undefined,
)
const summaryRefreshing = computed(
  () => summaryQuery.data.value !== undefined && summaryQuery.isFetching.value,
)

type GroupEditorState = {
  dirty: boolean
  pending: boolean
  error: string
  saved: boolean
  invalidRowCount?: number
}

const settingsState = ref<GroupEditorState>({
  dirty: false,
  pending: false,
  error: '',
  saved: false,
})
const modelsState = ref<GroupEditorState>({
  dirty: false,
  pending: false,
  error: '',
  saved: false,
})
const deletePending = ref(false)
const unifiedDirty = computed(() => settingsState.value.dirty || modelsState.value.dirty)
const unifiedPending = computed(
  () => settingsState.value.pending || modelsState.value.pending || deletePending.value,
)
const unifiedError = computed(() => settingsState.value.error || modelsState.value.error)
const unifiedSaved = computed(
  () => !unifiedDirty.value && (settingsState.value.saved || modelsState.value.saved),
)
const unifiedSaveStatus = computed<'idle' | 'saved' | 'error'>(() => {
  if (unifiedError.value) return 'error'
  if (unifiedSaved.value) return 'saved'
  return 'idle'
})
const unifiedErrorActionLabel = computed(() =>
  modelsState.value.invalidRowCount ? t('group.modelEditor.locateFirstInvalid') : '',
)
const settingsEditor = ref<{ requestSave: () => void; discard: () => void }>()
const modelsEditor = ref<{
  requestSave: () => void
  discard: () => void
  focusFirstInvalid: () => Promise<void>
}>()

function saveUnified(): void {
  if (settingsState.value.dirty) settingsEditor.value?.requestSave()
  if (modelsState.value.dirty) modelsEditor.value?.requestSave()
}

function discardUnified(): void {
  if (settingsState.value.dirty) settingsEditor.value?.discard()
  if (modelsState.value.dirty) modelsEditor.value?.discard()
}

type UnifiedCredentialSummary = {
  status: OperationalStatus
  label: string
}

function unifiedCredentialSummary(credential: CredentialItemDto): UnifiedCredentialSummary {
  if (credential.auth_state === 'refreshing') {
    return {
      status: 'unknown',
      label: t('group.credentials.subscription.status.refreshing'),
    }
  }
  if (credential.auth_state === 'reauthorization_required') {
    return {
      status: 'unavailable',
      label: t('group.credentials.subscription.status.needs_reauth'),
    }
  }
  if (credential.auth_state === 'outcome_unknown') {
    return {
      status: 'unknown',
      label: t('group.credentials.subscription.status.outcome_unknown'),
    }
  }
  return {
    status: credential.effective_status,
    label: t(`group.credentials.effective.${credential.effective_status}`),
  }
}

watch(
  groupId,
  (id) => {
    if (id === undefined) return
    void Promise.allSettled([
      queryClient.prefetchQuery(
        credentialCollectionQueryOptions(client, id, parseCredentialRouteQuery(route.query)),
      ),
      queryClient.prefetchQuery(groupModelsQueryOptions(client, id)),
      queryClient.prefetchQuery(groupSettingsQueryOptions(client, id)),
    ])
  },
  { immediate: true },
)
</script>

<template>
  <PageFrame wide aria-labelledby="group-detail-title">
    <LedgerSheet class="group-detail-page">
      <div v-if="groupId === undefined" class="group-detail-invalid" role="alert">
        <h1 id="group-detail-title">{{ t('group.invalidTitle') }}</h1>
        <p>{{ t('group.invalidDescription') }}</p>
        <RouterLink class="button-link" :to="groupsLocation()">{{
          t('group.backToGroups')
        }}</RouterLink>
      </div>
      <template v-else>
        <AsyncRefreshIndicator :active="summaryRefreshing" :label="t('group.loading')" />
        <SkeletonSurface
          v-if="(summaryQuery.isPending.value && !summaryQuery.data.value) || initialLoading"
          variant="detail"
          :concealed="!initialLoading"
          :label="t('group.loading')"
        />
        <QueryFeedback
          v-else-if="summaryQuery.isError.value && !summaryQuery.data.value"
          state="error"
          :message="t('group.loadFailed')"
          :retry-label="t('common.retry')"
          @retry="summaryQuery.refetch()"
        />
        <template v-else-if="summaryQuery.data.value">
          <QueryFeedback
            v-if="summaryQuery.isError.value"
            state="stale"
            :message="t('group.stale')"
            :retry-label="t('common.retry')"
            @retry="summaryQuery.refetch()"
          />
          <GroupHeader :group="summaryQuery.data.value" />
          <GroupCredentialsTab
            v-if="managementOpen"
            :key="`management-${groupId}`"
            :group-id="groupId"
            :channel-id="summaryQuery.data.value.channel_id"
            :connection-type="summaryQuery.data.value.connection_type"
          />
          <template v-else>
            <GroupSettingsTab
              ref="settingsEditor"
              :key="`settings-${groupId}`"
              :group-id="groupId"
              unified
              :blocked="deletePending"
              @state="settingsState = $event"
            />
            <section class="group-detail__credentials" :aria-label="t('group.credentials.title')">
              <div
                v-if="credentialsQuery.data.value?.items.length"
                class="group-detail__credential-input-list"
              >
                <div
                  v-for="credential in credentialsQuery.data.value.items"
                  :key="credential.mask"
                  class="group-detail__credential-input-row"
                >
                  <label class="group-detail__credential-field">
                    <span>{{
                      credential.connection_type === 'subscription'
                        ? t('group.credentials.full.kind.account')
                        : t('group.credentials.full.kind.key')
                    }}</span>
                    <input :value="credential.mask" readonly autocomplete="off" />
                  </label>
                  <StatusBadge :status="unifiedCredentialSummary(credential).status" size="compact">
                    {{ unifiedCredentialSummary(credential).label }}
                  </StatusBadge>
                </div>
              </div>
              <p v-else class="group-detail__empty">{{ t('group.unified.noCredentials') }}</p>
            </section>
            <GroupModelsTab
              ref="modelsEditor"
              :key="`models-${groupId}`"
              :group-id="groupId"
              :channel-id="summaryQuery.data.value.channel_id"
              unified
              :blocked="deletePending"
              readonly-route-fields
              @state="modelsState = $event"
            />
            <div id="group-settings-advanced-target" class="group-detail__advanced-target" />
            <StickySaveBar
              class="group-detail__save-bar"
              appearance="ledger"
              always-visible
              :dirty="unifiedDirty"
              :pending="unifiedPending"
              :status="unifiedSaveStatus"
              :error="unifiedError"
              :error-action-label="unifiedErrorActionLabel"
              @error-action="modelsEditor?.focusFirstInvalid()"
            >
              <template #status>
                <div>
                  <strong>
                    {{
                      unifiedPending
                        ? t('group.settings.saving')
                        : unifiedSaved
                          ? t('group.settings.savedFeedback')
                          : unifiedDirty
                            ? t('group.settings.unsaved')
                            : t('group.settings.saved')
                    }}
                  </strong>
                  <span>{{ unifiedError || t('group.settings.saveNote') }}</span>
                </div>
              </template>
              <template #discard="{ disabled }">
                <AppButton
                  variant="ghost"
                  size="sm"
                  :disabled="disabled || !unifiedDirty"
                  @click="discardUnified"
                >
                  {{ t('common.discard') }}
                </AppButton>
              </template>
              <template #save="{ disabled }">
                <GroupDeleteDialog
                  :group-id="groupId"
                  :group-name="summaryQuery.data.value.name"
                  :disabled="disabled || unifiedPending || unifiedDirty"
                  @update:pending="deletePending = $event"
                />
                <AppButton size="sm" :disabled="disabled || !unifiedDirty" @click="saveUnified">
                  {{ t('group.settings.save') }}
                </AppButton>
              </template>
            </StickySaveBar>
          </template>
        </template>
      </template>
    </LedgerSheet>
  </PageFrame>
</template>

<style scoped>
.group-detail-page {
  display: grid;
  align-content: start;
  gap: var(--space-3);
  padding-bottom: var(--space-3);
}
.group-detail-page :deep(.group-settings),
.group-detail-page :deep(.group-models) {
  padding-top: 15px;
}
.group-detail-page :deep(.panel-header) {
  min-height: 47px;
  margin-bottom: 12px;
  padding-bottom: 10px;
}
.group-detail-page :deep(.group-settings__content) {
  gap: 12px;
}
.group-detail-page :deep(.group-settings__section) {
  gap: 10px;
  padding-top: 12px;
}
.group-detail-page :deep(.group-settings__section:first-child) {
  padding-top: 0;
}
.group-detail-page :deep(.group-settings__grid) {
  gap: 10px 14px;
}
.group-detail-page :deep(.group-settings__field) {
  gap: 4px;
}
.group-detail-page :deep(.group-settings__switch-row) {
  min-height: 42px;
  padding: 4px 2px;
}
.group-detail-page :deep(.group-settings__advanced) {
  padding-top: 8px;
}
.group-detail-page :deep(.group-settings__advanced-content) {
  gap: 8px;
  margin-top: 8px;
}
.group-detail-page :deep(.group-settings__advanced-sections) {
  gap: 12px;
}
.group-detail-page :deep(.group-settings__runtime) {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  column-gap: var(--space-6);
  row-gap: 0;
}
.group-detail-page :deep(.setting-row) {
  padding-block: 5px;
}
.group-detail-page :deep(.setting-row__cluster) {
  min-height: 26px;
  gap: 10px;
}
.group-detail-page :deep(.model-alias-editor__grid) {
  --ledger-record-list-record-min-height: 48px;
  --ledger-record-list-record-padding: 6px 0;
  --ledger-record-list-column-gap: 12px;
}
.group-detail-page :deep(.group-settings__section header p) {
  max-width: 720px;
}
.group-detail-page :deep(.group-detail__save-bar[data-status='idle']),
.group-detail-page :deep(.group-detail__save-bar[data-status='saved']) {
  position: static;
  margin: 12px 0 0;
}

.group-detail-invalid {
  display: grid;
  max-width: 640px;
  gap: var(--space-3);
}
.group-detail-invalid h1,
.group-detail-invalid p {
  margin: 0;
}
.group-detail-invalid p {
  color: var(--color-text-muted);
}
.group-detail__credentials {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  row-gap: 8px;
  border-top: 1px solid var(--color-border-subtle);
  padding: 12px 0;
}
.group-detail__empty,
.group-detail__credential-field > span {
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}
.group-detail__credential-input-list {
  display: grid;
  min-width: 0;
  grid-column: 1;
  gap: 6px;
}
.group-detail__credential-input-row {
  display: flex;
  min-width: 0;
  align-items: end;
  gap: var(--space-3);
}
.group-detail__credential-field {
  display: grid;
  min-width: 0;
  flex: 1;
  gap: 4px;
}
.group-detail__credential-field input {
  width: 100%;
  min-height: var(--control-md);
  border: 1px solid var(--color-border-control);
  border-radius: var(--radius-control);
  background: var(--color-surface);
  color: var(--color-text);
  padding: 0 var(--space-3);
  font: inherit;
  font-family: var(--font-mono);
}
.group-detail__credential-input-row > :last-child {
  flex: none;
  margin-bottom: 1px;
}
.group-detail__empty {
  grid-column: 1;
  margin: 0;
}
@media (min-width: 1100px) {
  .group-detail-page :deep(.group-settings__grid) {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .group-detail-page :deep(.group-settings__base-url-input) {
    grid-column: span 2;
  }
}
@media (max-width: 960px) {
  .group-detail-page :deep(.group-settings__runtime) {
    grid-template-columns: minmax(0, 1fr);
  }
}
@media (max-width: 640px) {
  .group-detail__credentials {
    padding-top: 10px;
    padding-bottom: 10px;
  }
}
</style>
