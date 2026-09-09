<script setup lang="ts">
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, watch } from 'vue'
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
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import SkeletonSurface from '@/components/ui/SkeletonSurface.vue'

import GroupHeader from './GroupHeader.vue'
import GroupCredentialsTab from './credentials/GroupCredentialsTab.vue'
import GroupModelsTab from './models/GroupModelsTab.vue'
import GroupSettingsTab from './settings/GroupSettingsTab.vue'
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
  <PageFrame aria-labelledby="group-detail-title">
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
            <GroupSettingsTab :key="`settings-${groupId}`" :group-id="groupId" unified />
            <section class="group-detail__credentials" aria-labelledby="group-credentials-summary">
              <div class="group-detail__section-heading">
                <div>
                  <h2 id="group-credentials-summary">{{ t('group.unified.credentialsTitle') }}</h2>
                  <p>{{ t('group.unified.credentialsSummary') }}</p>
                </div>
                <RouterLink
                  class="button-link"
                  :to="{
                    name: 'group-detail',
                    params: { id: groupId },
                    query: { tab: 'credentials' },
                  }"
                >
                  {{ t('group.unified.manageCredentials') }}
                </RouterLink>
              </div>
              <div
                v-if="credentialsQuery.data.value?.items.length"
                class="group-detail__credential-list"
              >
                <div
                  v-for="credential in credentialsQuery.data.value.items"
                  :key="credential.mask"
                  class="group-detail__credential-summary"
                >
                  <div>
                    <strong>{{ credential.mask }}</strong>
                    <span>{{
                      credential.connection_type === 'subscription'
                        ? t('group.credentials.full.kind.account')
                        : t('group.credentials.full.kind.key')
                    }}</span>
                  </div>
                  <StatusBadge :status="unifiedCredentialSummary(credential).status" size="compact">
                    {{ unifiedCredentialSummary(credential).label }}
                  </StatusBadge>
                </div>
              </div>
              <p v-else class="group-detail__empty">{{ t('group.unified.noCredentials') }}</p>
            </section>
            <GroupModelsTab
              :key="`models-${groupId}`"
              :group-id="groupId"
              :channel-id="summaryQuery.data.value.channel_id"
              readonly-route-fields
            />
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
  gap: 0;
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
  gap: var(--space-4);
  border-top: 1px solid var(--color-border-subtle);
  padding: var(--space-6) 0;
}
.group-detail__section-heading {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: var(--space-4);
}
.group-detail__section-heading h2,
.group-detail__section-heading p {
  margin: 0;
}
.group-detail__section-heading h2 {
  font-size: var(--text-body);
}
.group-detail__section-heading p,
.group-detail__empty,
.group-detail__credential-summary span {
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}
.group-detail__section-heading p {
  margin-top: 4px;
}
.group-detail__credential-list {
  display: grid;
  gap: var(--space-2);
}
.group-detail__credential-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-control);
  padding: var(--space-3);
}
.group-detail__credential-summary > div {
  display: grid;
  min-width: 0;
  gap: 3px;
}
.group-detail__credential-summary strong {
  overflow-wrap: anywhere;
  font-family: var(--font-mono);
}
.group-detail__empty {
  margin: 0;
}
</style>
