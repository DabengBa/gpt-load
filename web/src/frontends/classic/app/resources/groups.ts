import { keepPreviousData, queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { InvalidResponseError } from '@shared/http/errors'
import { controlQueryKeys } from '@shared/control/query-keys'
import type { GroupCollectionFilters } from '@shared/control/types'
import {
  getGroupModels,
  getGroupSettings,
  getGroupSummary,
  listGroupCollection,
  listGroupOptions,
  manualGroupQueryOptions,
} from '@shared/control/resources/groups'

export * from '@shared/control/resources/groups'

export function groupCollectionQueryOptions(
  client: ApiClient,
  filters: MaybeRefOrGetter<GroupCollectionFilters>,
) {
  return queryOptions({
    ...manualGroupQueryOptions,
    queryKey: computed(() => controlQueryKeys.groups.collection(toValue(filters))),
    queryFn: ({ queryKey, signal }) => listGroupCollection(client, queryKey[3], signal),
    placeholderData: keepPreviousData,
  })
}

export function groupOptionsQueryOptions(
  client: ApiClient,
  enabled: MaybeRefOrGetter<boolean> = true,
) {
  return queryOptions({
    ...manualGroupQueryOptions,
    queryKey: controlQueryKeys.groups.options(),
    queryFn: ({ signal }) => listGroupOptions(client, signal),
    enabled: computed(() => toValue(enabled)),
    // Group 选项目录可能由其他页面或标签页修改，进入消费者页面时必须重新校验。
    refetchOnMount: 'always',
  })
}

export function groupSummaryQueryOptions(
  client: ApiClient,
  groupID: MaybeRefOrGetter<number | undefined>,
) {
  return queryOptions({
    ...manualGroupQueryOptions,
    queryKey: computed(() => {
      const id = toValue(groupID)
      return id === undefined
        ? controlQueryKeys.groups.summaries()
        : controlQueryKeys.groups.summary(id)
    }),
    queryFn: ({ signal }) => {
      const id = toValue(groupID)
      if (id === undefined) throw new InvalidResponseError()
      return getGroupSummary(client, id, signal)
    },
    enabled: computed(() => toValue(groupID) !== undefined),
  })
}

export function groupSettingsQueryOptions(
  client: ApiClient,
  groupID: MaybeRefOrGetter<number | undefined>,
) {
  return queryOptions({
    ...manualGroupQueryOptions,
    queryKey: computed(() => {
      const id = toValue(groupID)
      return id === undefined
        ? controlQueryKeys.groups.settingsAll()
        : controlQueryKeys.groups.settings(id)
    }),
    queryFn: ({ signal }) => {
      const id = toValue(groupID)
      if (id === undefined) throw new InvalidResponseError()
      return getGroupSettings(client, id, signal)
    },
    enabled: computed(() => toValue(groupID) !== undefined),
  })
}

export function groupModelsQueryOptions(
  client: ApiClient,
  groupID: MaybeRefOrGetter<number | undefined>,
) {
  return queryOptions({
    ...manualGroupQueryOptions,
    queryKey: computed(() => {
      const id = toValue(groupID)
      return id === undefined
        ? controlQueryKeys.groups.modelsAll()
        : controlQueryKeys.groups.models(id)
    }),
    queryFn: ({ signal }) => {
      const id = toValue(groupID)
      if (id === undefined) throw new InvalidResponseError()
      return getGroupModels(client, id, signal)
    },
    enabled: computed(() => toValue(groupID) !== undefined),
  })
}
