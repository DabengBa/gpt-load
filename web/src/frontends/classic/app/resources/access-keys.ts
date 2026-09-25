import { keepPreviousData, queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { controlQueryKeys } from '@shared/control/query-keys'
import type { AccessKeyCollectionFilters } from '@shared/control/types'
import {
  listAccessKeyCollection,
  listAccessKeyOptions,
} from '@shared/control/resources/access-keys'

export * from '@shared/control/resources/access-keys'

export function accessKeyCollectionQueryOptions(
  client: ApiClient,
  filters: MaybeRefOrGetter<AccessKeyCollectionFilters>,
) {
  return queryOptions({
    queryKey: computed(() => controlQueryKeys.accessKeys.collection(toValue(filters))),
    queryFn: ({ queryKey, signal }) => listAccessKeyCollection(client, queryKey[3], signal),
    placeholderData: keepPreviousData,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
}

export function accessKeyOptionsQueryOptions(
  client: ApiClient,
  enabled: MaybeRefOrGetter<boolean> = true,
) {
  return queryOptions({
    queryKey: controlQueryKeys.accessKeys.options(),
    queryFn: ({ signal }) => listAccessKeyOptions(client, signal),
    enabled: computed(() => toValue(enabled)),
    gcTime: 0,
  })
}
