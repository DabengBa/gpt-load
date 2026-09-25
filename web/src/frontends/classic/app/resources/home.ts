import { queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { controlQueryKeys } from '@shared/control/query-keys'
import type { HomeRange } from '@shared/control/resources/home'
import {
  getHomeStatistics,
  getHomeSubscriptionAccounts,
} from '@shared/control/resources/home'
import { projectEnum } from '@shared/control/resources/projector'

export * from '@shared/control/resources/home'

export function homeSubscriptionAccountsQueryOptions(
  client: ApiClient,
  enabled: MaybeRefOrGetter<boolean>,
) {
  return queryOptions({
    queryKey: controlQueryKeys.home.subscriptionAccounts(),
    queryFn: ({ signal }) => getHomeSubscriptionAccounts(client, signal),
    enabled: computed(() => toValue(enabled)),
    refetchOnMount: 'always',
  })
}

export function homeStatisticsQueryOptions(client: ApiClient, range: MaybeRefOrGetter<HomeRange>) {
  return queryOptions({
    queryKey: computed(() => controlQueryKeys.home.statistics(toValue(range))),
    queryFn: ({ queryKey, signal }) => {
      const queryRange = projectEnum(queryKey[3], ['24h', '30d'] as const)
      return getHomeStatistics(client, queryRange, signal)
    },
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnMount: 'always',
  })
}
