import { keepPreviousData, queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { controlQueryKeys } from '@shared/control/query-keys'
import {
  listChannels,
  normalizeChannelSearch,
} from '@shared/control/resources/channels'

export * from '@shared/control/resources/channels'

export function channelsQueryOptions(client: ApiClient, search: MaybeRefOrGetter<string>) {
  return queryOptions({
    queryKey: computed(() =>
      controlQueryKeys.channels.list(normalizeChannelSearch(toValue(search))),
    ),
    queryFn: ({ queryKey, signal }) => listChannels(client, queryKey[3], signal),
    staleTime: 5 * 60 * 1_000,
    placeholderData: keepPreviousData,
  })
}
