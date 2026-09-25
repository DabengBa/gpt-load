import { keepPreviousData, queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { controlQueryKeys } from '@shared/control/query-keys'
import type { CredentialCollectionFilters } from '@shared/control/types'
import {
  getCredentialCollection,
  manualGroupQueryOptions,
} from '@shared/control/resources/credentials'

export * from '@shared/control/resources/credentials'

export function credentialCollectionQueryOptions(
  client: ApiClient,
  groupID: MaybeRefOrGetter<number>,
  filters: MaybeRefOrGetter<CredentialCollectionFilters>,
) {
  return queryOptions({
    ...manualGroupQueryOptions,
    queryKey: computed(() =>
      controlQueryKeys.groups.credentials(toValue(groupID), toValue(filters)),
    ),
    queryFn: ({ queryKey, signal }) =>
      getCredentialCollection(client, queryKey[3], queryKey[5], signal),
    placeholderData: keepPreviousData,
  })
}
