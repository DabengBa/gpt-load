import { keepPreviousData, queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { controlQueryKeys } from '@shared/control/query-keys'
import type { ModelCollectionFilters } from '@shared/control/resources/models'
import {
  listModels,
  normalizeModelCollectionFilters,
} from '@shared/control/resources/models'

export * from '@shared/control/resources/models'

export function modelCollectionQueryOptions(
  client: ApiClient,
  filters: MaybeRefOrGetter<ModelCollectionFilters>,
  isAccessKey: MaybeRefOrGetter<boolean> = false,
) {
  return queryOptions({
    queryKey: computed(() =>
      controlQueryKeys.models.collection(normalizeModelCollectionFilters(toValue(filters))),
    ),
    queryFn: ({ queryKey, signal }) =>
      listModels(client, queryKey[3], signal, toValue(isAccessKey)),
    placeholderData: keepPreviousData,
  })
}
