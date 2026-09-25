import { keepPreviousData, queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { controlQueryKeys } from '@shared/control/query-keys'
import type { ModelPriceFilters } from '@shared/control/resources/model-prices'
import {
  listModelPrices,
  normalizeModelPriceFilters,
} from '@shared/control/resources/model-prices'

export * from '@shared/control/resources/model-prices'

export function modelPriceCollectionQueryOptions(
  client: ApiClient,
  filters: MaybeRefOrGetter<ModelPriceFilters>,
) {
  return queryOptions({
    queryKey: computed(() =>
      controlQueryKeys.modelPriceCollection(normalizeModelPriceFilters(toValue(filters))),
    ),
    queryFn: ({ queryKey, signal }) => listModelPrices(client, queryKey[3], signal),
    placeholderData: keepPreviousData,
  })
}
