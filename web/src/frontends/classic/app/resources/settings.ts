import { queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { getSettings, settingsQueryIdentity } from '@shared/control/resources/settings'

export * from '@shared/control/resources/settings'

export function settingsQueryOptions(client: ApiClient, locale: MaybeRefOrGetter<string>) {
  return queryOptions({
    queryKey: computed(() => settingsQueryIdentity(toValue(locale))),
    queryFn: ({ signal }) => getSettings(client, signal),
    gcTime: 0,
  })
}
