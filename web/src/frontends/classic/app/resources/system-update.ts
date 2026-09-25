import { queryOptions } from '@tanstack/vue-query'
import type { MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@shared/http/client'
import { controlQueryKeys } from '@shared/control/query-keys'
import { getSystemUpdate } from '@shared/control/resources/system-update'

export * from '@shared/control/resources/system-update'

export function systemUpdateQueryOptions(client: ApiClient, enabled?: MaybeRefOrGetter<boolean>) {
  return queryOptions({
    queryKey: controlQueryKeys.systemUpdate(),
    queryFn: ({ signal }) => getSystemUpdate(client, signal),
    retry: false,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnMount: 'always',
    // 失败保持静默；只有重新进入首页才再次调用按需检查接口。
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    ...(enabled === undefined ? {} : { enabled }),
  })
}
