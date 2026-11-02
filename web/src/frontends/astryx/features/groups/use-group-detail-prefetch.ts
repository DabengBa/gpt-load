import { useCallback } from 'react'

import { groupSummaryQueryOptions } from '@shared/control/resources/groups'

import { useAppServices } from '../../app/services'

/**
 * Detail is prefetched on intent (hover/focus) so a table row never pays for a
 * second round trip after the click. Only admins can open a group, so only
 * admins get the speculative fetch.
 */
export function useGroupDetailPrefetch(): (groupID: number) => void {
  const { apiClient, queryClient, authSession } = useAppServices()
  return useCallback(
    (groupID: number): void => {
      const session = authSession.getState()
      if (groupID > 0 && session.phase === 'validated' && session.principalType === 'admin') {
        void queryClient.prefetchQuery(groupSummaryQueryOptions(apiClient, groupID))
      }
    },
    [apiClient, authSession, queryClient],
  )
}
