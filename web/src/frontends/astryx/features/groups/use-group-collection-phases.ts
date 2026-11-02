import type { UseQueryResult } from '@tanstack/react-query'

import type { GroupCollectionResponseDto } from '@shared/control/types'

import { useCollectionLoading } from '../../app/collection-loading'

/**
 * Loading is a three-phase contract, not a single boolean: cold load, a
 * placeholder transition that keeps the stale table and sizes the skeleton to
 * the incoming row count, and a silent background refresh.
 */
export function useGroupCollectionPhases(groupsQuery: UseQueryResult<GroupCollectionResponseDto>) {
  const data = groupsQuery.data
  const {
    initial: initialLoading,
    transition: collectionTransition,
    refreshing: collectionRefreshing,
    rows: skeletonRows,
  } = useCollectionLoading(
    {
      pending: groupsQuery.isPending,
      placeholder: groupsQuery.isPlaceholderData,
      fetching: groupsQuery.isFetching,
      hasData: data !== undefined,
      itemCount: data?.items.length ?? 0,
    },
    { fallbackRows: 20 },
  )

  return {
    collectionBusy: data !== undefined && groupsQuery.isFetching,
    initialLoading,
    collectionTransition,
    collectionRefreshing,
    skeletonRows,
  }
}
