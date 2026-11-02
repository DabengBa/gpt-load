import { useQuery } from '@tanstack/react-query'
import { useRouterState } from '@tanstack/react-router'
import { useEffect } from 'react'

import { groupCollectionQueryOptions } from '@shared/control/resources/groups'

import { useAppServices } from '../../app/services'
import { useVisibleRefetch } from '../../app/use-visible-refetch'
import { useGroupChannels } from './use-group-channels'
import { useGroupFilterRouting } from './use-group-filter-routing'
import { useGroupCollectionPhases } from './use-group-collection-phases'
import { useGroupDetailPrefetch } from './use-group-detail-prefetch'
import { useGroupRowMutations } from './use-group-row-mutations'
import { useGroupSummaryItems } from './use-group-summary-items'

/**
 * Collection data + controller composition: server read model, derived loading
 * phases, filter routing and row mutations. The view below only renders.
 */
export function useGroupCollection() {
  const { apiClient } = useAppServices()
  const { searchStr } = useRouterState({ select: (state) => ({ searchStr: state.location.searchStr }) })
  const {
    filters,
    searchDraft,
    onSearchChange,
    setStatus,
    setSort,
    setPage,
    resetConditions,
    correctPage,
  } = useGroupFilterRouting()

  const groupsQuery = useQuery(groupCollectionQueryOptions(apiClient, filters))
  const channelsByID = useGroupChannels()

  const data = groupsQuery.data
  const totalPages = data?.pagination.total_pages
  const isPlaceholder = groupsQuery.isPlaceholderData
  useEffect(() => {
    correctPage(totalPages, isPlaceholder)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on server pagination
  }, [totalPages, filters.page, isPlaceholder, searchStr])

  useVisibleRefetch([groupsQuery.refetch])

  const hasFilterCriteria = filters.q !== undefined || filters.status !== undefined
  const hasChangedConditions = hasFilterCriteria || filters.sort !== 'recent'
  const phases = useGroupCollectionPhases(groupsQuery)
  const summaryItems = useGroupSummaryItems(data?.summary)
  const prefetchGroupSummary = useGroupDetailPrefetch()
  const rowMutations = useGroupRowMutations()

  return {
    filters,
    searchDraft,
    data,
    groupsQuery,
    channelsByID,
    ...rowMutations,
    ...phases,
    hasFilterCriteria,
    hasChangedConditions,
    summaryItems,
    prefetchGroupSummary,
    onSearchChange,
    setStatus,
    setSort,
    setPage,
    resetConditions,
  }
}
