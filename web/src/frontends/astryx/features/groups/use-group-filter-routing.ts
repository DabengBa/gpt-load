import type {
  GroupCollectionFilters,
  GroupCollectionSort,
  GroupCollectionStatus,
} from '@shared/control/types'
import { constrainGroupCollectionSearchQuery } from '@shared/routing/group-collection-route'

import { useGroupCollectionRoute } from './use-group-collection-route'
import { useGroupSearchDraft } from './use-group-search-draft'

/**
 * URL ↔ filter round trip for the group collection: the parsed filters, the
 * debounced search draft, and the accepted mutations of both. The view never
 * writes route state directly.
 */
export function useGroupFilterRouting() {
  const { filters, searchStr, pathname, groupsPath, go } = useGroupCollectionRoute()
  const { searchDraft, searchDraftRef, onSearchChange, resetDraft } = useGroupSearchDraft({
    filters,
    searchStr,
  })

  function updateConditions(
    patch: Partial<Pick<GroupCollectionFilters, 'q' | 'status' | 'sort'>>,
  ): void {
    const q = constrainGroupCollectionSearchQuery(searchDraftRef.current)
    go({ ...filters, q, ...patch, page: 1 }, false)
  }

  function setStatus(status: GroupCollectionStatus | undefined): void {
    updateConditions({ status })
  }

  function setSort(value: string): void {
    updateConditions({ sort: value as GroupCollectionSort })
  }

  function resetConditions(): void {
    resetDraft()
    go({ sort: 'recent', page: 1, page_size: 100 }, false)
  }

  function setPage(page: number): void {
    go({ ...filters, page }, false)
  }

  // Page correction: when the server clamps a deep page, mirror the
  // correction into the URL (classic parity).
  function correctPage(totalPages: number | undefined, isPlaceholder: boolean): void {
    if (pathname !== groupsPath) return
    if (isPlaceholder || totalPages === undefined || totalPages <= 0) return
    if (filters.page <= totalPages) return
    go({ ...filters, page: totalPages }, true)
  }

  return {
    filters,
    searchDraft,
    onSearchChange: (value: string) => onSearchChange(value, (q) => updateConditions({ q })),
    setStatus,
    setSort,
    setPage,
    resetConditions,
    correctPage,
  }
}
