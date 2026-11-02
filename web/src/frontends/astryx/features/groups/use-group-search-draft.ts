import { useEffect, useRef, useState } from 'react'

import type { GroupCollectionFilters } from '@shared/control/types'
import { constrainGroupCollectionSearchQuery } from '@shared/routing/group-collection-route'

import { useDebouncedAction } from '../../app/use-debounced-action'

/**
 * The search box is a draft over the URL: typing is debounced into `?q=`, an
 * explicit clear applies at once, and a URL change (back/forward, reset)
 * rewrites the draft instead of racing it.
 */
export function useGroupSearchDraft({
  filters,
  searchStr,
}: {
  filters: GroupCollectionFilters
  searchStr: string
}) {
  const [searchDraft, setSearchDraft] = useState(filters.q ?? '')
  // The debounced callback must read the draft at fire time, not at schedule
  // time — a ref mirrors the Vue `searchDraft.value` read.
  const searchDraftRef = useRef(searchDraft)
  useEffect(() => {
    searchDraftRef.current = searchDraft
  }, [searchDraft])
  const debounce = useDebouncedAction(250)

  // Route → draft sync via the render-adjustment pattern (set-state during
  // render is the sanctioned alternative to an effect that only mirrors
  // external state back into local state).
  const [lastSearchStr, setLastSearchStr] = useState(searchStr)
  if (lastSearchStr !== searchStr) {
    setLastSearchStr(searchStr)
    debounce.cancel()
    setSearchDraft(filters.q ?? '')
  }

  function onSearchChange(value: string, apply: (q: string | undefined) => void): void {
    debounce.cancel()
    setSearchDraft(value)
    // The clear affordance emits '' through onChange; applying it immediately
    // keeps the classic `@clear` contract (no debounce on explicit clear).
    if (value === '') {
      apply(undefined)
      return
    }
    debounce.schedule(() => apply(constrainGroupCollectionSearchQuery(searchDraftRef.current)))
  }

  function resetDraft(): void {
    debounce.cancel()
    setSearchDraft('')
  }

  return { searchDraft, searchDraftRef, onSearchChange, resetDraft }
}
