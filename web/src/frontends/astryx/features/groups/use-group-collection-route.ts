import { useNavigate, useRouterState } from '@tanstack/react-router'
import { useCallback, useEffect, useMemo } from 'react'

import type { GroupCollectionFilters } from '@shared/control/types'
import {
  isCanonicalGroupCollectionRouteQuery,
  parseGroupCollectionRouteQuery,
  serializeGroupCollectionRouteQuery,
} from '@shared/routing/group-collection-route'
import { pagePath } from '@shared/routing/page-routes'
import type { SharedRouteQuery } from '@shared/routing/route-query'

/**
 * The URL is the single source of truth for the collection's filters: this
 * hook parses it and hands back one navigate helper, so no caller writes
 * search params by hand.
 */
export function useGroupCollectionRoute() {
  const navigate = useNavigate()
  const { rawSearch, searchStr, pathname } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
      pathname: state.location.pathname,
    }),
  })
  const filters = useMemo(() => parseGroupCollectionRouteQuery(rawSearch), [rawSearch])
  const groupsPath = pagePath('groups')

  const go = useCallback(
    (next: GroupCollectionFilters, replace: boolean): void => {
      void navigate({
        to: groupsPath,
        search: serializeGroupCollectionRouteQuery(next),
        replace,
        resetScroll: false,
      })
    },
    [navigate, groupsPath],
  )

  // Canonicalize non-canonical query params — mirrors the classic router
  // watch: junk/duplicated keys are dropped via a history replace.
  useEffect(() => {
    // During a pending transition the outgoing route keeps rendering while
    // location already moved — a late canonicalization must not resurrect it.
    if (pathname !== groupsPath) return
    if (!isCanonicalGroupCollectionRouteQuery(rawSearch, filters)) go(filters, true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the URL string
  }, [searchStr])

  return { filters, searchStr, pathname, groupsPath, go }
}
