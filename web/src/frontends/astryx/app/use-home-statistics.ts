import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'

import type { HomeRange } from '@shared/control/resources/home'
import {
  createHomeStatisticsController,
  type HomeStatisticsSnapshot,
} from '@shared/controllers/home-statistics'
import { homeStatisticsQueryOptions } from '@shared/control/resources/home'

import { useAppServices } from './services'
import { useVisibleRefetch } from './use-visible-refetch'

export interface HomeStatisticsHandle {
  // Memoized controller state — read render-time values only from here (React
  // Compiler can freeze stable-reference `controller.*()` calls).
  snapshot: HomeStatisticsSnapshot
  refreshing: boolean
}

export function useHomeStatistics(options: { initialRange?: HomeRange }): HomeStatisticsHandle {
  const { apiClient, queryClient } = useAppServices()

  const [controller] = useState(() =>
    createHomeStatisticsController({
      queryClient,
      initialRange: options.initialRange,
    }),
  )

  const snapshot = useSyncExternalStore(
    useCallback((listener) => controller.subscribe(listener), [controller]),
    () => controller.getSnapshot(),
  )

  const statisticsQuery = useQuery(homeStatisticsQueryOptions(apiClient, snapshot.requestedRange))
  const statisticsRefetch = statisticsQuery.refetch

  useEffect(() => {
    controller.setQueryOps({
      refetch: () => statisticsRefetch({ cancelRefetch: false }),
    })
  }, [controller, statisticsRefetch])

  // Mirrors the classic watch on [requestedRange, dataUpdatedAt,
  // errorUpdateCount, status] — post-commit reconcile is fine here: unlike the
  // drawer, this state machine never feeds a mounted-inside-transition render.
  useEffect(() => {
    controller.reconcile()
  }, [
    controller,
    snapshot.requestedRange,
    statisticsQuery.dataUpdatedAt,
    statisticsQuery.errorUpdateCount,
    statisticsQuery.status,
  ])

  const retry = useCallback(() => controller.retry(), [controller])
  useVisibleRefetch([retry])

  useEffect(() => () => controller.dispose(), [controller])

  return {
    snapshot,
    refreshing:
      statisticsQuery.isFetching &&
      snapshot.state.kind !== 'initial' &&
      snapshot.state.kind !== 'switching',
  }
}
