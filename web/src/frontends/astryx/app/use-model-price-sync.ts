import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'

import {
  createModelPriceSyncController,
  type ModelPriceSyncController,
  type ModelPriceSyncSnapshot,
} from '@shared/controllers/model-price-sync'

import { useAppServices } from './services'

export interface ModelPriceSyncHandle {
  controller: ModelPriceSyncController
  /** Memoized controller state — always read render-time values from here. */
  snapshot: ModelPriceSyncSnapshot
}

export function useModelPriceSync(): ModelPriceSyncHandle {
  const { apiClient, queryClient } = useAppServices()

  const [controller] = useState(() =>
    createModelPriceSyncController({ client: apiClient, queryClient }),
  )

  const snapshot = useSyncExternalStore(
    useCallback((listener) => controller.subscribe(listener), [controller]),
    () => controller.getSnapshot(),
  )

  useEffect(() => () => controller.dispose(), [controller])

  return { controller, snapshot }
}
