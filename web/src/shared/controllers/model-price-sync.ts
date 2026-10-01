import type { QueryClient } from '@tanstack/query-core'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { syncModelPrices } from '@shared/control/resources/providers'
import type { ApiClient } from '@shared/http/client'
import { RequestCancelledError } from '@shared/http/errors'

import { createTransientFlag, type TransientFlagController } from './transient-flag'

export interface ModelPriceSyncControllerOptions {
  client: ApiClient
  queryClient: QueryClient
  succeededDurationMs?: number
}

export interface ModelPriceSyncSnapshot {
  pending: boolean
  failed: boolean
  succeeded: boolean
}

export interface ModelPriceSyncController {
  isPending(): boolean
  hasFailed(): boolean
  hasSucceeded(): boolean
  /** Memoized store snapshot for useSyncExternalStore (see model-price-editor). */
  getSnapshot(): ModelPriceSyncSnapshot
  subscribe(listener: () => void): () => void
  run(): Promise<void>
  dispose(): void
}

export function createModelPriceSyncController(
  options: ModelPriceSyncControllerOptions,
): ModelPriceSyncController {
  let pending = false
  let failed = false
  let mounted = true
  let requestController: AbortController | undefined
  const succeededFlag: TransientFlagController = createTransientFlag(
    options.succeededDurationMs ?? 2_000,
  )
  const listeners = new Set<() => void>()
  let snapshot = { pending, failed, succeeded: succeededFlag.getValue() }
  const notify = () => {
    snapshot = { pending, failed, succeeded: succeededFlag.getValue() }
    for (const listener of listeners) listener()
  }
  const unsubscribeFlag = succeededFlag.subscribe(notify)

  function isCurrent(controller: AbortController): boolean {
    return mounted && requestController === controller && !controller.signal.aborted
  }

  async function run(): Promise<void> {
    if (pending) return
    pending = true
    failed = false
    notify()
    requestController?.abort()
    const controller = new AbortController()
    requestController = controller
    try {
      await syncModelPrices(options.client, controller.signal)
      if (!isCurrent(controller)) return
      await applyInvalidationPlan(
        options.queryClient,
        mutationInvalidationPlans.modelPrice.sync,
        () => isCurrent(controller),
      )
      if (!isCurrent(controller)) return
      succeededFlag.show()
    } catch (error: unknown) {
      if (isCurrent(controller) && !(error instanceof RequestCancelledError)) {
        failed = true
        notify()
      }
    } finally {
      if (isCurrent(controller)) {
        requestController = undefined
        pending = false
        notify()
      }
    }
  }

  return {
    isPending: () => pending,
    hasFailed: () => failed,
    hasSucceeded: () => succeededFlag.getValue(),
    getSnapshot: () => snapshot,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    run,
    dispose() {
      mounted = false
      requestController?.abort()
      requestController = undefined
      unsubscribeFlag()
      succeededFlag.clear()
      listeners.clear()
    },
  }
}
