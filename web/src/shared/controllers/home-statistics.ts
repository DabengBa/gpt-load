import type { QueryClient } from '@tanstack/query-core'

import type {
  HomeRange,
  HomeStatisticsDto,
} from '@shared/control/resources/home'
import { createEmptyHomeStatistics } from '@shared/control/resources/home'
import { controlQueryKeys } from '@shared/control/query-keys'

// Framework-free core of the classic `useHomeStatisticsPresenter`: a small
// state machine reconciled against TanStack query states. The mutable
// controller publishes an immutable snapshot so compiler-enabled React
// consumers can subscribe through `useSyncExternalStore` and never call
// controller getters during render.

export type HomeStatisticsState =
  | { kind: 'initial'; requestedRange: HomeRange }
  | { kind: 'ready'; selectedRange: HomeRange; snapshot: HomeStatisticsDto }
  | {
      kind: 'switching'
      selectedRange: HomeRange
      targetRange: HomeRange
      snapshot: HomeStatisticsDto
    }
  | {
      kind: 'stale'
      selectedRange: HomeRange
      snapshot: HomeStatisticsDto
      error: unknown
    }

export interface HomeStatisticsSnapshot {
  state: HomeStatisticsState
  requestedRange: HomeRange
  selectedRange: HomeRange
  targetRange: HomeRange | null
  lastSuccessfulObservedAtMS: number | null
}

// Pushed by the host whenever the active statistics query handle changes;
// keeps the controller free of per-render getter closures. `isFetching` is
// read straight from the query state (fetchStatus), so only `refetch` needs
// an observer-bound op.
export interface HomeStatisticsQueryOps {
  refetch(): Promise<unknown>
}

interface HandledQueryUpdates {
  data: number
  error: number
}

export interface HomeStatisticsControllerOptions {
  queryClient: QueryClient
  initialRange?: HomeRange
  now?: () => number
}

export interface HomeStatisticsController {
  getSnapshot(): HomeStatisticsSnapshot
  subscribe(listener: () => void): () => void
  setQueryOps(ops: HomeStatisticsQueryOps): void
  reconcile(): void
  selectRange(range: HomeRange): void
  retry(): Promise<void>
  dispose(): void
}

export function beginHomeStatisticsRange(
  current: HomeStatisticsState,
  targetRange: HomeRange,
): HomeStatisticsState {
  switch (current.kind) {
    case 'initial':
      return current.requestedRange === targetRange
        ? current
        : { kind: 'initial', requestedRange: targetRange }
    case 'ready':
    case 'stale':
      return current.selectedRange === targetRange
        ? current
        : {
            kind: 'switching',
            selectedRange: current.selectedRange,
            targetRange,
            snapshot: current.snapshot,
          }
    case 'switching':
      if (current.targetRange === targetRange) return current
      if (current.selectedRange === targetRange) {
        return {
          kind: 'ready',
          selectedRange: current.selectedRange,
          snapshot: current.snapshot,
        }
      }
      return { ...current, targetRange }
  }
}

export function commitHomeStatisticsSnapshot(
  current: HomeStatisticsState,
  snapshot: HomeStatisticsDto,
): HomeStatisticsState {
  const expectedRange =
    current.kind === 'initial'
      ? current.requestedRange
      : current.kind === 'switching'
        ? current.targetRange
        : current.selectedRange
  if (snapshot.range !== expectedRange) return current
  return {
    kind: 'ready',
    selectedRange: expectedRange,
    snapshot,
  }
}

export function rejectHomeStatisticsSnapshot(
  current: HomeStatisticsState,
  error: unknown,
  observedAtMS: number = Date.now(),
): HomeStatisticsState {
  if (current.kind === 'initial') {
    return {
      kind: 'stale',
      selectedRange: current.requestedRange,
      snapshot: createEmptyHomeStatistics(current.requestedRange, observedAtMS),
      error,
    }
  }
  if (current.kind === 'switching') {
    return {
      kind: 'stale',
      selectedRange: current.selectedRange,
      snapshot: current.snapshot,
      error,
    }
  }
  return {
    kind: 'stale',
    selectedRange: current.selectedRange,
    snapshot: current.snapshot,
    error,
  }
}

function deriveSelectedRange(state: HomeStatisticsState): HomeRange {
  if (state.kind === 'initial') return state.requestedRange
  if (state.kind === 'switching') return state.targetRange
  return state.selectedRange
}

export function createHomeStatisticsController(
  options: HomeStatisticsControllerOptions,
): HomeStatisticsController {
  const { queryClient } = options
  const now = options.now ?? Date.now
  const initialRange = options.initialRange ?? '24h'

  let requestedRange: HomeRange = initialRange
  let state: HomeStatisticsState = { kind: 'initial', requestedRange: initialRange }
  let lastSuccessfulObservedAtMS: number | null = null
  let queryOps: HomeStatisticsQueryOps | null = null
  let manualRefetch: Promise<void> | null = null
  const handledUpdates = new Map<HomeRange, HandledQueryUpdates>()
  const listeners = new Set<() => void>()

  let snapshot: HomeStatisticsSnapshot = buildSnapshot()

  function buildSnapshot(): HomeStatisticsSnapshot {
    return {
      state,
      requestedRange,
      selectedRange: deriveSelectedRange(state),
      targetRange: state.kind === 'switching' ? state.targetRange : null,
      lastSuccessfulObservedAtMS,
    }
  }

  function notify(): void {
    snapshot = buildSnapshot()
    for (const listener of listeners) listener()
  }

  function reconcile(): void {
    const queryState = queryClient.getQueryState<HomeStatisticsDto>(
      controlQueryKeys.home.statistics(requestedRange),
    )
    if (!queryState) return
    const handled = handledUpdates.get(requestedRange) ?? { data: 0, error: 0 }
    if (
      handled.data === queryState.dataUpdateCount &&
      handled.error === queryState.errorUpdateCount
    ) {
      return
    }
    handledUpdates.set(requestedRange, {
      data: queryState.dataUpdateCount,
      error: queryState.errorUpdateCount,
    })

    if (queryState.status === 'success' && queryState.data !== undefined) {
      const next = commitHomeStatisticsSnapshot(state, queryState.data)
      if (next !== state) {
        state = next
        lastSuccessfulObservedAtMS = queryState.data.observed_at_ms
        notify()
      }
      return
    }
    if (queryState.status === 'error' && queryState.error !== null) {
      const next = rejectHomeStatisticsSnapshot(state, queryState.error, now())
      state = next
      if (next.kind === 'stale' && requestedRange !== next.selectedRange) {
        requestedRange = next.selectedRange
      }
      notify()
    }
  }

  function retry(): Promise<void> {
    if (manualRefetch) return manualRefetch
    const queryState = queryClient.getQueryState<HomeStatisticsDto>(
      controlQueryKeys.home.statistics(requestedRange),
    )
    if (queryState?.fetchStatus === 'fetching') return Promise.resolve()
    if (!queryOps) return Promise.resolve()
    manualRefetch = queryOps
      .refetch()
      .then(() => undefined)
      .finally(() => {
        manualRefetch = null
      })
    return manualRefetch
  }

  function selectRange(range: HomeRange): void {
    const currentRange = requestedRange
    const next = beginHomeStatisticsRange(state, range)
    if (next === state) {
      if (state.kind === 'stale' && state.selectedRange === range) {
        void retry()
      }
      return
    }
    state = next
    if (currentRange !== range) {
      requestedRange = range
      void queryClient.cancelQueries({
        queryKey: controlQueryKeys.home.statistics(currentRange),
        exact: true,
      })
    }
    notify()
  }

  return {
    getSnapshot: () => snapshot,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    setQueryOps(ops) {
      queryOps = ops
    },
    reconcile,
    selectRange,
    retry,
    dispose() {
      listeners.clear()
      queryOps = null
    },
  }
}
