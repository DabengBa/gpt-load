import { useEffect, useRef, useState } from 'react'

import { loadingTimings } from '@shared/lib/loading-timings'

export { loadingTimings }

export function useStableLoading(
  active: boolean,
  options: { delayMs?: number; minimumVisibleMs?: number } = {},
): boolean {
  const delayMs = options.delayMs ?? loadingTimings.delayMs
  const minimumVisibleMs = options.minimumVisibleMs ?? loadingTimings.minimumVisibleMs
  const [visible, setVisible] = useState(false)
  const stateRef = useRef({
    shownAt: 0,
    showTimer: undefined as ReturnType<typeof setTimeout> | undefined,
    hideTimer: undefined as ReturnType<typeof setTimeout> | undefined,
  })

  useEffect(() => {
    const state = stateRef.current

    function clearShowTimer(): void {
      if (state.showTimer === undefined) return
      clearTimeout(state.showTimer)
      state.showTimer = undefined
    }

    function clearHideTimer(): void {
      if (state.hideTimer === undefined) return
      clearTimeout(state.hideTimer)
      state.hideTimer = undefined
    }

    function hide(): void {
      state.hideTimer = undefined
      setVisible(false)
      state.shownAt = 0
    }

    if (active) {
      clearHideTimer()
      if (!visible && state.showTimer === undefined) {
        state.showTimer = setTimeout(() => {
          state.showTimer = undefined
          state.shownAt = Date.now()
          setVisible(true)
        }, delayMs)
      }
    } else {
      clearShowTimer()
      if (visible && state.hideTimer === undefined) {
        const remaining = Math.max(0, minimumVisibleMs - (Date.now() - state.shownAt))
        if (remaining === 0) hide()
        else state.hideTimer = setTimeout(hide, remaining)
      }
    }

    return () => {
      clearShowTimer()
      clearHideTimer()
    }
  }, [active, visible, delayMs, minimumVisibleMs])

  return visible
}

export interface CollectionLoadingInput {
  pending: boolean
  placeholder: boolean
  fetching: boolean
  hasData: boolean
  itemCount: number
}

export function useCollectionLoading(
  input: CollectionLoadingInput,
  options: { fallbackRows?: number; maximumRows?: number } = {},
) {
  const fallbackRows = options.fallbackRows ?? 5
  const maximumRows = options.maximumRows ?? 100
  const transitionActive = input.placeholder && input.hasData
  const [rows, setRows] = useState(fallbackRows)
  const wasActiveRef = useRef(false)

  useEffect(() => {
    if (!transitionActive || wasActiveRef.current) {
      wasActiveRef.current = transitionActive
      return
    }
    wasActiveRef.current = true
    const current = Math.trunc(input.itemCount)
    setRows(Math.min(maximumRows, Math.max(1, current || fallbackRows)))
  }, [transitionActive, input.itemCount, fallbackRows, maximumRows])

  return {
    initial: useStableLoading(input.pending),
    transition: useStableLoading(transitionActive),
    refreshing: input.hasData && input.fetching && !input.placeholder,
    rows,
  }
}
