import { useEffect, useMemo, useRef } from 'react'

export interface DebouncedAction {
  schedule(action: () => void): void
  cancel(): void
}

export function useDebouncedAction(delayMs: number): DebouncedAction {
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const action = useMemo<DebouncedAction>(() => {
    function cancel(): void {
      if (timerRef.current !== undefined) clearTimeout(timerRef.current)
      timerRef.current = undefined
    }
    return {
      cancel,
      schedule(callback) {
        cancel()
        timerRef.current = setTimeout(() => {
          timerRef.current = undefined
          callback()
        }, delayMs)
      },
    }
  }, [delayMs])

  useEffect(() => action.cancel, [action])
  return action
}
