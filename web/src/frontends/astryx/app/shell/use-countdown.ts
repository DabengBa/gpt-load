import { useCallback, useEffect, useRef, useState } from 'react'

import { normalizeSeconds } from '@shared/lib/time'

export interface Countdown {
  seconds: number
  active: boolean
  reset(value: number): void
}

// React port of the classic use-countdown composable: one tick per second to
// zero, `reset` restarts, and a source-seconds change reseeds the timer.
export function useCountdown(sourceSeconds: number): Countdown {
  const [seconds, setSeconds] = useState(() => normalizeSeconds(sourceSeconds))
  const lastSource = useRef(sourceSeconds)

  const reset = useCallback((value: number) => {
    setSeconds(normalizeSeconds(value))
  }, [])

  useEffect(() => {
    if (lastSource.current === sourceSeconds) return
    lastSource.current = sourceSeconds
    setSeconds(normalizeSeconds(sourceSeconds))
  }, [sourceSeconds])

  useEffect(() => {
    if (seconds <= 0) return
    const timer = setTimeout(() => setSeconds((current) => Math.max(0, current - 1)), 1_000)
    return () => clearTimeout(timer)
  }, [seconds])

  return { seconds, active: seconds > 0, reset }
}
