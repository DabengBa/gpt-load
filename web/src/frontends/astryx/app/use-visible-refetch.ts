import { useEffect, useRef } from 'react'

// Mirrors the classic composable: refetch everything when the tab returns
// from hidden. Server-driven collections own freshness via manual refetch,
// so visibility is the only automatic trigger.
export function useVisibleRefetch(
  refetchers: ReadonlyArray<() => unknown | Promise<unknown>>,
): void {
  const refetchersRef = useRef(refetchers)

  useEffect(() => {
    refetchersRef.current = refetchers
  }, [refetchers])

  useEffect(() => {
    let wasHidden = document.hidden
    function handleVisibilityChange(): void {
      const hidden = document.hidden
      if (wasHidden && !hidden) {
        void Promise.allSettled(
          refetchersRef.current.map((refetch) => Promise.resolve().then(() => refetch())),
        )
      }
      wasHidden = hidden
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [])
}
