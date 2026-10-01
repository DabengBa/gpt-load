import { useCallback, useSyncExternalStore } from 'react'

/**
 * Live `matchMedia` binding. Needed where a JS-conditional value must stand in
 * for a CSS `@media` query — e.g. component `xstyle` props, whose narrowed
 * type surface rejects conditional values outright.
 */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    useCallback(
      (listener) => {
        const list = window.matchMedia(query)
        list.addEventListener('change', listener)
        return () => list.removeEventListener('change', listener)
      },
      [query],
    ),
    () => window.matchMedia(query).matches,
    () => false,
  )
}
