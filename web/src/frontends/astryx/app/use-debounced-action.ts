import { useEffect, useMemo } from 'react'

import {
  createDebouncedAction,
  type DebouncedAction,
} from '@shared/lib/debounced-action'

export type { DebouncedAction }

export function useDebouncedAction(delayMs: number): DebouncedAction {
  const action = useMemo(() => createDebouncedAction(delayMs), [delayMs])
  useEffect(() => action.cancel, [action])
  return action
}
