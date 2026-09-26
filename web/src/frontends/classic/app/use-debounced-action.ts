import { onScopeDispose } from 'vue'

import {
  createDebouncedAction,
  type DebouncedAction,
} from '@shared/lib/debounced-action'

export type { DebouncedAction }

export function useDebouncedAction(delayMs: number): DebouncedAction {
  const action = createDebouncedAction(delayMs)
  onScopeDispose(action.cancel)
  return action
}
