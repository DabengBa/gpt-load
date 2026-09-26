import { computed, onScopeDispose, ref, type ComputedRef } from 'vue'

import { createTransientFlag } from '@shared/controllers/transient-flag'

export interface TransientFlag {
  value: ComputedRef<boolean>
  show(): void
  clear(): void
}

export function useTransientFlag(durationMs: number): TransientFlag {
  const controller = createTransientFlag(durationMs)
  const value = ref(controller.getValue())
  const unsubscribe = controller.subscribe(() => {
    value.value = controller.getValue()
  })

  onScopeDispose(() => {
    unsubscribe()
    controller.clear()
  })

  return { value: computed(() => value.value), show: controller.show, clear: controller.clear }
}
