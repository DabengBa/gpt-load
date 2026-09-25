import { inject, readonly, ref, type InjectionKey, type Ref } from 'vue'

import {
  createToastController as createSharedToastController,
  type ToastInput,
  type ToastMessage,
} from '@shared/controllers/toast'

export type { ToastInput, ToastMessage, ToastTone } from '@shared/controllers/toast'

export interface ToastController {
  readonly current: Readonly<Ref<ToastMessage | null>>
  show(input: ToastInput): void
  dismiss(): void
  dispose(): void
}

export function createToastController(deps: {
  setTimer(callback: () => void, duration: number): number
  clearTimer(timer: number): void
}): ToastController {
  const core = createSharedToastController(deps)
  const current = ref<ToastMessage | null>(core.getCurrent())
  const unsubscribe = core.subscribe(() => {
    current.value = core.getCurrent()
  })

  return {
    current: readonly(current),
    show(input) {
      core.show(input)
    },
    dismiss() {
      core.dismiss()
    },
    dispose() {
      unsubscribe()
      core.dispose()
    },
  }
}

export const toastKey: InjectionKey<ToastController> = Symbol('toast')

export function useToast(): ToastController {
  const controller = inject(toastKey)
  if (!controller) throw new Error('TOAST_NOT_PROVIDED')
  return controller
}
