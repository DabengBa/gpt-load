export type ToastTone = 'info' | 'success' | 'warning' | 'danger'

export interface ToastMessage {
  id: number
  message: string
  tone: ToastTone
}

export interface ToastInput {
  message: string
  tone?: ToastTone
  duration?: number
}

export interface ToastController {
  getCurrent(): ToastMessage | null
  subscribe(listener: () => void): () => void
  show(input: ToastInput): void
  dismiss(): void
  dispose(): void
}

interface ToastControllerDependencies {
  setTimer(callback: () => void, duration: number): number
  clearTimer(timer: number): void
}

export function createToastController(deps: ToastControllerDependencies): ToastController {
  let current: ToastMessage | null = null
  let sequence = 0
  let timer: number | undefined
  const listeners = new Set<() => void>()
  const notify = () => {
    for (const listener of listeners) listener()
  }

  function dismiss(): void {
    if (timer !== undefined) deps.clearTimer(timer)
    timer = undefined
    current = null
    notify()
  }

  return {
    getCurrent: () => current,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    show(input) {
      if (timer !== undefined) deps.clearTimer(timer)
      const id = ++sequence
      current = {
        id,
        message: input.message,
        tone: input.tone ?? 'success',
      }
      notify()
      timer = deps.setTimer(() => {
        if (current?.id === id) {
          current = null
          notify()
        }
        timer = undefined
      }, input.duration ?? 2_000)
    },
    dismiss,
    dispose() {
      dismiss()
      listeners.clear()
    },
  }
}
