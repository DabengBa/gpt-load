export interface DebouncedAction {
  schedule(action: () => void): void
  cancel(): void
}

// Framework-neutral trailing-edge debounce; the per-framework hooks
// (useDebouncedAction in classic/astryx) add lifecycle disposal.
export function createDebouncedAction(delayMs: number): DebouncedAction {
  let timer: ReturnType<typeof setTimeout> | undefined

  function cancel(): void {
    if (timer !== undefined) clearTimeout(timer)
    timer = undefined
  }

  return {
    cancel,
    schedule(action) {
      cancel()
      timer = setTimeout(() => {
        timer = undefined
        action()
      }, delayMs)
    },
  }
}
