export interface UnsavedChangesController {
  getDialogOpen(): boolean
  subscribe(listener: () => void): () => void
  bypassNext(): void
  consumeBypass(): boolean
  requestConfirmation(): Promise<boolean>
  resolveConfirmation(confirmed: boolean): void
}

export function createUnsavedChangesController(): UnsavedChangesController {
  let bypass = false
  let dialogOpen = false
  let resolvePending: ((confirmed: boolean) => void) | undefined
  const listeners = new Set<() => void>()
  const notify = () => {
    for (const listener of listeners) listener()
  }

  return {
    getDialogOpen: () => dialogOpen,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    bypassNext() {
      resolvePending?.(false)
      resolvePending = undefined
      dialogOpen = false
      notify()
      bypass = true
    },
    consumeBypass() {
      const result = bypass
      bypass = false
      return result
    },
    requestConfirmation() {
      if (resolvePending) return Promise.resolve(false)
      dialogOpen = true
      notify()
      return new Promise<boolean>((resolve) => {
        resolvePending = resolve
      })
    },
    resolveConfirmation(confirmed) {
      const resolve = resolvePending
      resolvePending = undefined
      dialogOpen = false
      notify()
      resolve?.(confirmed)
    },
  }
}
