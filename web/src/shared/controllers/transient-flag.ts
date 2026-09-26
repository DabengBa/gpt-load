export interface TransientFlagController {
  getValue(): boolean
  subscribe(listener: () => void): () => void
  show(): void
  clear(): void
}

export function createTransientFlag(durationMs: number): TransientFlagController {
  let value = false
  let timer: ReturnType<typeof setTimeout> | undefined
  const listeners = new Set<() => void>()
  const notify = () => {
    for (const listener of listeners) listener()
  }

  function clear(): void {
    if (timer !== undefined) clearTimeout(timer)
    timer = undefined
    if (!value) return
    value = false
    notify()
  }

  function show(): void {
    if (timer !== undefined) clearTimeout(timer)
    timer = setTimeout(clear, durationMs)
    if (value) return
    value = true
    notify()
  }

  return {
    getValue: () => value,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    show,
    clear,
  }
}
