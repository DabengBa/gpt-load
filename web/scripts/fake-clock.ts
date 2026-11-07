type Timer = { at: number; every?: number; callback: () => void }

/** Settle microtasks and already-resolved promises without touching the virtual clock. */
export async function flush(): Promise<void> {
  for (let index = 0; index < 8; index++) await new Promise((resolve) => setImmediate(resolve))
}

/**
 * A virtual clock so a kept-open page can be advanced through real polling
 * intervals without sleeping. `setInterval` delays are recorded so a test can
 * assert which cadence the page settled on.
 */
export class FakeClock {
  now = 0
  intervals: number[] = []
  private nextID = 1
  private timers = new Map<number, Timer>()

  readonly provider = {
    setTimeout: (callback: () => void, delay: number) => this.add(callback, delay),
    clearTimeout: (id: number | undefined) => this.clear(id),
    setInterval: (callback: () => void, delay: number) => {
      this.intervals.push(delay)
      return this.add(callback, delay, delay)
    },
    clearInterval: (id: number | undefined) => this.clear(id),
  }

  private add(callback: () => void, delay: number, every?: number): number {
    const id = this.nextID++
    this.timers.set(id, { at: this.now + Math.max(delay, 0), every, callback })
    return id
  }

  private clear(id: number | undefined): void {
    if (id !== undefined) this.timers.delete(Number(id))
  }

  async advance(ms: number): Promise<void> {
    await flush()
    const target = this.now + ms
    for (;;) {
      const due = [...this.timers.entries()]
        .filter(([, timer]) => timer.at <= target)
        .sort(([, left], [, right]) => left.at - right.at)[0]
      if (due === undefined) break
      const [id, timer] = due
      this.now = timer.at
      if (timer.every === undefined) this.timers.delete(id)
      else timer.at = this.now + timer.every
      timer.callback()
      await flush()
    }
    this.now = target
  }
}
