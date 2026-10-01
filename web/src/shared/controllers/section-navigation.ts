export interface SectionNavigationControllerOptions {
  ids: readonly string[]
  initialId?: string
  topOffset?: number
}

export interface SectionNavigationController {
  getActiveSection(): string
  subscribe(listener: () => void): () => void
  selectSection(id: string, behavior?: ScrollBehavior): void
  mount(): () => void
  /** Push the latest id list — scrollspy reads it on the next sync pass. */
  updateIds(next: readonly string[]): void
  notifyUpdated(): void
}

export function createSectionNavigationController({
  ids,
  initialId,
  topOffset = 88,
}: SectionNavigationControllerOptions): SectionNavigationController {
  let currentIds = ids
  let activeSection = initialId ?? currentIds[0] ?? ''
  let sectionFrame = 0
  const listeners = new Set<() => void>()

  function setActiveSection(id: string): void {
    if (id === activeSection) return
    activeSection = id
    for (const listener of listeners) listener()
  }

  function synchronizeSection(): void {
    sectionFrame = 0
    const sectionIDs = currentIds
    const elements = sectionIDs
      .map((id) => document.getElementById(id))
      .filter((element): element is HTMLElement => element !== null)
    if (!sectionIDs.length || elements.length !== sectionIDs.length) return

    let current = elements[0]
    for (const element of elements) {
      if (element.getBoundingClientRect().top <= topOffset) current = element
      else break
    }
    const pageHeight = document.documentElement.scrollHeight
    const pageBottom = window.scrollY + window.innerHeight
    if (pageHeight > window.innerHeight + 2 && pageBottom >= pageHeight - 2)
      current = elements.at(-1) ?? current
    setActiveSection(current.id)
  }

  function scheduleSynchronization(): void {
    if (sectionFrame) return
    sectionFrame = window.requestAnimationFrame(synchronizeSection)
  }

  function selectSection(id: string, behavior?: ScrollBehavior): void {
    setActiveSection(id)
    const resolved =
      behavior ??
      (window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth')
    document.getElementById(id)?.scrollIntoView({ behavior: resolved, block: 'start' })
  }

  function mount(): () => void {
    window.addEventListener('scroll', scheduleSynchronization, { passive: true })
    window.addEventListener('resize', scheduleSynchronization, { passive: true })
    scheduleSynchronization()
    return () => {
      window.removeEventListener('scroll', scheduleSynchronization)
      window.removeEventListener('resize', scheduleSynchronization)
      if (sectionFrame) {
        window.cancelAnimationFrame(sectionFrame)
        sectionFrame = 0
      }
    }
  }

  return {
    getActiveSection: () => activeSection,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    selectSection,
    mount,
    updateIds(next) {
      currentIds = next
    },
    notifyUpdated: scheduleSynchronization,
  }
}
