export type AppTheme = 'system' | 'light' | 'dark'

export interface ThemeController {
  getTheme(): AppTheme
  subscribe(listener: () => void): () => void
  setTheme(theme: AppTheme): void
  dispose(): void
}

interface ThemeControllerDependencies {
  documentElement: HTMLElement
  storage?: Storage
  matchMedia(query: string): MediaQueryList
}

type BrowserThemeWindow = Pick<Window, 'localStorage'> & Partial<Pick<Window, 'matchMedia'>>

export const themeStorageKey = 'gpt-load.theme'

export function isTheme(value: unknown): value is AppTheme {
  return value === 'system' || value === 'light' || value === 'dark'
}

function createThemeController(deps: ThemeControllerDependencies): ThemeController {
  let theme: AppTheme = 'system'
  try {
    const stored = deps.storage?.getItem(themeStorageKey)
    if (isTheme(stored)) theme = stored
  } catch {
    // Browser preferences remain available in memory when storage is denied.
  }

  const listeners = new Set<() => void>()
  const notify = () => {
    for (const listener of listeners) listener()
  }
  let media: MediaQueryList | undefined
  let mediaListener: ((event: MediaQueryListEvent) => void) | undefined

  function apply(next: AppTheme): void {
    if (next === 'system') {
      deps.documentElement.removeAttribute('data-theme')
    } else {
      deps.documentElement.dataset.theme = next
    }
  }

  try {
    media = deps.matchMedia('(prefers-color-scheme: dark)')
    mediaListener = () => {
      if (theme === 'system') apply('system')
    }
    media.addEventListener('change', mediaListener)
  } catch {
    media = undefined
    mediaListener = undefined
  }

  apply(theme)

  return {
    getTheme: () => theme,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    setTheme(next) {
      theme = next
      apply(next)
      try {
        deps.storage?.setItem(themeStorageKey, next)
      } catch {
        // Persistence failure does not change the active in-memory preference.
      }
      notify()
    },
    dispose() {
      if (media && mediaListener) media.removeEventListener('change', mediaListener)
      listeners.clear()
    },
  }
}

export function createBrowserThemeController(
  browser: BrowserThemeWindow,
  documentElement: HTMLElement,
  storage?: Storage,
): ThemeController {
  return createThemeController({
    documentElement,
    storage,
    matchMedia:
      typeof browser.matchMedia === 'function'
        ? browser.matchMedia.bind(browser)
        : () => {
            throw new DOMException('matchMedia unavailable')
          },
  })
}
