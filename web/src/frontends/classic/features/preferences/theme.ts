import { inject, readonly, ref, type InjectionKey, type Ref } from 'vue'

import {
  createBrowserThemeController as createSharedBrowserThemeController,
  type AppTheme,
} from '@shared/controllers/theme'

export type { AppTheme }

export interface ThemeController {
  readonly theme: Readonly<Ref<AppTheme>>
  setTheme(theme: AppTheme): void
  dispose(): void
}

export function createBrowserThemeController(
  browser: Pick<Window, 'localStorage'> & Partial<Pick<Window, 'matchMedia'>>,
  documentElement: HTMLElement,
  storage?: Storage,
): ThemeController {
  const core = createSharedBrowserThemeController(browser, documentElement, storage)
  const theme = ref<AppTheme>(core.getTheme())
  const unsubscribe = core.subscribe(() => {
    theme.value = core.getTheme()
  })

  return {
    theme: readonly(theme),
    setTheme(next) {
      core.setTheme(next)
    },
    dispose() {
      unsubscribe()
      core.dispose()
    },
  }
}

export const themeControllerKey: InjectionKey<ThemeController> = Symbol('theme-controller')

export function useTheme(): ThemeController {
  const controller = inject(themeControllerKey)
  if (!controller) throw new Error('THEME_CONTROLLER_NOT_PROVIDED')
  return controller
}
