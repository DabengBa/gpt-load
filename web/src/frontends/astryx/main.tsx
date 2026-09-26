import { Theme } from '@astryxdesign/core/theme'
import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { createRoot } from 'react-dom/client'

import './entry.css'
import { AppI18nProviders, createAppI18n } from './app/i18n'
import { createAppRouter, type AppRouter } from './app/router'
import {
  AppServicesProvider,
  createAppServices,
  type AppServices,
} from './app/services'
import { gptloadTheme } from './theme/gptload'
import { useThemePreference } from './theme/theme-preference'

// Boot order mirrors classic main.ts: resolve the locale and load the core
// catalog before first render, then create services (the api client reads the
// locale from the controller) and the router.
const i18n = await createAppI18n()

// The services layer is created before the router; the ref lets the global
// unauthorized handler navigate once the router exists.
const routerRef: { current?: AppRouter } = {}

const services: AppServices = createAppServices({
  i18n,
  onSessionCleared: (currentHref) => {
    void routerRef.current?.navigate({
      href: `/login?redirect=${encodeURIComponent(currentHref)}`,
      replace: true,
    })
  },
})

const appRouter = createAppRouter(services)
routerRef.current = appRouter

function App() {
  const [mode] = useThemePreference()
  return (
    <Theme theme={gptloadTheme} mode={mode}>
      <AppI18nProviders i18n={i18n}>
        <QueryClientProvider client={services.queryClient}>
          <AppServicesProvider value={services}>
            <RouterProvider router={appRouter} />
          </AppServicesProvider>
        </QueryClientProvider>
      </AppI18nProviders>
    </Theme>
  )
}

const host = document.getElementById('app')
if (!host) throw new Error('missing #app mount point')
createRoot(host).render(<App />)
