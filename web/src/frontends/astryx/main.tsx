import { Theme } from '@astryxdesign/core/theme'
import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { createRoot } from 'react-dom/client'

import './entry.css'
import { createAppRouter, type AppRouter } from './app/router'
import {
  AppServicesProvider,
  createAppServices,
  type AppServices,
} from './app/services'
import { gptloadTheme } from './theme/gptload'
import { useThemePreference } from './theme/theme-preference'

// The services layer is created before the router; the ref lets the global
// unauthorized handler navigate once the router exists.
const routerRef: { current?: AppRouter } = {}

const services: AppServices = createAppServices({
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
      <QueryClientProvider client={services.queryClient}>
        <AppServicesProvider value={services}>
          <RouterProvider router={appRouter} />
        </AppServicesProvider>
      </QueryClientProvider>
    </Theme>
  )
}

const host = document.getElementById('app')
if (!host) throw new Error('missing #app mount point')
createRoot(host).render(<App />)
