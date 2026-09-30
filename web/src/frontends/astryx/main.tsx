import { LinkProvider } from '@astryxdesign/core/Link'
import { Theme } from '@astryxdesign/core/theme'
import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { createRoot } from 'react-dom/client'
import type { ComponentProps, ReactNode } from 'react'

import { getBrowserLocale } from '@shared/preferences/locale'

import './entry.css'
import { AppI18nProviders, createAppI18n, type AppI18n } from './app/i18n'
import { createAppRouter, type AppRouter } from './app/router'
import { RouteLink } from './app/route-link'
import { AppServicesProvider, createAppServices, type AppServices } from './app/services'
import { gptloadTheme } from './theme/gptload'
import { useThemePreference } from './theme/theme-preference'

// Astryx components render links through LinkProvider; RouteLink keeps
// internal paths as router navigations and external targets as anchors.
function RouterLinkAdapter({
  href,
  children,
  ...rest
}: { href?: string; children?: ReactNode } & Omit<ComponentProps<'a'>, 'href'>) {
  return (
    <RouteLink to={href ?? '/'} {...rest}>
      {children}
    </RouteLink>
  )
}

function App({
  i18n,
  services,
  router,
}: {
  i18n: AppI18n
  services: AppServices
  router: AppRouter
}) {
  const [mode] = useThemePreference()
  return (
    <Theme theme={gptloadTheme} mode={mode}>
      <AppI18nProviders i18n={i18n}>
        <QueryClientProvider client={services.queryClient}>
          <AppServicesProvider value={services}>
            <LinkProvider component={RouterLinkAdapter}>
              <RouterProvider router={router} />
            </LinkProvider>
          </AppServicesProvider>
        </QueryClientProvider>
      </AppI18nProviders>
    </Theme>
  )
}

// Resolve the locale and load the core catalog before first render, then
// create services (the api client reads the locale from the controller) and
// the router.
async function bootstrap(host: HTMLElement): Promise<void> {
  const i18n = await createAppI18n()

  // The services layer is created before the router; the ref lets the global
  // unauthorized handler navigate once the router exists.
  const routerRef: { current?: AppRouter } = {}
  const services = createAppServices({
    i18n,
    onSessionCleared: (currentHref) =>
      routerRef.current?.navigate({
        href: `/login?redirect=${encodeURIComponent(currentHref)}`,
        replace: true,
      }),
  })
  const router = createAppRouter(services)
  routerRef.current = router

  createRoot(host).render(<App i18n={i18n} services={services} router={router} />)
}

// Catalogs are not loaded yet when boot fails, so the fallback carries its own
// copy for every supported locale.
function showStartupFailure(host: HTMLElement): void {
  const labels = {
    'zh-CN': { message: '无法加载界面，请重试。', retry: '重新加载' },
    'en-US': { message: 'Unable to load the interface. Please retry.', retry: 'Reload' },
    'ja-JP': { message: '画面を読み込めません。再試行してください。', retry: '再読み込み' },
  }[getBrowserLocale()]
  const message = document.createElement('p')
  message.setAttribute('role', 'alert')
  message.textContent = labels.message
  const retry = document.createElement('button')
  retry.type = 'button'
  retry.textContent = labels.retry
  retry.addEventListener('click', () => window.location.reload())
  host.replaceChildren(message, retry)
}

const host = document.getElementById('app')
if (!host) throw new Error('missing #app mount point')
void bootstrap(host).catch(() => showStartupFailure(host))
