import { createContext, useContext } from 'react'

import { QueryClient } from '@tanstack/react-query'

import { createApiClient, type ApiClientWithResponse } from '@shared/http/client'
import type { AuthSessionPayload } from '@shared/http/types'
import { createAuthSession, type AuthSession } from '@shared/controllers/auth-session'
import { clearEphemeralState } from '@shared/controllers/ephemeral-state'
import {
  createImportRecoveryService,
  type ImportRecoveryService,
} from '@shared/controllers/import-recovery'
import {
  createUnsavedChangesController,
  type UnsavedChangesController,
} from '@shared/controllers/unsaved-changes'
import type { AppI18n } from './i18n'

export interface AppServices {
  readonly queryClient: QueryClient
  readonly apiClient: ApiClientWithResponse
  readonly authSession: AuthSession
  readonly unsavedChanges: UnsavedChangesController
  readonly importRecovery: ImportRecoveryService
  readonly i18n: AppI18n
}

function getBrowserStorage(name: 'localStorage' | 'sessionStorage'): Storage | undefined {
  try {
    return window[name]
  } catch {
    return undefined
  }
}

export interface AppServicesOptions {
  // Called when the API client sees a global 401; the composition root wires
  // this to router navigation because services are created before the router.
  onSessionCleared(currentHref: string): void
  // The resolved locale controller — created before services in main.tsx so
  // the api client's locale header follows locale switches.
  i18n: AppI18n
}

export function createAppServices(options: AppServicesOptions): AppServices {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  const unsavedChanges = createUnsavedChangesController()
  // The api client needs the session's key getter, the session needs the
  // client for validation — the ref breaks the construction cycle.
  const authRef: { current?: AuthSession } = {}

  const apiClient = createApiClient({
    fetch: window.fetch.bind(window),
    getAuthKey: () => authRef.current?.getAuthKey() ?? '',
    getLocale: () => options.i18n.getLocale(),
    onUnauthorized: () => {
      const session = authRef.current
      if (session === undefined) return
      session.clear()
      options.onSessionCleared(`${window.location.pathname}${window.location.search}`)
    },
  })

  const authSession = createAuthSession({
    storage: getBrowserStorage('localStorage'),
    queryClient,
    onClear: clearEphemeralState,
    validate: (key, globalUnauthorized, signal) =>
      apiClient.request<AuthSessionPayload>('/api/auth/session', {
        authKey: key,
        handleUnauthorized: globalUnauthorized,
        signal,
      }),
  })
  authRef.current = authSession

  const importRecovery = createImportRecoveryService({
    storage: getBrowserStorage('localStorage'),
    now: () => Date.now(),
    setTimer: (callback, delayMs) => setTimeout(callback, delayMs),
    clearTimer: (timer) => clearTimeout(timer),
  })

  return {
    queryClient,
    apiClient,
    authSession,
    unsavedChanges,
    importRecovery,
    i18n: options.i18n,
  }
}

const AppServicesContext = createContext<AppServices | null>(null)

export const AppServicesProvider = AppServicesContext.Provider

export function useAppServices(): AppServices {
  const services = useContext(AppServicesContext)
  if (services === null) throw new Error('APP_SERVICES_NOT_PROVIDED')
  return services
}
