import { InternationalizationProvider } from '@astryxdesign/core/i18n'
import astryxJaJP from '@astryxdesign/core/locales/ja-JP.json'
import astryxZhCN from '@astryxdesign/core/locales/zh-CN.json'
import { useCallback, useSyncExternalStore, type ReactNode } from 'react'
import {
  IntlProvider,
  useIntl,
  type IntlConfig,
  type PrimitiveType,
} from 'react-intl'

import { catalogLoader, flattenMessages } from '@shared/i18n/catalogs'
import type { MessageId } from '@shared/i18n/message-ids'
import type { MessageNamespace } from '@shared/i18n/namespaces'
import {
  getBrowserLocale,
  localeStorageKey,
  type AppLocale,
} from '@shared/preferences/locale'

export interface I18nSnapshot {
  readonly locale: AppLocale
  // Flattened dict for react-intl: en-US merged first, active locale overlaid
  // per namespace — same net fallback as classic's fallbackLocale='en-US'.
  readonly messages: Record<string, string>
}

export interface AppI18n {
  subscribe(listener: () => void): () => void
  getSnapshot(): I18nSnapshot
  getLocale(): AppLocale
  setLocale(locale: AppLocale): Promise<void>
  ensureNamespaces(namespaces: readonly MessageNamespace[]): Promise<void>
}

// Dev/test fail fast on MISSING_TRANSLATION/FORMAT errors (mirrors classic's
// dev missing-handler); production logs and keeps rendering fallbacks.
const onIntlError: NonNullable<IntlConfig['onError']> = (error) => {
  if (!import.meta.env.PROD) throw error
  console.error(`[i18n] ${error.code}: ${error.message}`)
}

function persistLocale(storage: Storage | undefined, locale: AppLocale): void {
  try {
    storage?.setItem(localeStorageKey, locale)
  } catch {
    // Persistence failures do not change the active in-memory preference.
  }
}

function resolveStorage(): Storage | undefined {
  try {
    return window.localStorage
  } catch {
    return undefined
  }
}

export async function createAppI18n(): Promise<AppI18n> {
  const storage = resolveStorage()
  const locale = getBrowserLocale()
  persistLocale(storage, locale)

  // Draft accumulates flattened catalogs; each emit() snapshots it so
  // useSyncExternalStore sees a fresh object identity.
  const draft: Record<string, string> = {}
  const loaded = new Set<string>()
  const pending = new Map<string, Promise<void>>()
  const listeners = new Set<() => void>()
  const namespacesToLoad = new Set<MessageNamespace>()
  let requestedLocale = locale
  let snapshot: I18nSnapshot = { locale, messages: { ...draft } }

  const emit = () => {
    snapshot = { locale: snapshot.locale, messages: { ...draft } }
    document.documentElement.lang = snapshot.locale
    for (const listener of listeners) listener()
  }

  async function ensure(targetLocale: AppLocale, namespace: 'core' | MessageNamespace) {
    const identity = `${targetLocale}:${namespace}`
    if (loaded.has(identity)) return
    const existing = pending.get(identity)
    if (existing) return existing
    const request = (async () => {
      const module = await catalogLoader(targetLocale, namespace)()
      Object.assign(draft, flattenMessages(module.default))
      loaded.add(identity)
      emit()
    })().finally(() => pending.delete(identity))
    pending.set(identity, request)
    return request
  }

  // Per-namespace: merge en-US first, then overlay the active locale.
  async function ensureMerged(targetLocale: AppLocale, namespace: 'core' | MessageNamespace) {
    if (targetLocale !== 'en-US') await ensure('en-US', namespace)
    await ensure(targetLocale, namespace)
  }

  await ensureMerged(locale, 'core')
  document.documentElement.lang = locale

  return {
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    getSnapshot() {
      return snapshot
    },
    getLocale() {
      return snapshot.locale
    },
    async setLocale(nextLocale) {
      requestedLocale = nextLocale
      persistLocale(storage, nextLocale)
      await Promise.all(
        ['core' as const, ...namespacesToLoad].map((namespace) =>
          ensureMerged(nextLocale, namespace),
        ),
      )
      if (requestedLocale !== nextLocale) return
      snapshot = { locale: nextLocale, messages: { ...draft } }
      document.documentElement.lang = nextLocale
      for (const listener of listeners) listener()
    },
    async ensureNamespaces(namespaces) {
      for (const namespace of namespaces) namespacesToLoad.add(namespace)
      await Promise.all(
        namespaces.map((namespace) => ensureMerged(snapshot.locale, namespace)),
      )
    },
  }
}

const astryxCatalogs = {
  'zh-CN': astryxZhCN,
  'ja-JP': astryxJaJP,
}

export function AppI18nProviders({
  i18n,
  children,
}: {
  i18n: AppI18n
  children: ReactNode
}) {
  const snap = useSyncExternalStore(i18n.subscribe, i18n.getSnapshot)
  return (
    <IntlProvider
      locale={snap.locale}
      messages={snap.messages}
      onError={onIntlError}
      // An authored empty string is a real translation (vue-i18n returns ''
      // for it), not a missing one — e.g. en-US quotaResetSuffix.
      fallbackOnEmptyString={false}
    >
      <InternationalizationProvider locale={snap.locale} messages={astryxCatalogs}>
        {children}
      </InternationalizationProvider>
    </IntlProvider>
  )
}

// Catalogs are plain ICU (verify-i18n-icu parses every message; no XML/rich
// tags), so values are primitives and formatMessage returns a string.
export type TValues = Record<string, PrimitiveType>

export function useT() {
  const intl = useIntl()
  return useCallback(
    (id: MessageId, values?: TValues): string =>
      intl.formatMessage({ id }, values) as string,
    [intl],
  )
}
