import { useSyncExternalStore } from 'react'

import { isTheme, themeStorageKey, type AppTheme } from '@shared/controllers/theme'

/**
 * Theme preference store for the Astryx frontend.
 *
 * Owns only the persisted preference (`gpt-load.theme` localStorage, shared
 * with classic). It deliberately does not touch `documentElement` — the root
 * `<Theme mode>` provider is the single writer of `data-theme`, matching
 * `theme-bootstrap.js` so first paint has no mode flash.
 */

function readPreference(): AppTheme {
  try {
    const stored = window.localStorage.getItem(themeStorageKey)
    if (isTheme(stored)) return stored
  } catch {
    // Storage denied — fall through to the system default.
  }
  return 'system'
}

let current: AppTheme = readPreference()
const listeners = new Set<() => void>()

function publish(next: AppTheme): void {
  if (next === current) return
  current = next
  for (const listener of listeners) listener()
}

function onStorage(event: StorageEvent): void {
  if (event.key !== null && event.key !== themeStorageKey) return
  publish(readPreference())
}

let observing = false

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  if (!observing) {
    window.addEventListener('storage', onStorage)
    observing = true
  }
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) {
      window.removeEventListener('storage', onStorage)
      observing = false
    }
  }
}

export function setThemePreference(theme: AppTheme): void {
  try {
    window.localStorage.setItem(themeStorageKey, theme)
  } catch {
    // Persistence failure still updates the in-memory preference.
  }
  publish(theme)
}

export function useThemePreference(): [AppTheme, typeof setThemePreference] {
  const theme = useSyncExternalStore(subscribe, () => current)
  return [theme, setThemePreference]
}
