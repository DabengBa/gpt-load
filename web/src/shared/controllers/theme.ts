export type AppTheme = 'system' | 'light' | 'dark'

export const themeStorageKey = 'gpt-load.theme'

export function isTheme(value: unknown): value is AppTheme {
  return value === 'system' || value === 'light' || value === 'dark'
}
