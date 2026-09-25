// The value must stay identical to frontendCookieName in
// internal/webui/server.go; scripts/verify-frontend-cookie.mjs enforces the
// cross-language contract.
export const frontendCookieName = 'gpt-load.frontend'

export type FrontendPreference = 'classic' | 'astryx'

export function isFrontendPreference(value: unknown): value is FrontendPreference {
  return value === 'classic' || value === 'astryx'
}

export function readFrontendPreference(
  cookieHeader: string | null | undefined,
): FrontendPreference {
  if (cookieHeader == null) return 'classic'
  for (const pair of cookieHeader.split(';')) {
    const separator = pair.indexOf('=')
    if (separator === -1) continue
    if (pair.slice(0, separator).trim() !== frontendCookieName) continue
    const value = pair.slice(separator + 1).trim()
    return isFrontendPreference(value) ? value : 'classic'
  }
  return 'classic'
}

export function frontendPreferenceCookie(value: FrontendPreference): string {
  return `${frontendCookieName}=${value}; path=/; samesite=strict; max-age=31536000`
}
