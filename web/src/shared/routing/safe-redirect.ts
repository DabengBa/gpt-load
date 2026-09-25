import { pagePath, pagePathMatches } from './page-routes'
import { sharedPageRouteNames } from './route-names'

export interface SafeRedirectResolution {
  readonly matched: readonly unknown[]
  readonly name: unknown
  readonly path: string
  readonly fullPath: string
  readonly meta: Record<string, unknown>
}

export function decodedPathSegments(path: string): string[] {
  try {
    const segments = decodeURIComponent(path).split('/').filter(Boolean)
    return segments.length > 0 ? segments : ['invalid-path']
  } catch {
    return ['invalid-path']
  }
}

export function safeRedirectTarget(
  raw: unknown,
  resolve: (raw: string) => SafeRedirectResolution,
  blockedRouteNames: readonly string[],
): string {
  const fallback = pagePath(sharedPageRouteNames.home)
  if (
    typeof raw !== 'string' ||
    !raw.startsWith('/') ||
    raw.startsWith('//') ||
    raw.includes('\\')
  ) {
    return fallback
  }

  let decodedRaw: string
  try {
    decodedRaw = decodeURIComponent(raw)
  } catch {
    return fallback
  }
  if (decodedRaw.startsWith('//') || decodedRaw.includes('\\')) {
    return fallback
  }

  const resolved = resolve(raw)
  if (
    resolved.matched.length === 0 ||
    typeof resolved.name !== 'string' ||
    !pagePathMatches(resolved.name, resolved.path) ||
    blockedRouteNames.includes(resolved.name) ||
    resolved.meta.requiresAuth !== true
  ) {
    return fallback
  }
  return resolved.fullPath
}
