export function readSingleCredential(raw: string): string | null {
  const value = raw.trim()
  if (!value || value.startsWith('[')) return null
  if (value.startsWith('{')) {
    try {
      const parsed: unknown = JSON.parse(value)
      return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed) ? value : null
    } catch {
      return null
    }
  }
  return /[\r\n]/u.test(value) ? null : value
}
