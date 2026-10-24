type CredentialInputResult =
  | { credential: string; error: null }
  | { credential: null; error: 'required' | 'tooMany' | 'invalidJSON' }

export function readSingleCredential(raw: string): CredentialInputResult {
  const value = raw.trim()
  if (!value) return { credential: null, error: 'required' }
  if (value.startsWith('[')) return { credential: null, error: 'invalidJSON' }
  if (value.startsWith('{')) {
    try {
      JSON.parse(value)
      return { credential: value, error: null }
    } catch {
      return { credential: null, error: 'invalidJSON' }
    }
  }
  return /[\r\n]/u.test(value)
    ? { credential: null, error: 'tooMany' }
    : { credential: value, error: null }
}
