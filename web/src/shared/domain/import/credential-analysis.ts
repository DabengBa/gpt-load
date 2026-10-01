export interface CredentialAnalysis {
  raw: string
  nonEmptyCount: number
  emptyLineCount: number
  duplicateCount: number
  likelyAccessKeyCount: number
  tooManyCredentials: boolean
}

export function analyzeCredentials(raw: string): CredentialAnalysis {
  const normalizedRaw = raw.replace(/\r/g, '')
  const trimmedRaw = normalizedRaw.trim()
  const wholeObject =
    trimmedRaw.startsWith('{') &&
    (() => {
      try {
        return typeof JSON.parse(trimmedRaw) === 'object'
      } catch {
        return false
      }
    })()
  const lines = (wholeObject ? [trimmedRaw] : normalizedRaw.split('\n')).map((line) => line.trim())
  const nonEmpty = lines.filter(Boolean)
  const seen = new Set<string>()
  let duplicateCount = 0

  for (const line of nonEmpty) {
    if (seen.has(line)) duplicateCount += 1
    else seen.add(line)
  }

  return {
    raw,
    nonEmptyCount: nonEmpty.length,
    emptyLineCount: raw ? lines.length - nonEmpty.length : 0,
    duplicateCount,
    likelyAccessKeyCount: nonEmpty.filter((line) => /^sk-gl-/i.test(line)).length,
    tooManyCredentials: seen.size > 1,
  }
}
