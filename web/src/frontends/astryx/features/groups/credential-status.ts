import type { CredentialStatus } from '@shared/control/types'

/** Credential row status — `CredentialStatus` plus the two summary-only
 *  states the unified list surfaces (`unavailable` for reauth-required,
 *  `unknown` for refreshing/outcome-unknown). */
export type OperationalStatus = CredentialStatus | 'unavailable' | 'unknown'

// Mirrors classic status-presenter.ts operationalPresentations:
// available→success, cooldown→warning, blacklisted/unavailable→danger,
// else neutral. Shared by the unified credential list and the credentials
// management ledger rows.
export function credentialStatusBadgeVariant(
  status: OperationalStatus,
): 'success' | 'error' | 'warning' | 'neutral' {
  if (status === 'available') return 'success'
  if (status === 'cooldown') return 'warning'
  if (status === 'blacklisted' || status === 'unavailable') return 'error'
  return 'neutral'
}
