import { useState } from 'react'

import { fingerprintRunFailure, runModelFingerprint } from '@shared/control/resources/fingerprint'
import {
  createFingerprintRunner,
  initialFingerprintRunState,
} from '@shared/domain/models/fingerprint-run'

import { useAppServices } from '../../app/services'

/**
 * Single-target fingerprint detection. The runner owns the duplicate-click
 * guard because the operation is a real paid model execution, and this hook
 * only mirrors its state into React.
 */
export function useModelFingerprint() {
  const { apiClient } = useAppServices()
  const [run, setRun] = useState(initialFingerprintRunState)
  // App services (and therefore the API client) are created once per session,
  // so the runner can capture the client for its lifetime.
  const [runner] = useState(() =>
    createFingerprintRunner({
      run: (target) => runModelFingerprint(apiClient, target),
      describeFailure: fingerprintRunFailure,
      onChange: setRun,
    }),
  )

  return { run, start: runner.start, close: runner.close }
}
