import type {
  FingerprintRunFailure,
  ModelFingerprintResultDto,
  ModelFingerprintTargetDto,
} from '@shared/control/resources/fingerprint'

/**
 * Single-target fingerprint run state machine. The run owns at most one target
 * because the operation is an explicit per-model diagnostic, not a batch: a
 * second click while running must not queue another paid upstream execution.
 */
export type FingerprintRunStatus = 'idle' | 'running' | 'failed' | 'succeeded'

export interface FingerprintRunState {
  readonly status: FingerprintRunStatus
  readonly target: ModelFingerprintTargetDto | null
  readonly result: ModelFingerprintResultDto | null
  readonly failure: FingerprintRunFailure | null
}

export const initialFingerprintRunState: FingerprintRunState = {
  status: 'idle',
  target: null,
  result: null,
  failure: null,
}

export function startFingerprintRun(
  state: FingerprintRunState,
  target: ModelFingerprintTargetDto,
): FingerprintRunState {
  if (state.status === 'running') return state
  return { status: 'running', target, result: null, failure: null }
}

export function completeFingerprintRun(
  state: FingerprintRunState,
  result: ModelFingerprintResultDto,
): FingerprintRunState {
  if (state.status !== 'running') return state
  return { status: 'succeeded', target: state.target, result, failure: null }
}

export function failFingerprintRun(
  state: FingerprintRunState,
  failure: FingerprintRunFailure,
): FingerprintRunState {
  if (state.status !== 'running') return state
  return { status: 'failed', target: state.target, result: null, failure }
}

// Closing is blocked while the request is in flight: the upstream execution
// already happened and its result (or failure) must stay observable.
export function closeFingerprintRun(state: FingerprintRunState): FingerprintRunState {
  if (state.status === 'running') return state
  return initialFingerprintRunState
}

export function isFingerprintDialogOpen(state: FingerprintRunState): boolean {
  return state.status !== 'idle'
}

export interface FingerprintRunner {
  getState(): FingerprintRunState
  start(target: ModelFingerprintTargetDto): void
  close(): void
}

export interface FingerprintRunnerDeps {
  run(target: ModelFingerprintTargetDto): Promise<ModelFingerprintResultDto>
  describeFailure(cause: unknown): FingerprintRunFailure
  onChange(state: FingerprintRunState): void
}

/**
 * Framework-free owner of the run transitions and the in-flight guard. The
 * state machine alone cannot stop a duplicate click: React state lands after
 * the handler returns, so the runner keeps its own synchronous `inFlight`
 * flag and refuses to overlap two paid executions.
 */
export function createFingerprintRunner(deps: FingerprintRunnerDeps): FingerprintRunner {
  let state = initialFingerprintRunState
  let inFlight = false

  function update(next: FingerprintRunState): void {
    state = next
    deps.onChange(next)
  }

  function start(target: ModelFingerprintTargetDto): void {
    if (inFlight) return
    inFlight = true
    update(startFingerprintRun(state, target))
    void deps.run(target).then(
      (result) => {
        inFlight = false
        update(completeFingerprintRun(state, result))
      },
      (cause: unknown) => {
        inFlight = false
        update(failFingerprintRun(state, deps.describeFailure(cause)))
      },
    )
  }

  function close(): void {
    if (inFlight) return
    update(closeFingerprintRun(state))
  }

  return { getState: () => state, start, close }
}
