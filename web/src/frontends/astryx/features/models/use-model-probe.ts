import { useRef, useState } from 'react'

import {
  runModelProbe,
  type ModelProbeResultDto,
  type ModelProbeTargetDto,
} from '@shared/control/resources/model-probe'

import { useAppServices } from '../../app/services'

// One batch request stays bounded, so the dialog sends targets in sequential
// chunks instead of one long-lived HTTP request.
const chunkSize = 8

export interface ModelProbeSummary {
  passed: number
  failed: number
  inconclusive: number
}

/**
 * A disabled group still serves a probe on explicit request; the dialog labels
 * those targets so an opted-in result is never mistaken for a routable group.
 */
export interface ModelProbeStartOptions {
  disabledGroupIds?: readonly number[]
}

export interface ModelProbeRunState {
  readonly open: boolean
  readonly pending: boolean
  readonly failed: boolean
  readonly stopped: boolean
  readonly targets: ModelProbeTargetDto[]
  readonly disabledGroupIds: number[]
  readonly results: ModelProbeResultDto[]
}

const idleRunState: ModelProbeRunState = {
  open: false,
  pending: false,
  failed: false,
  stopped: false,
  targets: [],
  disabledGroupIds: [],
  results: [],
}

/**
 * Shared model-probe state machine. Both probe entry points (group models tab and
 * dispatch center) own an instance and only differ in which targets they collect.
 */
export function useModelProbe() {
  const { apiClient } = useAppServices()
  const [run, setRun] = useState<ModelProbeRunState>(idleRunState)
  // Bumped on close/restart so an in-flight chunk cannot append into a newer run.
  const generationRef = useRef(0)
  const stoppedRef = useRef(false)

  const total = run.targets.length
  const completed = run.results.length
  const summary: ModelProbeSummary = { passed: 0, failed: 0, inconclusive: 0 }
  for (const result of run.results) summary[result.outcome] += 1

  async function start(
    nextTargets: readonly ModelProbeTargetDto[],
    options: ModelProbeStartOptions = {},
  ): Promise<void> {
    if (nextTargets.length === 0) return
    generationRef.current += 1
    const current = generationRef.current
    const targets = [...nextTargets]
    stoppedRef.current = false
    setRun({
      open: true,
      pending: true,
      failed: false,
      stopped: false,
      targets,
      disabledGroupIds: [...(options.disabledGroupIds ?? [])],
      results: [],
    })
    try {
      for (let index = 0; index < targets.length; index += chunkSize) {
        if (stoppedRef.current || generationRef.current !== current) return
        const chunkResults = await runModelProbe(apiClient, targets.slice(index, index + chunkSize))
        if (generationRef.current !== current) return
        setRun((prev) => ({ ...prev, results: [...prev.results, ...chunkResults] }))
      }
    } catch {
      if (generationRef.current === current) setRun((prev) => ({ ...prev, failed: true }))
    } finally {
      if (generationRef.current === current) setRun((prev) => ({ ...prev, pending: false }))
    }
  }

  // Stopping only stops sending further chunks: results for an already-issued
  // request stay visible, because that upstream request (and its cost) happened.
  function stop(): void {
    stoppedRef.current = true
    setRun((prev) => ({ ...prev, stopped: true }))
  }

  function close(): void {
    if (run.pending) return
    generationRef.current += 1
    stoppedRef.current = false
    setRun(idleRunState)
  }

  return {
    ...run,
    total,
    completed,
    summary,
    start,
    stop,
    close,
  }
}
