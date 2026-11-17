import { useEffect, useRef, useState } from 'react'

import {
  runModelProbe,
  type ModelProbeResultDto,
  type ModelProbeTargetDto,
} from '@shared/control/resources/model-probe'

import { useAppServices } from '../../app/services'

const concurrency = 4

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
  // A late callback must never append to a newer run.
  const generationRef = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)
  useEffect(() => () => controllerRef.current?.abort(), [])

  const total = run.targets.length
  const completed = run.results.length
  const summary: ModelProbeSummary = { passed: 0, failed: 0, inconclusive: 0 }
  for (const result of run.results) summary[result.outcome] += 1

  async function start(
    nextTargets: readonly ModelProbeTargetDto[],
    options: ModelProbeStartOptions = {},
  ): Promise<void> {
    if (nextTargets.length === 0) return
    controllerRef.current?.abort()
    generationRef.current += 1
    const current = generationRef.current
    const targets = nextTargets.filter(
      (target, index) =>
        nextTargets.findIndex(
          (other) =>
            other.group_id === target.group_id &&
            other.model === target.model &&
            other.protocol === target.protocol,
        ) === index,
    )
    const controller = new AbortController()
    controllerRef.current = controller
    setRun({
      open: true,
      pending: true,
      failed: false,
      stopped: false,
      targets,
      disabledGroupIds: [...(options.disabledGroupIds ?? [])],
      results: [],
    })
    let index = 0
    const resultsByTarget: ModelProbeResultDto[][] = Array.from(
      { length: targets.length },
      () => [],
    )
    async function worker(): Promise<void> {
      while (
        !controller.signal.aborted &&
        generationRef.current === current &&
        index < targets.length
      ) {
        const targetIndex = index++
        const target = targets[targetIndex]!
        try {
          const results = await runModelProbe(apiClient, [target], controller.signal)
          if (generationRef.current === current) {
            resultsByTarget[targetIndex] = results
            const orderedResults = resultsByTarget.flat()
            setRun((prev) => ({ ...prev, results: orderedResults }))
          }
        } catch {
          if (!controller.signal.aborted && generationRef.current === current) {
            setRun((prev) => ({ ...prev, failed: true }))
            controller.abort()
          }
        }
      }
    }
    await Promise.all(Array.from({ length: Math.min(concurrency, targets.length) }, () => worker()))
    if (generationRef.current === current) {
      controllerRef.current = null
      setRun((prev) => ({ ...prev, pending: false }))
    }
  }

  function stop(): void {
    controllerRef.current?.abort()
    setRun((prev) => ({ ...prev, stopped: true }))
  }

  function close(): void {
    if (controllerRef.current) stop()
    setRun((prev) => ({ ...prev, open: false }))
  }

  function reopen(): void {
    setRun((prev) => ({ ...prev, open: true }))
  }

  return {
    ...run,
    total,
    completed,
    summary,
    start,
    stop,
    close,
    reopen,
  }
}
