import { computed, ref } from 'vue'

import { useApiClient } from '@/api/client-context'
import {
  runModelProbe,
  type ModelProbeResultDto,
  type ModelProbeTargetDto,
} from '@/app/resources/model-probe'

// One batch request stays bounded, so the dialog sends targets in sequential
// chunks instead of one long-lived HTTP request.
const chunkSize = 8

export interface ModelProbeSummary {
  passed: number
  failed: number
  inconclusive: number
}

/**
 * Shared model-probe state machine. Both probe entry points (group models tab and
 * dispatch center) own an instance and only differ in which targets they collect.
 */
export function useModelProbe() {
  const client = useApiClient()
  const open = ref(false)
  const pending = ref(false)
  const failed = ref(false)
  const stopped = ref(false)
  const targets = ref<ModelProbeTargetDto[]>([])
  const results = ref<ModelProbeResultDto[]>([])
  // Bumped on close/restart so an in-flight chunk cannot append into a newer run.
  let generation = 0

  const total = computed(() => targets.value.length)
  const completed = computed(() => results.value.length)
  const summary = computed<ModelProbeSummary>(() => {
    const value: ModelProbeSummary = { passed: 0, failed: 0, inconclusive: 0 }
    for (const result of results.value) value[result.outcome] += 1
    return value
  })

  async function start(nextTargets: readonly ModelProbeTargetDto[]): Promise<void> {
    if (nextTargets.length === 0) return
    generation += 1
    const current = generation
    targets.value = [...nextTargets]
    results.value = []
    failed.value = false
    stopped.value = false
    pending.value = true
    open.value = true
    try {
      for (let index = 0; index < targets.value.length; index += chunkSize) {
        if (stopped.value || generation !== current) return
        const chunk = targets.value.slice(index, index + chunkSize)
        const chunkResults = await runModelProbe(client, chunk)
        if (generation !== current) return
        results.value = [...results.value, ...chunkResults]
      }
    } catch {
      if (generation === current) failed.value = true
    } finally {
      if (generation === current) pending.value = false
    }
  }

  // Stopping only stops sending further chunks: results for an already-issued
  // request stay visible, because that upstream request (and its cost) happened.
  function stop(): void {
    stopped.value = true
  }

  function close(): void {
    if (pending.value) return
    open.value = false
    generation += 1
    targets.value = []
    results.value = []
    failed.value = false
    stopped.value = false
  }

  return {
    open,
    pending,
    failed,
    stopped,
    targets,
    results,
    total,
    completed,
    summary,
    start,
    stop,
    close,
  }
}
