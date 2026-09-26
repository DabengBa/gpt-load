import type { QueryClient } from '@tanstack/query-core'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import {
  projectModelPriceMutationIssue,
  updateModelPrice,
  type ModelPriceDto,
} from '@shared/control/resources/model-prices'
import {
  buildModelPriceRequest,
  createEmptyTierDraft,
  createModelPriceDraft,
  modelPriceDraftChanged,
  modelPriceDraftIsAllNull,
  modelPriceFormHasErrors,
  validateModelPriceDraft,
  type ModelPriceDraft,
  type ModelPriceFormErrors,
  type ModelPriceScheduleDraft,
} from '@shared/domain/model-prices/model-price-form'
import type { ApiClient } from '@shared/http/client'
import { RequestCancelledError } from '@shared/http/errors'

export interface ModelPriceEditorControllerOptions {
  client: ApiClient
  queryClient: QueryClient
  /** Generic save failure text, resolved by the host's i18n layer. */
  failureMessage: () => string
}

export interface ModelPriceEditorSnapshot {
  revision: number
  baseline: ModelPriceDto
  draft: ModelPriceDraft
  errors: ModelPriceFormErrors
  pending: boolean
  failure: string
  unpricedConfirmOpen: boolean
  changed: boolean
  allNull: boolean
  canSave: boolean
}

export interface ModelPriceEditorController {
  getBaseline(): ModelPriceDto
  getDraft(): ModelPriceDraft
  getErrors(): ModelPriceFormErrors
  isPending(): boolean
  getFailure(): string
  getUnpricedConfirmOpen(): boolean
  /**
   * Memoized store snapshot for useSyncExternalStore. Render-time reads must
   * go through it: React Compiler can otherwise cache a stable controller's
   * method results and serve stale state inside transition-mounted views.
   */
  getSnapshot(): ModelPriceEditorSnapshot
  setUnpricedConfirmOpen(open: boolean): void
  hasChanged(): boolean
  isAllNull(): boolean
  canSave(): boolean
  subscribe(listener: () => void): () => void
  /** Row-watch parity: id 变化强制重建;updated_at_ms 变化仅在无草稿改动时重建。 */
  setRow(row: ModelPriceDto): void
  setDraft(draft: ModelPriceDraft): void
  /** 默认调度传 `undefined`;具名 mode(如 `fast`)传其 key。 */
  setScheduleDraft(mode: string | undefined, schedule: ModelPriceScheduleDraft): void
  addTier(mode?: string): void
  removeTier(key: string, mode?: string): void
  requestSave(): void
  confirmUnpricedSave(): void
  cancel(): void
  dispose(): void
}

export function createModelPriceEditorController(
  options: ModelPriceEditorControllerOptions,
  initialRow: ModelPriceDto,
): ModelPriceEditorController {
  let baseline = initialRow
  let draft = createModelPriceDraft(initialRow)
  let pending = false
  let failure = ''
  let unpricedConfirmOpen = false
  let mounted = true
  let requestController: AbortController | undefined
  let revision = 0
  const listeners = new Set<() => void>()

  const errors = () => validateModelPriceDraft(draft)
  const hasErrors = () => modelPriceFormHasErrors(errors())
  const changed = () => modelPriceDraftChanged(baseline, draft)
  const allNull = () => modelPriceDraftIsAllNull(draft)
  const ownershipIntent = () => allNull() && baseline.method !== 'user_marked_unpriced'
  const canSave = () => !hasErrors() && (changed() || ownershipIntent())

  let snapshot = buildSnapshot()
  function buildSnapshot(): ModelPriceEditorSnapshot {
    return {
      revision,
      baseline,
      draft,
      errors: errors(),
      pending,
      failure,
      unpricedConfirmOpen,
      changed: changed(),
      allNull: allNull(),
      canSave: canSave(),
    }
  }
  const notify = () => {
    revision += 1
    snapshot = buildSnapshot()
    for (const listener of listeners) listener()
  }

  function isCurrent(controller: AbortController): boolean {
    return mounted && requestController === controller && !controller.signal.aborted
  }

  function clearRequest(): void {
    requestController?.abort()
    requestController = undefined
    pending = false
  }

  function resetDraft(): void {
    clearRequest()
    draft = createModelPriceDraft(baseline)
    failure = ''
    unpricedConfirmOpen = false
    notify()
  }

  function setRow(row: ModelPriceDto): void {
    if (row.id !== baseline.id) {
      baseline = row
      resetDraft()
      return
    }
    if (row.updated_at_ms === baseline.updated_at_ms) return
    // changed() 与旧 baseline 比对:无本地改动才接纳新行;有改动时
    // baseline 与 draft 都保留,避免用户草稿被服务器刷新吞掉。
    if (!changed()) {
      baseline = row
      resetDraft()
    }
  }

  function setDraft(next: ModelPriceDraft): void {
    draft = next
    notify()
  }

  function scheduleDraft(mode?: string) {
    return mode ? draft.modeSchedules[mode] : draft
  }

  function cloneSchedule(mode?: string): ModelPriceScheduleDraft | null {
    const schedule = scheduleDraft(mode)
    if (!schedule) return null
    return { ...schedule, base: { ...schedule.base }, tiers: [...schedule.tiers] }
  }

  function commitSchedule(
    mode: string | undefined,
    schedule: ModelPriceScheduleDraft | null,
  ): void {
    if (!schedule) return
    if (mode === undefined) {
      setDraft({ ...schedule, modeSchedules: draft.modeSchedules })
      return
    }
    setDraft({
      ...draft,
      modeSchedules: { ...draft.modeSchedules, [mode]: schedule },
    })
  }

  function addTier(mode?: string): void {
    const schedule = cloneSchedule(mode)
    if (!schedule) return
    schedule.tiers = [...schedule.tiers, createEmptyTierDraft()]
    commitSchedule(mode, schedule)
  }

  function removeTier(key: string, mode?: string): void {
    const schedule = cloneSchedule(mode)
    if (!schedule) return
    schedule.tiers = schedule.tiers.filter((tier) => tier.key !== key)
    commitSchedule(mode, schedule)
  }

  function failureMessage(error: unknown): string {
    try {
      const issue = projectModelPriceMutationIssue(error)
      if (issue?.code === 'MODEL_PRICE_UNPRICED_CONFIRMATION_REQUIRED') {
        unpricedConfirmOpen = true
        notify()
        return ''
      }
    } catch {
      return options.failureMessage()
    }
    return options.failureMessage()
  }

  function canSubmit(confirmUnpriced: boolean): boolean {
    if (hasErrors()) return false
    if (allNull()) return confirmUnpriced && (changed() || ownershipIntent())
    return !confirmUnpriced && changed()
  }

  async function save(confirmUnpriced: boolean): Promise<void> {
    const request = buildModelPriceRequest(draft, confirmUnpriced)
    if (request === null || !canSubmit(confirmUnpriced) || pending) return
    pending = true
    failure = ''
    notify()
    requestController?.abort()
    const controller = new AbortController()
    requestController = controller
    try {
      const updated = await updateModelPrice(
        options.client,
        baseline.id,
        request,
        controller.signal,
      )
      if (!isCurrent(controller)) return
      baseline = updated
      draft = createModelPriceDraft(updated)
      await applyInvalidationPlan(
        options.queryClient,
        mutationInvalidationPlans.modelPrice.update,
      )
      if (!isCurrent(controller)) return
      unpricedConfirmOpen = false
    } catch (error: unknown) {
      if (
        isCurrent(controller) &&
        !(error instanceof RequestCancelledError)
      ) {
        failure = failureMessage(error)
      }
    } finally {
      if (isCurrent(controller)) {
        requestController = undefined
        pending = false
        notify()
      }
    }
  }

  function requestSave(): void {
    if (pending) return
    if (allNull()) {
      unpricedConfirmOpen = true
      notify()
      return
    }
    if (canSave()) void save(false)
  }

  function confirmUnpricedSave(): void {
    void save(true)
  }

  function cancel(): void {
    resetDraft()
  }

  return {
    getBaseline: () => baseline,
    getDraft: () => draft,
    getErrors: errors,
    isPending: () => pending,
    getFailure: () => failure,
    getUnpricedConfirmOpen: () => unpricedConfirmOpen,
    getSnapshot: () => snapshot,
    setUnpricedConfirmOpen(open) {
      if (open === unpricedConfirmOpen) return
      unpricedConfirmOpen = open
      notify()
    },
    hasChanged: changed,
    isAllNull: allNull,
    canSave,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    setRow,
    setDraft,
    setScheduleDraft: commitSchedule,
    addTier,
    removeTier,
    requestSave,
    confirmUnpricedSave,
    cancel,
    dispose() {
      mounted = false
      clearRequest()
      listeners.clear()
    },
  }
}
