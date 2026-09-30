import type { QueryClient } from '@tanstack/query-core'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { controlQueryKeys } from '@shared/control/query-keys'
import {
  updateSettings,
  type RuntimeSettingKey,
  type SettingsPatch,
  type SettingsResource,
} from '@shared/control/resources/settings'
import {
  buildSettingsPatch,
  createSettingsDraft,
  validateSettingsSection,
  type SettingsDraft,
} from '@shared/domain/settings/settings-patch'
import type { ApiClient } from '@shared/http/client'
import { RequestCancelledError } from '@shared/http/errors'

export interface SettingsDraftChange {
  key: RuntimeSettingKey
  draft: SettingsDraft
}

export interface SettingsDraftControllerOptions {
  client: ApiClient
  queryClient: QueryClient
  locale: string
  now?: () => Date
}

export interface SettingsDraftController {
  getBase(): SettingsResource | null
  getDraft(): SettingsDraft | null
  getPatch(): SettingsPatch
  isDirty(): boolean
  isValid(): boolean
  isPending(): boolean
  hasFailed(): boolean
  isOperationLocked(): boolean
  getSavedAt(): Date | null
  subscribe(listener: () => void): () => void
  setLocale(next: string): void
  setResource(next: SettingsResource | null): void
  setLocalEdits(hasLocalEdits: boolean): void
  updateDraft(change: SettingsDraftChange): void
  discard(): void
  saveAll(extra?: SettingsPatch): Promise<void>
  dispose(): void
}

function cloneDraft(draft: SettingsDraft): SettingsDraft {
  return createSettingsDraft({
    values: draft.values,
    overrides: [...draft.overrides],
    read_only: [...draft.readOnly],
  })
}

export function createSettingsDraftController(
  options: SettingsDraftControllerOptions,
): SettingsDraftController {
  const now = options.now ?? (() => new Date())
  let currentLocale = options.locale
  const settingsQueryKey = () => controlQueryKeys.settings(currentLocale)

  let base: SettingsResource | null = null
  let draft: SettingsDraft | null = null
  let resource: SettingsResource | null = null
  let pending = false
  let failed = false
  let savedAt: Date | null = null
  let hasLocalEdits = false
  let requestOwner = 0
  let requestController: AbortController | undefined
  let mounted = true
  const listeners = new Set<() => void>()
  const notify = () => {
    for (const listener of listeners) listener()
  }

  const isDirty = () => {
    if (!base || !draft) return false
    return Object.keys(buildSettingsPatch(base.settings, draft, 'all')).length > 0
  }

  const isValid = () =>
    draft !== null &&
    validateSettingsSection(draft, 'request-forwarding') &&
    validateSettingsSection(draft, 'affinity') &&
    validateSettingsSection(draft, 'browser-access') &&
    validateSettingsSection(draft, 'logs-maintenance') &&
    validateSettingsSection(draft, 'model-prices')

  function isCurrent(owner: number, controller: AbortController): boolean {
    return (
      mounted &&
      owner === requestOwner &&
      requestController === controller &&
      !controller.signal.aborted
    )
  }

  function reset(next: SettingsResource): void {
    base = next
    draft = createSettingsDraft(next.settings)
    failed = false
    notify()
  }

  function consumeCurrentResource(): void {
    const next = resource
    if (!next || next === base) return
    if (!base || !draft) {
      reset(next)
      return
    }
    if (isDirty() || hasLocalEdits || pending) return
    reset(next)
  }

  function updateDraft(change: SettingsDraftChange): void {
    if (pending || !draft) return
    draft = cloneDraft(change.draft)
    failed = false
    notify()
    consumeCurrentResource()
  }

  function discard(): void {
    if (pending || !base) return
    reset(resource ?? base)
  }

  async function markSaved(next: SettingsResource): Promise<void> {
    reset(next)
    savedAt = new Date(now().getTime())
    notify()
    options.queryClient.setQueryData(settingsQueryKey(), next)
    await applyInvalidationPlan(options.queryClient, mutationInvalidationPlans.settings.update())
  }

  async function saveAll(extra: SettingsPatch = {}): Promise<void> {
    if (pending || !isValid() || !base || !draft) return
    const patch = buildSettingsPatch(base.settings, draft, 'all')
    if (Object.keys(patch).length === 0 && Object.keys(extra).length === 0) return

    const normalizedPatch = { ...patch, ...extra }
    requestController?.abort()
    const owner = ++requestOwner
    const controller = new AbortController()
    requestController = controller
    pending = true
    failed = false
    notify()

    try {
      const response = await updateSettings(options.client, normalizedPatch, controller.signal)
      if (!isCurrent(owner, controller)) return
      await options.queryClient.cancelQueries({ queryKey: settingsQueryKey(), exact: true })
      if (!isCurrent(owner, controller)) return
      await markSaved(response)
    } catch (error: unknown) {
      if (!isCurrent(owner, controller) || error instanceof RequestCancelledError) return
      failed = true
      notify()
    } finally {
      if (isCurrent(owner, controller)) {
        requestController = undefined
        pending = false
        notify()
        consumeCurrentResource()
      }
    }
  }

  return {
    getBase: () => base,
    getDraft: () => draft,
    getPatch: () => (base && draft ? buildSettingsPatch(base.settings, draft, 'all') : {}),
    isDirty,
    isValid,
    isPending: () => pending,
    hasFailed: () => failed,
    isOperationLocked: () => pending,
    getSavedAt: () => savedAt,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    setLocale(next) {
      currentLocale = next
    },
    setResource(next) {
      resource = next
      consumeCurrentResource()
    },
    setLocalEdits(next) {
      if (next === hasLocalEdits) return
      hasLocalEdits = next
      if (!hasLocalEdits) consumeCurrentResource()
    },
    updateDraft,
    discard,
    saveAll,
    dispose() {
      mounted = false
      requestOwner += 1
      requestController?.abort()
      requestController = undefined
      listeners.clear()
    },
  }
}
