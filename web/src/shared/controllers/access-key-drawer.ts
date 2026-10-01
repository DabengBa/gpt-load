import type { QueryClient } from '@tanstack/react-query'
import type { ApiClient } from '@shared/http/client'
import { RequestCancelledError } from '@shared/http/errors'
import type { AccessKeyDto } from '@shared/control/types'
import {
  createAccessKey,
  updateAccessKey,
  type CreateAccessKeyRequest,
} from '@shared/control/resources/access-keys'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { classifyMutationOutcome } from '@shared/control/mutation-outcome'
import {
  accessKeyMatchesUpdatePatch,
  buildAccessKeyUpdatePatch,
  buildCreateAccessKeyInput,
  createAccessKeyDraft,
  createAccessKeyDraftFromCreateInput,
  createAccessKeyDraftFromUpdate,
  isAccessKeyDraftDirty,
  type AccessKeyDraft,
} from '@shared/domain/access-keys/access-key-patch'
import {
  cloneAccessKeyCreatePayload,
  type PendingAccessKeyCreateOperation,
} from '@shared/domain/access-keys/access-key-create-operation'
import {
  findAccessKeyForReconciliation,
  type PendingAccessKeyEditOperation,
} from '@shared/domain/access-keys/access-key-edit-operation'

export type AccessKeyDrawerMutationState = 'idle' | 'indeterminate' | 'reconciling'

export interface AccessKeyDrawerSnapshot {
  revision: number
  base: AccessKeyDto | null
  draft: AccessKeyDraft
  operationID: string
  createPayload: CreateAccessKeyRequest | null
  createOperationRetained: boolean
  editOperationRetained: boolean
  pending: boolean
  rotationPending: boolean
  failed: boolean
  mutationState: AccessKeyDrawerMutationState
  editReconciliation: PendingAccessKeyEditOperation | null
  editNotApplied: boolean
  modelInput: string
}

export interface AccessKeyDrawerOpenInput {
  accessKey: AccessKeyDto | null
  createOperation: PendingAccessKeyCreateOperation | null
  editOperation: PendingAccessKeyEditOperation | null
}

// Static construction deps; per-render callbacks are pushed via setRuntime
// (the Phase 2 controller convention — keeps refs out of useState initializers).
export interface AccessKeyDrawerDeps {
  client: ApiClient
  queryClient: QueryClient
  generateOperationID(): string
}

export interface AccessKeyDrawerRuntime {
  isOpen(): boolean
  onSaved(kind: 'created' | 'updated', name: string): void
  onDeleted(name: string): void
  setCreateOperation(operation: PendingAccessKeyCreateOperation | null): void
  setEditOperation(operation: PendingAccessKeyEditOperation | null): void
}

function createSnapshot(): AccessKeyDrawerSnapshot {
  return {
    revision: 0,
    base: null,
    draft: createAccessKeyDraft(),
    operationID: '',
    createPayload: null,
    createOperationRetained: false,
    editOperationRetained: false,
    pending: false,
    rotationPending: false,
    failed: false,
    mutationState: 'idle',
    editReconciliation: null,
    editNotApplied: false,
    modelInput: '',
  }
}

export function isAccessKeyDrawerCreateOperationActive(snapshot: AccessKeyDrawerSnapshot): boolean {
  return (
    snapshot.base === null &&
    snapshot.createPayload !== null &&
    (snapshot.mutationState !== 'idle' || snapshot.failed)
  )
}

export function isAccessKeyDrawerUnsavedDirty(snapshot: AccessKeyDrawerSnapshot): boolean {
  return (
    isAccessKeyDraftDirty(snapshot.draft, snapshot.base) &&
    !(isAccessKeyDrawerCreateOperationActive(snapshot) && snapshot.createOperationRetained) &&
    !(snapshot.editReconciliation !== null && snapshot.editOperationRetained)
  )
}

const idleRuntime: AccessKeyDrawerRuntime = {
  isOpen: () => false,
  onSaved: () => {},
  onDeleted: () => {},
  setCreateOperation: () => {},
  setEditOperation: () => {},
}

export function createAccessKeyDrawerController(deps: AccessKeyDrawerDeps) {
  let runtime: AccessKeyDrawerRuntime = idleRuntime
  const ops: AccessKeyDrawerDeps & AccessKeyDrawerRuntime = {
    client: deps.client,
    queryClient: deps.queryClient,
    generateOperationID: deps.generateOperationID,
    isOpen: () => runtime.isOpen(),
    onSaved: (kind, name) => runtime.onSaved(kind, name),
    onDeleted: (name) => runtime.onDeleted(name),
    setCreateOperation: (operation) => runtime.setCreateOperation(operation),
    setEditOperation: (operation) => runtime.setEditOperation(operation),
  }
  let state = createSnapshot()
  let snapshot: AccessKeyDrawerSnapshot = { ...state }
  let controller: AbortController | null = null
  const listeners = new Set<() => void>()

  function notify(): void {
    snapshot = { ...state, revision: state.revision + 1 }
    state = snapshot
    for (const listener of listeners) listener()
  }

  function patchState(patch: Partial<AccessKeyDrawerSnapshot>): void {
    state = { ...state, ...patch }
    notify()
  }

  function clearLocalState(): void {
    controller?.abort()
    controller = null
    state = { ...createSnapshot(), revision: state.revision }
    notify()
  }

  function resetForOpen(input: AccessKeyDrawerOpenInput): void {
    const carriedCreateOperation = input.accessKey ? null : input.createOperation
    const carriedEditOperation =
      input.accessKey && input.editOperation?.base.id === input.accessKey.id
        ? input.editOperation
        : null
    state = {
      ...createSnapshot(),
      revision: state.revision,
      base: carriedEditOperation?.base ?? input.accessKey,
      draft: carriedCreateOperation
        ? createAccessKeyDraftFromCreateInput(carriedCreateOperation.payload)
        : carriedEditOperation
          ? createAccessKeyDraftFromUpdate(carriedEditOperation.base, carriedEditOperation.patch)
          : createAccessKeyDraft(input.accessKey),
      operationID: input.accessKey
        ? ''
        : (carriedCreateOperation?.idempotencyKey ?? ops.generateOperationID()),
      createPayload: carriedCreateOperation
        ? cloneAccessKeyCreatePayload(carriedCreateOperation.payload)
        : null,
      createOperationRetained: carriedCreateOperation !== null,
      editOperationRetained: carriedEditOperation !== null,
      mutationState: carriedCreateOperation?.state ?? carriedEditOperation?.state ?? 'idle',
      editReconciliation: carriedEditOperation,
      modelInput: '',
    }
    notify()
  }

  async function save(valid: boolean): Promise<void> {
    if (state.pending) return
    if (state.editReconciliation) {
      await reconcileEdit()
      return
    }
    const createOperationActive = isAccessKeyDrawerCreateOperationActive(state)
    if (!createOperationActive && !(valid && isAccessKeyDraftDirty(state.draft, state.base))) {
      return
    }
    const currentBase = state.base
    const updateBody = currentBase ? buildAccessKeyUpdatePatch(currentBase, state.draft) : null
    const activeCreatePayload = currentBase
      ? null
      : (state.createPayload ?? buildCreateAccessKeyInput(state.draft))
    if (updateBody && Object.keys(updateBody).length === 0) return
    if (activeCreatePayload && !state.createPayload) {
      state.createPayload = cloneAccessKeyCreatePayload(activeCreatePayload)
    }
    patchState({ pending: true, failed: false, editNotApplied: false, mutationState: 'idle' })
    controller?.abort()
    controller = new AbortController()
    const activeController = controller
    const activeOperationID = state.operationID
    let savedName = ''
    let savedKind: 'created' | 'updated' | null = null
    let createdAccessKey: AccessKeyDto | null = null
    const stale = () =>
      controller !== activeController || !ops.isOpen() || state.operationID !== activeOperationID
    try {
      if (currentBase) {
        const saved = await updateAccessKey(
          ops.client,
          currentBase.id,
          updateBody!,
          activeController.signal,
        )
        if (stale()) return
        patchState({
          base: saved,
          draft: createAccessKeyDraft(saved),
          editReconciliation: null,
          editOperationRetained: false,
        })
        savedName = saved.name
        ops.setEditOperation(null)
      } else {
        const saved = await createAccessKey(
          ops.client,
          activeCreatePayload!,
          activeOperationID,
          activeController.signal,
        )
        if (stale()) return
        savedName = saved.name
        createdAccessKey = saved
        patchState({ createPayload: null, createOperationRetained: false })
        ops.setCreateOperation(null)
      }
      await applyInvalidationPlan(
        ops.queryClient,
        mutationInvalidationPlans.accessKey[currentBase ? 'update' : 'create'],
      )
      if (createdAccessKey) {
        patchState({
          base: createdAccessKey,
          draft: createAccessKeyDraft(createdAccessKey),
        })
      }
      savedKind = currentBase ? 'updated' : 'created'
    } catch (error: unknown) {
      if (stale()) return
      if (error instanceof RequestCancelledError) return
      const outcome = classifyMutationOutcome({ kind: 'error', error, requestSent: true })
      patchState({ failed: outcome.kind === 'failed' })
      if (!currentBase && outcome.kind === 'failed' && outcome.reason === 'rejected') {
        patchState({
          operationID: ops.generateOperationID(),
          createPayload: null,
          createOperationRetained: false,
        })
        ops.setCreateOperation(null)
      } else if (
        !currentBase &&
        activeCreatePayload &&
        (outcome.kind === 'indeterminate' || outcome.kind === 'reconciling')
      ) {
        ops.setCreateOperation({
          idempotencyKey: activeOperationID,
          payload: cloneAccessKeyCreatePayload(activeCreatePayload),
          state: outcome.kind,
        })
        patchState({ createOperationRetained: true })
      } else if (
        currentBase &&
        updateBody &&
        (outcome.kind === 'indeterminate' || outcome.kind === 'reconciling')
      ) {
        const operation: PendingAccessKeyEditOperation = {
          base: currentBase,
          patch: updateBody,
          state: outcome.kind,
        }
        patchState({ editReconciliation: operation, editOperationRetained: true })
        ops.setEditOperation(operation)
      }
      patchState({
        mutationState:
          outcome.kind === 'reconciling'
            ? 'reconciling'
            : outcome.kind === 'indeterminate'
              ? 'indeterminate'
              : 'idle',
      })
    } finally {
      if (controller === activeController) {
        controller = null
        patchState({ pending: false })
      }
    }
    if (savedKind) ops.onSaved(savedKind, savedName)
  }

  async function reconcileEdit(): Promise<void> {
    const attempt = state.editReconciliation
    if (!attempt || state.pending) return
    patchState({ pending: true, failed: false, editNotApplied: false })
    controller?.abort()
    controller = new AbortController()
    const activeController = controller
    let confirmedName: string | null = null
    const stale = () =>
      controller !== activeController || state.editReconciliation !== attempt || !ops.isOpen()
    try {
      const latest = await findAccessKeyForReconciliation(
        ops.client,
        attempt.base.id,
        activeController.signal,
      )
      if (stale()) return
      await applyInvalidationPlan(
        ops.queryClient,
        mutationInvalidationPlans.accessKey.reconcile,
        () => !stale(),
      )
      if (stale()) return
      if (!latest) {
        patchState({
          editReconciliation: null,
          editOperationRetained: false,
          mutationState: 'idle',
          failed: true,
        })
        ops.setEditOperation(null)
        return
      }
      if (accessKeyMatchesUpdatePatch(latest, attempt.patch, attempt.base)) {
        patchState({
          base: latest,
          draft: createAccessKeyDraft(latest),
          editReconciliation: null,
          editOperationRetained: false,
          mutationState: 'idle',
        })
        ops.setEditOperation(null)
        await applyInvalidationPlan(
          ops.queryClient,
          mutationInvalidationPlans.accessKey.reconcileConfirmed,
          () => controller === activeController && ops.isOpen(),
        )
        if (controller === activeController && ops.isOpen()) confirmedName = latest.name
      } else if (
        Object.keys(buildAccessKeyUpdatePatch(attempt.base, createAccessKeyDraft(latest)))
          .length === 0
      ) {
        patchState({
          base: latest,
          editReconciliation: null,
          editOperationRetained: false,
          mutationState: 'idle',
          failed: true,
          editNotApplied: true,
        })
        ops.setEditOperation(null)
        return
      } else {
        const operation: PendingAccessKeyEditOperation = { ...attempt, state: 'indeterminate' }
        patchState({ editReconciliation: operation, mutationState: operation.state })
        ops.setEditOperation(operation)
      }
    } catch (error: unknown) {
      if (
        controller === activeController &&
        state.editReconciliation === attempt &&
        !(error instanceof RequestCancelledError)
      ) {
        const operation: PendingAccessKeyEditOperation = { ...attempt, state: 'indeterminate' }
        patchState({ editReconciliation: operation, mutationState: operation.state })
        ops.setEditOperation(operation)
      }
    } finally {
      if (controller === activeController) {
        controller = null
        patchState({ pending: false })
      }
    }
    if (confirmedName && ops.isOpen()) ops.onSaved('updated', confirmedName)
  }

  return {
    subscribe(listener: () => void): () => void {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    getSnapshot(): AccessKeyDrawerSnapshot {
      return snapshot
    },
    resetForOpen,
    close: clearLocalState,
    setDraft(draft: AccessKeyDraft): void {
      patchState({ draft })
    },
    setModelInput(value: string): void {
      patchState({ modelInput: value })
    },
    setRotationPending(value: boolean): void {
      patchState({ rotationPending: value })
    },
    applyRotated(accessKey: AccessKeyDto): void {
      patchState({ base: accessKey, draft: createAccessKeyDraft(accessKey) })
    },
    applyDeleted(name: string): void {
      clearLocalState()
      ops.setEditOperation(null)
      ops.onDeleted(name)
    },
    save,
    reconcileEdit,
    setRuntime(next: AccessKeyDrawerRuntime): void {
      runtime = next
    },
    dispose(): void {
      controller?.abort()
      controller = null
      listeners.clear()
    },
  }
}

export type AccessKeyDrawerController = ReturnType<typeof createAccessKeyDrawerController>
