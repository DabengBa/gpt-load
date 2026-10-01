import { useSyncExternalStore } from 'react'

import { classifyMutationOutcome, type MutationOutcome } from '@shared/control/mutation-outcome'
import type { CredentialConnectResult } from '@shared/control/resources/credential-stages'
import type {
  GroupCreateRequest,
  GroupCreateResult,
  CredentialImportResult,
} from '@shared/control/resources/groups'
import type { ImportDraft, ImportRecoveryDraft } from '@shared/domain/import/model-draft'
import { registerEphemeralStateCleaner } from '@shared/controllers/ephemeral-state'
import { createUUID } from '@shared/lib/uuid'

export interface StableImportOperation<TPayload> {
  idempotencyKey: string
  payload: TPayload
}

export interface OperationSnapshot<TPayload, TResult> {
  operation: StableImportOperation<TPayload> | null
  outcome: MutationOutcome<TResult> | null
  lastError: unknown
  pending: boolean
  retryReadyAt: number
  canRetry: boolean
}

const idleSnapshot: OperationSnapshot<never, never> = {
  operation: null,
  outcome: null,
  lastError: undefined,
  pending: false,
  retryReadyAt: 0,
  canRetry: false,
}

/**
 * Store-based port of classic `useStableImportOperation`: same idempotency
 * envelope, retry scheduling, and stale-owner guards, exposed through
 * useSyncExternalStore so pending/outcome flips re-render subscribers.
 */
export class StableOperationStore<TPayload, TResult> {
  private listeners = new Set<() => void>()
  private snapshot: OperationSnapshot<TPayload, TResult> = idleSnapshot as OperationSnapshot<
    TPayload,
    TResult
  >
  private controller: AbortController | undefined
  private retryTimer: ReturnType<typeof setTimeout> | undefined
  private clock = 0
  private owner = 0
  private readonly clonePayload: (payload: TPayload) => TPayload

  constructor(clonePayload?: (payload: TPayload) => TPayload) {
    this.clonePayload = clonePayload ?? ((payload) => structuredClone(payload))
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  getSnapshot = (): OperationSnapshot<TPayload, TResult> => this.snapshot

  private commit(patch: Partial<OperationSnapshot<TPayload, TResult>>): void {
    const next = { ...this.snapshot, ...patch }
    next.canRetry = !next.pending && next.operation !== null && this.clock >= next.retryReadyAt
    this.snapshot = next
    for (const listener of this.listeners) listener()
  }

  private clearRetryTimer(): void {
    if (this.retryTimer !== undefined) {
      clearTimeout(this.retryTimer)
      this.retryTimer = undefined
    }
  }

  private scheduleRetry(delayMs: number): void {
    this.clearRetryTimer()
    this.clock = Date.now()
    if (delayMs === 0) {
      this.commit({ retryReadyAt: this.clock })
      return
    }
    this.commit({ retryReadyAt: this.clock + delayMs })
    this.retryTimer = setTimeout(() => {
      this.retryTimer = undefined
      this.clock = Date.now()
      this.commit({})
    }, delayMs)
  }

  begin(payload: TPayload): StableImportOperation<TPayload> {
    const existing = this.snapshot.operation
    if (existing) return existing
    const operation: StableImportOperation<TPayload> = {
      idempotencyKey: createUUID(),
      payload: this.clonePayload(payload),
    }
    this.clearRetryTimer()
    this.clock = Date.now()
    this.commit({ operation, outcome: null, lastError: undefined, retryReadyAt: 0 })
    return operation
  }

  reset(): void {
    this.owner += 1
    this.controller?.abort()
    this.controller = undefined
    this.clearRetryTimer()
    this.commit({
      operation: null,
      outcome: null,
      lastError: undefined,
      retryReadyAt: 0,
      pending: false,
    })
  }

  async execute(
    send: (operation: StableImportOperation<TPayload>, signal: AbortSignal) => Promise<TResult>,
  ): Promise<MutationOutcome<TResult> | null> {
    const current = this.snapshot.operation
    if (!current || !this.snapshot.canRetry) return null

    const controller = new AbortController()
    this.controller = controller
    const executionOwner = ++this.owner
    this.commit({ pending: true, lastError: undefined })
    try {
      const value = await send(current, controller.signal)
      if (this.owner !== executionOwner || this.controller !== controller) return null
      const classified = classifyMutationOutcome<TResult>({ kind: 'success', value })
      this.commit({ outcome: classified })
      return classified
    } catch (error: unknown) {
      if (this.owner !== executionOwner || this.controller !== controller) return null
      const classified = classifyMutationOutcome<TResult>({
        kind: 'error',
        error,
        requestSent: true,
      })
      this.commit({ lastError: error, outcome: classified })
      if (classified.kind === 'failed' && classified.reason === 'retryable-precondition') {
        this.scheduleRetry(classified.retry_after_ms)
      }
      return classified
    } finally {
      if (this.owner === executionOwner && this.controller === controller) {
        this.controller = undefined
        this.commit({ pending: false })
      }
    }
  }

  dispose(): void {
    this.owner += 1
    this.controller?.abort()
    this.controller = undefined
    this.clearRetryTimer()
    if (this.snapshot.pending) this.commit({ pending: false })
  }
}

export interface CreateGroupImportOperationPayload {
  request: GroupCreateRequest
  draft: ImportDraft
}

export interface ImportCredentialsOperationPayload {
  groupID: number
  credentials: string
  draft: ImportRecoveryDraft
}

export interface ConnectCredentialsOperationPayload {
  groupID: number
  stageIDs: string[]
  draft: ImportDraft
}

export type ImportOperationMode = 'new' | 'existing'

export interface ImportOperationOwnerSnapshot {
  operationMode: ImportOperationMode | null
}

/**
 * Page-level owner for the three stable import operations. Classic keeps a
 * module singleton (survives mode switches and page remounts) whose clear() is
 * registered as an ephemeral-state cleaner — same shape here.
 */
export class ImportOperationOwnerStore {
  readonly createGroup = new StableOperationStore<
    CreateGroupImportOperationPayload,
    GroupCreateResult
  >()
  readonly importCredentials = new StableOperationStore<
    ImportCredentialsOperationPayload,
    CredentialImportResult
  >()
  readonly connectCredentials = new StableOperationStore<
    ConnectCredentialsOperationPayload,
    CredentialConnectResult
  >()

  private listeners = new Set<() => void>()
  private snapshot: ImportOperationOwnerSnapshot = { operationMode: null }
  private readonly unsubscribeOps: (() => void)[]

  constructor() {
    // Classic flush:'sync' watch: once every in-flight operation is gone the
    // mode lock releases so the page-level segmented control re-enables.
    const sync = (): void => {
      const locked =
        this.createGroup.getSnapshot().operation !== null ||
        this.importCredentials.getSnapshot().operation !== null ||
        this.connectCredentials.getSnapshot().operation !== null
      if (!locked && this.snapshot.operationMode !== null) {
        this.snapshot = { operationMode: null }
        for (const listener of this.listeners) listener()
      }
    }
    this.unsubscribeOps = [
      this.createGroup.subscribe(sync),
      this.importCredentials.subscribe(sync),
      this.connectCredentials.subscribe(sync),
    ]
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  getSnapshot = (): ImportOperationOwnerSnapshot => this.snapshot

  private setMode(mode: ImportOperationMode): void {
    this.snapshot = { operationMode: mode }
    for (const listener of this.listeners) listener()
  }

  beginCreate(
    request: GroupCreateRequest,
    draft: ImportDraft,
  ): StableImportOperation<CreateGroupImportOperationPayload> | null {
    if (
      this.importCredentials.getSnapshot().operation ||
      this.connectCredentials.getSnapshot().operation
    ) {
      return null
    }
    const operation = this.createGroup.begin({ request, draft })
    this.setMode('new')
    return operation
  }

  beginConnectCredentials(
    payload: { groupID: number; stageIDs: string[] },
    draft: ImportDraft,
  ): StableImportOperation<ConnectCredentialsOperationPayload> | null {
    if (
      this.createGroup.getSnapshot().operation ||
      this.importCredentials.getSnapshot().operation
    ) {
      return null
    }
    const operation = this.connectCredentials.begin({ ...payload, draft })
    this.setMode('new')
    return operation
  }

  beginImportCredentials(
    payload: { groupID: number; credentials: string },
    mode: ImportOperationMode,
    draft: ImportRecoveryDraft,
  ): StableImportOperation<ImportCredentialsOperationPayload> | null {
    if (this.createGroup.getSnapshot().operation) return null
    if (this.importCredentials.getSnapshot().operation && this.snapshot.operationMode !== mode) {
      return null
    }
    const operation = this.importCredentials.begin({ ...payload, draft })
    this.setMode(mode)
    return operation
  }

  clear(): void {
    this.createGroup.reset()
    this.importCredentials.reset()
    this.connectCredentials.reset()
    this.snapshot = { operationMode: null }
    for (const listener of this.listeners) listener()
  }

  dispose(): void {
    this.clear()
    for (const unsubscribe of this.unsubscribeOps) unsubscribe()
  }
}

// Classic parity: a module-level default owner so operations survive page
// remounts; the ephemeral-state cleaner wipes it when the session drops.
const defaultImportOperationOwner = new ImportOperationOwnerStore()
registerEphemeralStateCleaner(() => defaultImportOperationOwner.clear())

export function useImportOperationOwner(): ImportOperationOwnerStore {
  return defaultImportOperationOwner
}

export function useOperationSnapshot<TPayload, TResult>(
  store: StableOperationStore<TPayload, TResult>,
): OperationSnapshot<TPayload, TResult> {
  return useSyncExternalStore(store.subscribe, store.getSnapshot)
}

export function useImportOperationMode(): ImportOperationMode | null {
  const owner = useImportOperationOwner()
  return useSyncExternalStore(owner.subscribe, owner.getSnapshot).operationMode
}
