import * as stylex from '@stylexjs/stylex'
import {
  Banner,
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
} from '@astryxdesign/core'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { classifyMutationOutcome } from '@shared/control/mutation-outcome'
import {
  accessKeyResources,
  revealAccessKey,
  rotateAccessKey,
} from '@shared/control/resources/access-keys'
import type { AccessKeyDto, AccessKeyRotateResultDto } from '@shared/control/types'
import { RequestCancelledError } from '@shared/http/errors'
import { createUUID } from '@shared/lib/uuid'
import type { MessageId } from '@shared/i18n/message-ids'
import { findAccessKeyForReconciliation } from '@shared/domain/access-keys/access-key-edit-operation'
import type { PendingAccessKeyRotateOperation } from '@shared/domain/access-keys/access-key-rotate-operation'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { CopyChip } from './AccessKeyCopyChip'

const styles = stylex.create({
  body: {
    display: 'grid',
    gap: 10,
  },
  result: {
    display: 'grid',
    gap: 10,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: 10,
    paddingInline: 12,
  },
  resultLabel: {
    fontSize: 'var(--text-sm)',
  },
  resultHint: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
})

type RotateOperationState = 'idle' | 'indeterminate' | 'reconciling'

export function AccessKeyRotateDialog({
  accessKey,
  open,
  onOpenChange,
  operation = null,
  onRotated,
  onPendingChange,
  onOperationChange,
}: {
  accessKey: AccessKeyDto
  open: boolean
  onOpenChange(open: boolean): void
  operation?: PendingAccessKeyRotateOperation | null
  onRotated(accessKey: AccessKeyDto): void
  onPendingChange(pending: boolean): void
  onOperationChange(operation: PendingAccessKeyRotateOperation | null): void
}) {
  const t = useT()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const [pending, setPending] = useState(false)
  const [failed, setFailed] = useState(false)
  const [refreshFailed, setRefreshFailed] = useState(false)
  const [operationState, setOperationState] = useState<RotateOperationState>(
    operation?.state ?? 'idle',
  )
  const [operationID, setOperationID] = useState(operation?.idempotencyKey ?? createUUID())
  const [result, setResult] = useState<AccessKeyRotateResultDto | null>(null)
  const [current, setCurrent] = useState<AccessKeyDto | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  // Mirror the classic watch on props.operation: an externally surfaced
  // pending operation adopts the dialog's idempotency key/state; clearing it
  // while idle resets to a fresh key. Render-phase adjustment, not an effect —
  // the state must be consistent in the same commit that renders the dialog.
  const [lastOperation, setLastOperation] = useState(operation ?? null)
  if (lastOperation !== (operation ?? null)) {
    setLastOperation(operation ?? null)
    if (!pending && !result) {
      if (operation) {
        setOperationID(operation.idempotencyKey)
        setOperationState(operation.state)
      } else if (operationState !== 'idle') {
        setOperationID(createUUID())
        setOperationState('idle')
      }
    }
  }

  // Classic `watch(pending)` → emit only on transitions. The callback goes
  // through a ref so the effect depends on `pending` alone — the parent's
  // inline `setRotationPending` changes identity every commit and would
  // otherwise re-fire notify → forceStoreRerender in a loop.
  const onPendingChangeRef = useRef(onPendingChange)
  useEffect(() => {
    onPendingChangeRef.current = onPendingChange
  })
  useEffect(() => {
    onPendingChangeRef.current(pending)
  }, [pending])

  useEffect(
    () => () => {
      controllerRef.current?.abort()
    },
    [],
  )

  const feedbackKey =
    operationState === 'reconciling'
      ? 'accessKeys.rotate.reconciling'
      : operationState === 'indeterminate'
        ? 'accessKeys.rotate.indeterminate'
        : ''
  const confirmLabel = result
    ? t('accessKeys.rotate.done')
    : operationState !== 'idle'
      ? t('accessKeys.rotate.checkResult')
      : t('accessKeys.rotate.confirm')
  const displayKey = result?.key ?? current?.masked_key ?? accessKey.masked_key

  const resetOperation = () => {
    controllerRef.current?.abort()
    controllerRef.current = null
    setPending(false)
    setFailed(false)
    setRefreshFailed(false)
    setOperationState('idle')
    setOperationID(createUUID())
    setResult(null)
    setCurrent(null)
    onOperationChange(null)
  }

  const refreshQueries = async () => {
    try {
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.accessKey.rotate)
    } catch {
      void queryClient.invalidateQueries({ queryKey: accessKeyResources.collection.queryKey })
    }
  }

  const setDialogOpen = (value: boolean) => {
    if (!value && pending) return
    const refreshAfterClose = !value && result !== null
    if (refreshAfterClose) resetOperation()
    onOpenChange(value)
    if (refreshAfterClose) void refreshQueries()
  }

  const refreshCurrent = async (signal: AbortSignal): Promise<AccessKeyDto | null> => {
    try {
      return (await findAccessKeyForReconciliation(apiClient, accessKey.id, signal)) ?? null
    } catch {
      setRefreshFailed(true)
      return null
    }
  }

  const resolveCurrentKey = async (): Promise<string> => {
    const value = await revealAccessKey(apiClient, accessKey.id)
    return value.key
  }

  const completeRotation = async (
    rotation: AccessKeyRotateResultDto,
    signal: AbortSignal,
  ): Promise<void> => {
    setResult(rotation)
    let resolved: AccessKeyDto = rotation
    if (rotation.replayed) {
      resolved = (await refreshCurrent(signal)) ?? rotation
    }
    setCurrent(resolved)
    onRotated(resolved)
    onOperationChange(null)
    setOperationState('idle')
  }

  const confirmRotate = async () => {
    if (pending) return
    if (result) {
      setDialogOpen(false)
      return
    }
    setPending(true)
    setFailed(false)
    setRefreshFailed(false)
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    const stale = () => controllerRef.current !== controller || !open
    try {
      const rotation = await rotateAccessKey(
        apiClient,
        accessKey.id,
        operationID,
        controller.signal,
      )
      if (stale()) return
      await completeRotation(rotation, controller.signal)
    } catch (error: unknown) {
      if (stale() || error instanceof RequestCancelledError) return
      const outcome = classifyMutationOutcome({ kind: 'error', error, requestSent: true })
      if (outcome.kind === 'indeterminate' || outcome.kind === 'reconciling') {
        setOperationState(outcome.kind)
        onOperationChange({
          base: accessKey,
          idempotencyKey: operationID,
          state: outcome.kind,
        })
      } else if (outcome.kind === 'failed' && outcome.reason === 'expired-known') {
        const latest = await refreshCurrent(controller.signal)
        if (latest) {
          setCurrent(latest)
          setResult({ ...latest, replayed: true })
          onRotated(latest)
          onOperationChange(null)
          setOperationState('idle')
        } else {
          setOperationState('indeterminate')
          onOperationChange({
            base: accessKey,
            idempotencyKey: operationID,
            state: 'indeterminate',
          })
        }
      } else if (outcome.kind === 'failed' && outcome.reason === 'retryable-precondition') {
        setOperationState('reconciling')
        onOperationChange({
          base: accessKey,
          idempotencyKey: operationID,
          state: 'reconciling',
        })
      } else {
        setFailed(true)
        if (outcome.kind === 'failed' && outcome.reason === 'rejected') {
          setOperationID(createUUID())
          setOperationState('idle')
          onOperationChange(null)
        }
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setPending(false)
      }
    }
  }

  return (
    <Dialog isOpen={open} onOpenChange={setDialogOpen} width={440}>
      <Layout
        header={
          <DialogHeader
            title={t('accessKeys.rotate.title')}
            subtitle={t('accessKeys.rotate.description', { name: accessKey.name })}
            onOpenChange={setDialogOpen}
            hasDivider
          />
        }
        content={
          <LayoutContent isScrollable>
            <div {...stylex.props(styles.body)}>
              <Banner status="warning" title={t('accessKeys.rotate.impact')} />
              {feedbackKey !== '' && (
                <Banner status="warning" title={t(feedbackKey as MessageId)} />
              )}
              {failed && <Banner status="error" title={t('accessKeys.rotate.failed')} />}
              {refreshFailed && (
                <Banner status="warning" title={t('accessKeys.rotate.refreshFailed')} />
              )}
              {result && (
                <div {...stylex.props(styles.result)} aria-live="polite">
                  <strong {...stylex.props(styles.resultLabel)}>
                    {t(result.key ? 'accessKeys.rotate.newKey' : 'accessKeys.rotate.currentKey')}
                  </strong>
                  {open && (
                    <CopyChip
                      key={`${accessKey.id}:${result.updated_at_ms}`}
                      value={displayKey}
                      label={t('accessKeys.copy')}
                      successLabel={t('common.copied')}
                      failureLabel={t('common.copyFailed')}
                      resolveValue={result.key ? undefined : resolveCurrentKey}
                    />
                  )}
                  <small {...stylex.props(styles.resultHint)}>
                    {t(
                      result.key
                        ? 'accessKeys.rotate.newKeyHint'
                        : 'accessKeys.rotate.replayedHint',
                    )}
                  </small>
                </div>
              )}
            </div>
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            <Button
              variant="secondary"
              label={result ? t('common.close') : t('common.cancel')}
              isDisabled={pending}
              onClick={() => setDialogOpen(false)}
            />
            <Button
              variant="destructive"
              label={confirmLabel}
              isLoading={pending}
              onClick={() => void confirmRotate()}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}
