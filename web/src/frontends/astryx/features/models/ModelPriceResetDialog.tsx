import * as stylex from '@stylexjs/stylex'
import {
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
} from '@astryxdesign/core'
import { useQueryClient } from '@tanstack/react-query'
import { RotateCcw, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import {
  deleteModelPrice,
  projectModelPriceMutationIssue,
  resetModelPrice,
  type ModelPriceDto,
} from '@shared/control/resources/model-prices'
import { RequestCancelledError } from '@shared/http/errors'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'

const styles = stylex.create({
  feedbackStack: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  warning: {
    borderRadius: 'var(--radius-control, 6px)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    backgroundColor:
      'var(--color-warning-bg, color-mix(in srgb, var(--color-warning) 12%, transparent))',
    padding: '9px 12px',
    fontSize: 'var(--text-meta)',
  },
  danger: {
    borderRadius: 'var(--radius-control, 6px)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-danger)',
    backgroundColor:
      'var(--color-danger-bg, color-mix(in srgb, var(--color-danger) 10%, transparent))',
    padding: '9px 12px',
    fontSize: 'var(--text-meta)',
  },
})

export function ModelPriceResetDialog({
  row,
  action,
  disabled = false,
  onCompleted,
  onPendingChange,
}: {
  row: ModelPriceDto
  action: 'reset' | 'delete'
  disabled?: boolean
  onCompleted: () => void
  onPendingChange?: (pending: boolean) => void
}) {
  const t = useT()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [pending, setPending] = useState(false)
  const [failure, setFailure] = useState('')
  const requestRef = useRef<AbortController | undefined>(undefined)

  useEffect(() => {
    onPendingChange?.(pending)
  }, [pending, onPendingChange])

  useEffect(
    () => () => {
      requestRef.current?.abort()
      requestRef.current = undefined
    },
    [],
  )

  function setDialogOpen(value: boolean): void {
    if (!value && pending) return
    if (!value) {
      requestRef.current?.abort()
      requestRef.current = undefined
      setPending(false)
      setFailure('')
    }
    setOpen(value)
  }

  function failureMessage(error: unknown): string {
    try {
      const issue = projectModelPriceMutationIssue(error)
      if (issue?.code === 'MODEL_PRICE_REFERENCED') {
        return t('modelPrices.errors.referenced', {
          entries: issue.reference_count,
          groups: issue.reference_group_count,
        })
      }
      if (issue?.code === 'MODEL_PRICE_AUTOMATIC_DELETE_FORBIDDEN') {
        return t('modelPrices.errors.automaticDeleteForbidden')
      }
    } catch {
      return t(`modelPrices.${action}.failed`)
    }
    return t(`modelPrices.${action}.failed`)
  }

  async function confirm(): Promise<void> {
    if (pending) return
    setPending(true)
    setFailure('')
    const controller = new AbortController()
    requestRef.current = controller
    try {
      if (action === 'reset') {
        await resetModelPrice(apiClient, row.id, controller.signal)
      } else {
        await deleteModelPrice(apiClient, row.id, controller.signal)
      }
      if (requestRef.current !== controller) return
      await applyInvalidationPlan(
        queryClient,
        action === 'reset'
          ? mutationInvalidationPlans.modelPrice.reset
          : mutationInvalidationPlans.modelPrice.delete,
      )
      if (requestRef.current !== controller) return
      setOpen(false)
      onCompleted()
    } catch (error: unknown) {
      if (
        requestRef.current === controller &&
        !controller.signal.aborted &&
        !(error instanceof RequestCancelledError)
      ) {
        setFailure(failureMessage(error))
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = undefined
        setPending(false)
      }
    }
  }

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        isDisabled={disabled}
        aria-label={t(`modelPrices.${action}.open`, { model: row.model_id })}
        icon={
          action === 'reset' ? (
            <RotateCcw size={14} aria-hidden />
          ) : (
            <Trash2 size={14} aria-hidden />
          )
        }
        label={t(`modelPrices.${action}.confirm`)}
        onClick={() => setDialogOpen(true)}
      />
      <Dialog isOpen={open} onOpenChange={setDialogOpen} width={440}>
        <Layout
          header={
            <DialogHeader
              title={t(`modelPrices.${action}.title`)}
              subtitle={t(`modelPrices.${action}.description`, { model: row.model_id })}
              onOpenChange={setDialogOpen}
              hasDivider
            />
          }
          content={
            <LayoutContent>
              <div {...stylex.props(styles.feedbackStack)}>
                <div {...stylex.props(styles.warning)} role="status">
                  {t(`modelPrices.${action}.warning`)}
                </div>
                {failure && (
                  <div {...stylex.props(styles.danger)} role="alert">
                    {failure}
                  </div>
                )}
              </div>
            </LayoutContent>
          }
          footer={
            <LayoutFooter hasDivider>
              <Button
                variant="secondary"
                size="sm"
                label={t('common.cancel')}
                onClick={() => setDialogOpen(false)}
              />
              <Button
                variant={action === 'delete' ? 'destructive' : 'primary'}
                size="sm"
                isLoading={pending}
                label={t(`modelPrices.${action}.confirm`)}
                onClick={() => void confirm()}
              />
            </LayoutFooter>
          }
        />
      </Dialog>
    </>
  )
}
