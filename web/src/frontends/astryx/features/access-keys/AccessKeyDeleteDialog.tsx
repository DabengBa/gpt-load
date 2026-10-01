import * as stylex from '@stylexjs/stylex'
import {
  Banner,
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
  TextInput,
} from '@astryxdesign/core'
import { useEffect, useRef, useState } from 'react'

import { deleteAccessKey } from '@shared/control/resources/access-keys'
import type { AccessKeyDto } from '@shared/control/types'
import { RequestCancelledError } from '@shared/http/errors'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { plainTextInputAttrs } from '../../components/input-attrs'

const styles = stylex.create({
  body: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  summary: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: 9,
    paddingInline: 10,
  },
  summaryName: {
    fontSize: 'var(--text-sm)',
  },
  summaryKey: {
    minWidth: 0,
    color: 'var(--color-code)',
    fontSize: 'var(--text-label-xs)',
    overflowWrap: 'anywhere',
  },
})

/**
 * Typed-confirmation delete — the Astryx counterpart of classic
 * AccessKeyDeleteDialog. `open` is controlled by the caller (the classic
 * trigger slot inverts to a plain button in the row actions).
 */
export function AccessKeyDeleteDialog({
  accessKey,
  total,
  open,
  onOpenChange,
  onDeleted,
}: {
  accessKey: AccessKeyDto
  total: number
  open: boolean
  onOpenChange(open: boolean): void
  onDeleted(name: string): void
}) {
  const t = useT()
  const { apiClient } = useAppServices()
  const [typedName, setTypedName] = useState('')
  const [pending, setPending] = useState(false)
  const [failed, setFailed] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)
  const nameInputRef = useRef<HTMLInputElement | null>(null)

  const confirmed = typedName === accessKey.name

  const setDialogOpen = (value: boolean) => {
    if (!value && pending) return
    if (!value) {
      controllerRef.current?.abort()
      controllerRef.current = null
      setFailed(false)
      setTypedName('')
    }
    onOpenChange(value)
  }

  // Classic awaits two nextTicks before focusing the confirmation input — the
  // rAF lands after the <dialog> mounts and its own focus-on-open settles.
  useEffect(() => {
    if (!open) return
    const frame = requestAnimationFrame(() => nameInputRef.current?.focus())
    return () => cancelAnimationFrame(frame)
  }, [open])

  useEffect(
    () => () => {
      controllerRef.current?.abort()
    },
    [],
  )

  const confirmDelete = async () => {
    if (!confirmed || pending) return
    setPending(true)
    setFailed(false)
    const controller = new AbortController()
    controllerRef.current = controller
    try {
      await deleteAccessKey(apiClient, accessKey.id, controller.signal)
      onOpenChange(false)
      setTypedName('')
      onDeleted(accessKey.name)
    } catch (error: unknown) {
      if (!(error instanceof RequestCancelledError)) setFailed(true)
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
      setPending(false)
    }
  }

  return (
    <Dialog isOpen={open} onOpenChange={setDialogOpen} width={440}>
      <Layout
        header={
          <DialogHeader
            title={t('accessKeys.delete.title')}
            subtitle={t('accessKeys.delete.description', { name: accessKey.name })}
            onOpenChange={setDialogOpen}
            hasDivider
          />
        }
        content={
          <LayoutContent isScrollable>
            <div {...stylex.props(styles.body)}>
              <div {...stylex.props(styles.summary)}>
                <strong {...stylex.props(styles.summaryName)}>{accessKey.name}</strong>
                <code {...stylex.props(styles.summaryKey)}>{accessKey.masked_key}</code>
              </div>
              {total === 1 && (
                <Banner status="warning" title={t('accessKeys.delete.lastWarning')} />
              )}
              <Banner status="warning" title={t('accessKeys.delete.impact')} />
              <TextInput
                ref={nameInputRef}
                label={t('accessKeys.delete.typeName', { name: accessKey.name })}
                value={typedName}
                onChange={setTypedName}
                isDisabled={pending}
                autoComplete="off"
                {...plainTextInputAttrs}
              />
              {failed && <Banner status="error" title={t('accessKeys.delete.failed')} />}
            </div>
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            <Button
              variant="secondary"
              label={t('common.cancel')}
              isDisabled={pending}
              onClick={() => setDialogOpen(false)}
            />
            <Button
              variant="destructive"
              label={t('accessKeys.delete.confirm')}
              isDisabled={!confirmed}
              isLoading={pending}
              onClick={() => void confirmDelete()}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}
