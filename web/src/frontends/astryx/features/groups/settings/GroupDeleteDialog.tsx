import * as stylex from '@stylexjs/stylex'
import {
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
  TextInput,
} from '@astryxdesign/core'
import { useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import {
  clearGroupResourceCaches,
  deleteGroup,
  isGroupInUseData,
  type AccessKeyReferenceDto,
} from '@shared/control/resources/groups'
import { ApiError, RequestCancelledError } from '@shared/http/errors'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../../app/i18n'
import { useAppServices } from '../../../app/services'
import { InlineNotice } from '../../../components/InlineNotice'
import { plainTextInputAttrs } from '../../../components/input-attrs'
import { GroupInUseFeedback } from './GroupInUseFeedback'

const styles = stylex.create({
  body: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
})

/**
 * Typed-confirmation group delete — the Astryx counterpart of classic
 * GroupDeleteDialog. Owns its trigger button (classic trigger slot inverts to
 * this plain secondary button in the save bar).
 */
export function GroupDeleteDialog({
  groupId,
  groupName,
  disabled = false,
  onDeleted,
  onPendingChange,
}: {
  groupId: number
  groupName: string
  disabled?: boolean
  onDeleted?(): void
  onPendingChange?(pending: boolean): void
}) {
  const t = useT()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [typedName, setTypedName] = useState('')
  const nameInputRef = useRef<HTMLInputElement | null>(null)
  const [pending, setPending] = useState(false)
  const [genericError, setGenericError] = useState(false)
  const [references, setReferences] = useState<AccessKeyReferenceDto[]>([])
  const controllerRef = useRef<AbortController | undefined>(undefined)

  const confirmed = typedName === groupName

  useEffect(() => {
    onPendingChange?.(pending)
  }, [pending, onPendingChange])

  // Classic awaits two nextTicks before focusing the confirmation input.
  useEffect(() => {
    if (!open) return
    const frame = requestAnimationFrame(() => nameInputRef.current?.focus())
    return () => cancelAnimationFrame(frame)
  }, [open])

  useEffect(() => () => controllerRef.current?.abort(), [])

  function setDialogOpen(value: boolean): void {
    if ((pending || disabled) && !value) return
    if (disabled && value) return
    setOpen(value)
    if (value) {
      setGenericError(false)
      setReferences([])
    } else {
      setTypedName('')
    }
  }

  async function confirmDelete(): Promise<void> {
    if (!confirmed || pending || disabled) return
    setPending(true)
    setGenericError(false)
    setReferences([])
    const controller = new AbortController()
    controllerRef.current = controller
    try {
      await deleteGroup(apiClient, groupId, controller.signal)
      onDeleted?.()
      setOpen(false)
      setTypedName('')
      // Leave the deleted detail before cache removal or refetch can delay navigation.
      try {
        await navigate({ to: pagePath('groups'), replace: true, ignoreBlocker: true })
      } catch {
        window.location.replace(pagePath('groups'))
      }
      clearGroupResourceCaches(queryClient, groupId)
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.group.delete)
    } catch (error: unknown) {
      if (error instanceof RequestCancelledError) return
      if (
        error instanceof ApiError &&
        error.code === 'GROUP_IN_USE' &&
        isGroupInUseData(error.data)
      ) {
        setReferences(error.data.access_keys)
      } else {
        setGenericError(true)
      }
    } finally {
      if (controllerRef.current === controller) controllerRef.current = undefined
      setPending(false)
    }
  }

  return (
    <>
      <Button
        variant="secondary"
        size="sm"
        isDisabled={disabled}
        label={t('group.settings.delete.open')}
        onClick={() => setDialogOpen(true)}
      />
      <Dialog isOpen={open} onOpenChange={setDialogOpen} width={440}>
        <Layout
          header={
            <DialogHeader
              title={t('group.settings.delete.title')}
              subtitle={t('group.settings.delete.description', { name: groupName })}
              onOpenChange={setDialogOpen}
              hasDivider
            />
          }
          content={
            <LayoutContent isScrollable>
              <div {...stylex.props(styles.body)}>
                <TextInput
                  ref={nameInputRef}
                  id="group-delete-name"
                  label={t('group.settings.delete.typeName', { name: groupName })}
                  value={typedName}
                  onChange={setTypedName}
                  isDisabled={pending}
                  autoComplete="off"
                  {...plainTextInputAttrs}
                />
                {references.length > 0 ? (
                  <GroupInUseFeedback references={references} />
                ) : genericError ? (
                  <InlineNotice tone="danger">{t('group.settings.delete.failed')}</InlineNotice>
                ) : null}
              </div>
            </LayoutContent>
          }
          footer={
            <LayoutFooter hasDivider>
              <Button
                variant="secondary"
                size="sm"
                label={t('group.settings.delete.cancel')}
                isDisabled={pending}
                onClick={() => setDialogOpen(false)}
              />
              <Button
                variant="destructive"
                size="sm"
                label={t('group.settings.delete.confirm')}
                isDisabled={!confirmed || disabled}
                isLoading={pending}
                onClick={() => void confirmDelete()}
              />
            </LayoutFooter>
          }
        />
      </Dialog>
    </>
  )
}
