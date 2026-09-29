import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import type { CredentialItemDto } from '@shared/control/types'
import {
  cacheCredentialItem,
  revealCredential,
  updateCredential,
} from '@shared/control/resources/credentials'
import { invalidateGroupSettingsDependents } from '@shared/control/resources/groups'

import { useT } from '../../../app/i18n'
import { useAppServices } from '../../../app/services'
import { InlineNotice } from '../../../components/InlineNotice'
import { CopyChip } from '../../access-keys/AccessKeyCopyChip'

const small = '@media (max-width: 520px)'

const styles = stylex.create({
  root: {
    display: 'grid',
    minWidth: 0,
    gap: '7px',
  },
  summary: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '8px',
  },
  form: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: { default: 'end', [small]: 'stretch' },
    gap: '8px',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: '8px',
  },
  field: {
    display: 'grid',
    minWidth: 'min(260px, 100%)',
    flex: '1 1 280px',
    gap: '4px',
  },
  fieldLabel: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
  input: {
    width: '100%',
    minHeight: 'var(--control-sm)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingBlock: 0,
    paddingInline: '10px',
    font: 'inherit',
    fontFamily: 'var(--font-mono)',
  },
})

export function GroupApiKeyEditor({
  groupId,
  credential,
  disabled = false,
}: {
  groupId: number
  credential: CredentialItemDto
  disabled?: boolean
}) {
  const t = useT()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const copyControllersRef = useRef(new Set<AbortController>())
  const updateControllerRef = useRef<AbortController | undefined>(undefined)
  const [editing, setEditing] = useState(false)
  const [nextCredential, setNextCredential] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  const isAPIKey = credential.connection_type === 'api_key'

  async function resolveCopyValue(): Promise<string> {
    const controller = new AbortController()
    copyControllersRef.current.add(controller)
    try {
      const result = await revealCredential(
        apiClient,
        groupId,
        credential.credential_id,
        controller.signal,
      )
      const values = Object.values(result.credential)
      return values.length === 1 ? values[0]! : JSON.stringify(result.credential)
    } finally {
      copyControllersRef.current.delete(controller)
    }
  }

  function startEditing(): void {
    if (disabled || !isAPIKey) return
    setError('')
    setSaved(false)
    setEditing(true)
  }

  function stopEditing(): void {
    if (pending) return
    setEditing(false)
    setNextCredential('')
    setError('')
  }

  async function submit(): Promise<void> {
    const value = nextCredential.trim()
    if (pending || disabled || !isAPIKey) return
    if (!value) {
      setError(t('group.credentials.update.required'))
      setSaved(false)
      return
    }

    const controller = new AbortController()
    updateControllerRef.current = controller
    setPending(true)
    setError('')
    setSaved(false)
    try {
      const result = await updateCredential(
        apiClient,
        groupId,
        credential.credential_id,
        { credentials: value },
        controller.signal,
      )
      if (updateControllerRef.current !== controller) return
      setNextCredential('')
      try {
        await cacheCredentialItem(queryClient, groupId, result)
      } catch {
        setError(t('group.credentials.reconcileFailed'))
        return
      }
      setSaved(true)
      try {
        await invalidateGroupSettingsDependents(queryClient, groupId)
      } catch {
        setError(t('group.credentials.reconcileFailed'))
      }
    } catch {
      if (controller.signal.aborted || updateControllerRef.current !== controller) return
      setError(t('group.credentials.updateFailed'))
    } finally {
      if (updateControllerRef.current === controller) {
        updateControllerRef.current = undefined
        setPending(false)
      }
    }
  }

  // Classic watch([groupId, credential.secret_version]): reset editing state
  // when the row's identity changes. In-flight aborts are side effects, so they
  // run in an effect keyed on the same signature; the state resets are safe
  // render-time adjustments.
  const watchKey = `${groupId}:${credential.secret_version}`
  const [lastWatchKey, setLastWatchKey] = useState(watchKey)
  if (lastWatchKey !== watchKey) {
    setLastWatchKey(watchKey)
    setPending(false)
    setEditing(false)
    setNextCredential('')
    setSaved(false)
    setError('')
  }
  useEffect(() => {
    if (lastWatchKey === watchKey) return
    updateControllerRef.current?.abort()
    updateControllerRef.current = undefined
    for (const controller of copyControllersRef.current) controller.abort()
    copyControllersRef.current.clear()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the row signature
  }, [watchKey])

  useEffect(
    () => () => {
      updateControllerRef.current?.abort()
      for (const controller of copyControllersRef.current) controller.abort()
    },
    [],
  )

  if (!isAPIKey) return null

  return (
    <div {...stylex.props(styles.root)}>
      <div {...stylex.props(styles.summary)}>
        <CopyChip
          key={credential.secret_version}
          value={credential.mask}
          label={t('group.credentials.copy')}
          successLabel={t('common.copied')}
          failureLabel={t('common.copyFailed')}
          resolveValue={resolveCopyValue}
        />
        <Button
          variant="ghost"
          size="sm"
          isDisabled={disabled || pending}
          aria-expanded={editing}
          label={
            editing
              ? t('group.credentials.update.cancel')
              : t('group.credentials.update.action')
          }
          onClick={editing ? stopEditing : startEditing}
        />
      </div>
      {editing && (
        <form
          {...stylex.props(styles.form)}
          onSubmit={(event) => {
            event.preventDefault()
            void submit()
          }}
        >
          <label {...stylex.props(styles.field)}>
            <span {...stylex.props(styles.fieldLabel)}>
              {t('group.credentials.update.inputLabel')}
            </span>
            <input
              {...stylex.props(styles.input)}
              type="password"
              value={nextCredential}
              placeholder={t('group.credentials.update.placeholder')}
              disabled={pending}
              aria-invalid={error ? true : undefined}
              autoComplete="new-password"
              spellCheck={false}
              onChange={(event) => setNextCredential(event.target.value)}
            />
          </label>
          <Button
            type="submit"
            size="sm"
            isLoading={pending}
            isDisabled={!nextCredential.trim()}
            label={
              pending
                ? t('group.credentials.update.saving')
                : t('group.credentials.update.submit')
            }
          />
        </form>
      )}
      {error ? (
        <InlineNotice tone="danger" appearance="ledger">
          {error}
        </InlineNotice>
      ) : saved ? (
        <InlineNotice tone="success" appearance="ledger">
          {t('group.credentials.update.succeeded')}
        </InlineNotice>
      ) : null}
    </div>
  )
}
