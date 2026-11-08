import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import type { RefObject } from 'react'

import { useT } from '../../app/i18n'
import { StickySaveBar } from '../../components/StickySaveBar'
import { GroupDeleteDialog } from './settings/GroupDeleteDialog'
import type {
  GroupSettingsEditorHandle,
  GroupEditorState,
  GroupModelsEditorHandle,
} from './editor-handles'
import { getGroupModels, updateGroupConfiguration } from '@shared/control/resources/groups'
import { useAppServices } from '../../app/services'
import { ApiError } from '@shared/http/errors'

export interface GroupUnifiedSaveBarProps {
  groupId: number
  groupName: string
  settingsState: GroupEditorState
  modelsState: GroupEditorState
  deletePending: boolean
  settingsEditorRef: RefObject<GroupSettingsEditorHandle | null>
  modelsEditorRef: RefObject<GroupModelsEditorHandle | null>
  onDeletePendingChange(pending: boolean): void
  combinedPending: boolean
  onCombinedPendingChange(pending: boolean): void
}

const styles = stylex.create({
  saveBar: {
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-raised)',
    backdropFilter: 'none',
    boxShadow: '0 2px 8px light-dark(rgba(0, 0, 0, 0.08), rgba(0, 0, 0, 0.28))',
  },
  statusText: {
    display: 'grid',
    minWidth: 0,
    fontSize: { default: null, '@media (max-width: 680px)': 'var(--text-sm)' },
  },
  statusNote: {
    display: { default: 'inline', '@media (max-width: 680px)': 'none' },
  },
})

/**
 * Unified-mode save bar: the single place that aggregates the settings and
 * models editors into one dirty/pending/invalid verdict and fans the actions
 * back out through their imperative handles. GroupDetailView keeps layout and
 * routing; this keeps the aggregation policy.
 */
export function GroupUnifiedSaveBar({
  groupId,
  groupName,
  settingsState,
  modelsState,
  deletePending,
  settingsEditorRef,
  modelsEditorRef,
  onDeletePendingChange,
  combinedPending,
  onCombinedPendingChange,
}: GroupUnifiedSaveBarProps) {
  const t = useT()
  const { apiClient } = useAppServices()

  const dirty = settingsState.dirty || modelsState.dirty
  const pending = settingsState.pending || modelsState.pending || deletePending || combinedPending
  const invalid =
    settingsState.invalid === true ||
    modelsState.invalid === true ||
    (modelsState.invalidRowCount ?? 0) > 0
  const error = settingsState.error || modelsState.error
  const saved = !dirty && (settingsState.saved || modelsState.saved)
  const status: 'idle' | 'saved' | 'error' = error ? 'error' : saved ? 'saved' : 'idle'
  const errorActionLabel = modelsState.invalidRowCount
    ? t('group.modelEditor.locateFirstInvalid')
    : ''

  function saveUnified(confirmedEmpty = false): void {
    if (settingsState.dirty && modelsState.dirty) {
      const settingsEditor = settingsEditorRef.current
      const modelsEditor = modelsEditorRef.current
      const settings = settingsEditor?.prepareCombined()
      const models = modelsEditor?.prepareCombined()
      if (!settingsEditor || !modelsEditor || !settings || !models) return
      if (models.length === 0 && !confirmedEmpty) {
        modelsEditor.confirmCombinedEmpty(() => saveUnified(true))
        return
      }
      onCombinedPendingChange(true)
      void (async () => {
        try {
          const result = await updateGroupConfiguration(apiClient, groupId, settings, models)
          await settingsEditor.acceptCombined(result)
          const updatedModels = await getGroupModels(apiClient, groupId)
          await modelsEditor.acceptCombined(updatedModels)
        } catch (cause) {
          settingsEditor.failCombined(cause)
          // This validation response is known to precede any transaction commit.
          if (!(
            cause instanceof ApiError && cause.code === 'SITE_ROOT_CHANGE_CONFIRMATION_REQUIRED'
          )) {
            await Promise.all([settingsEditor.recoverCombined(), modelsEditor.recoverCombined()])
          }
        } finally {
          onCombinedPendingChange(false)
        }
      })()
      return
    }
    if (settingsState.dirty) settingsEditorRef.current?.requestSave()
    if (modelsState.dirty) modelsEditorRef.current?.requestSave()
  }

  function discardUnified(): void {
    if (settingsState.dirty) settingsEditorRef.current?.discard()
    if (modelsState.dirty) modelsEditorRef.current?.discard()
  }

  return (
    <StickySaveBar
      appearance="ledger"
      xstyle={styles.saveBar}
      alwaysVisible
      compactOnMobile
      dirty={dirty}
      pending={pending}
      status={status}
      error={error}
      errorActionLabel={errorActionLabel}
      onErrorAction={() => void modelsEditorRef.current?.focusFirstInvalid()}
      statusContent={
        <div {...stylex.props(styles.statusText)}>
          <strong>
            {pending
              ? t('group.settings.saving')
              : saved
                ? t('group.settings.savedFeedback')
                : dirty
                  ? t('group.settings.unsaved')
                  : t('group.settings.saved')}
          </strong>
          {error ? (
            <span>{error}</span>
          ) : (
            <span {...stylex.props(styles.statusNote)}>{t('group.settings.saveNote')}</span>
          )}
        </div>
      }
      actions={
        <>
          <Button
            variant="ghost"
            size="sm"
            label={t('common.discard')}
            isDisabled={pending || !dirty}
            onClick={discardUnified}
          />
          <GroupDeleteDialog
            groupId={groupId}
            groupName={groupName}
            disabled={pending || dirty}
            onPendingChange={onDeletePendingChange}
          />
          <Button
            size="sm"
            label={t('group.settings.save')}
            isDisabled={pending || !dirty || invalid}
            onClick={() => saveUnified()}
          />
        </>
      }
    />
  )
}
