import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import type { RefObject } from 'react'

import { useT } from '../../app/i18n'
import { StickySaveBar } from '../../components/StickySaveBar'
import { GroupDeleteDialog } from './settings/GroupDeleteDialog'
import type { GroupEditorHandle, GroupEditorState, GroupModelsEditorHandle } from './editor-handles'

export interface GroupUnifiedSaveBarProps {
  groupId: number
  groupName: string
  settingsState: GroupEditorState
  modelsState: GroupEditorState
  deletePending: boolean
  settingsEditorRef: RefObject<GroupEditorHandle | null>
  modelsEditorRef: RefObject<GroupModelsEditorHandle | null>
  onDeletePendingChange(pending: boolean): void
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
}: GroupUnifiedSaveBarProps) {
  const t = useT()

  const dirty = settingsState.dirty || modelsState.dirty
  const pending = settingsState.pending || modelsState.pending || deletePending
  const invalid = settingsState.invalid === true || (modelsState.invalidRowCount ?? 0) > 0
  const error = settingsState.error || modelsState.error
  const saved = !dirty && (settingsState.saved || modelsState.saved)
  const status: 'idle' | 'saved' | 'error' = error ? 'error' : saved ? 'saved' : 'idle'
  const errorActionLabel = modelsState.invalidRowCount
    ? t('group.modelEditor.locateFirstInvalid')
    : ''

  function saveUnified(): void {
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
            onClick={saveUnified}
          />
        </>
      }
    />
  )
}
