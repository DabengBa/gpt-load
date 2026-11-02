import { Button } from '@astryxdesign/core'

import { useT } from '../../../app/i18n'
import { StickySaveBar } from '../../../components/StickySaveBar'
import { GroupDeleteDialog } from './GroupDeleteDialog'

export function GroupSettingsSaveBar({
  groupId,
  groupName,
  dirty,
  pending,
  blocked,
  error,
  savedFeedback,
  valid,
  deletePending,
  deleted,
  onDeletePendingChange,
  onDeleted,
  onDiscard,
  onSave,
}: {
  groupId: number
  groupName: string
  dirty: boolean
  pending: boolean
  blocked: boolean
  error: string
  savedFeedback: boolean
  valid: boolean
  deletePending: boolean
  deleted: boolean
  onDeletePendingChange(pending: boolean): void
  onDeleted(): void
  onDiscard(): void
  onSave(): void
}) {
  const t = useT()
  const disabled = blocked || !dirty || deletePending

  return (
    <StickySaveBar
      appearance="ledger"
      alwaysVisible
      dirty={dirty}
      pending={blocked}
      status={error ? 'error' : savedFeedback ? 'saved' : 'idle'}
      error={error}
      statusContent={
        <div>
          <strong>
            {pending
              ? t('group.settings.saving')
              : savedFeedback
                ? t('group.settings.savedFeedback')
                : dirty
                  ? t('group.settings.unsaved')
                  : t('group.settings.saved')}
          </strong>
          <span>
            {pending
              ? t('group.settings.savingNote')
              : savedFeedback
                ? t('group.settings.savedFeedbackNote')
                : dirty
                  ? t('group.settings.dirtyNote')
                  : t('group.settings.saveNote')}
          </span>
        </div>
      }
      actions={
        <>
          <Button
            variant="ghost"
            size="sm"
            isDisabled={disabled}
            onClick={onDiscard}
            label={t('common.discard')}
          />
          <GroupDeleteDialog
            groupId={groupId}
            groupName={groupName}
            disabled={blocked || dirty || deleted}
            onPendingChange={onDeletePendingChange}
            onDeleted={onDeleted}
          />
          <Button
            size="sm"
            isDisabled={disabled || !valid}
            onClick={onSave}
            label={t('group.settings.save')}
          />
        </>
      }
    />
  )
}
