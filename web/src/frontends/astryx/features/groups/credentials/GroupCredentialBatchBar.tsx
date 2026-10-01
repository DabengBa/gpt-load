import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import { ListChecks, RefreshCw, Trash2 } from 'lucide-react'

import { useT } from '../../../app/i18n'

const styles = stylex.create({
  root: {
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
    gap: '6px',
  },
})

export function GroupCredentialBatchBar({
  selectedCount,
  allVisibleSelected,
  canSelectAll,
  pending = false,
  canSync = false,
  canDownload = false,
  onToggleSelect,
  onSync,
  onDownload,
  onRemove,
}: {
  selectedCount: number
  allVisibleSelected: boolean
  canSelectAll: boolean
  pending?: boolean
  canSync?: boolean
  canDownload?: boolean
  onToggleSelect(): void
  onSync(): void
  onDownload(): void
  onRemove(): void
}) {
  const t = useT()
  return (
    <div {...stylex.props(styles.root)}>
      <Button
        variant="secondary"
        size="sm"
        isLoading={pending}
        isDisabled={!canSelectAll}
        icon={<ListChecks size={15} aria-hidden="true" />}
        label={
          allVisibleSelected
            ? `${t('group.credentials.batch.clearAll')}${selectedCount > 0 ? ` ${selectedCount}` : ''}`
            : `${t('group.credentials.batch.selectAll')}${selectedCount > 0 ? ` ${selectedCount}` : ''}`
        }
        onClick={onToggleSelect}
      />
      {canSync && (
        <Button
          variant="secondary"
          size="sm"
          isLoading={pending}
          isDisabled={selectedCount === 0}
          icon={<RefreshCw size={15} aria-hidden="true" />}
          label={t('group.credentials.batch.sync')}
          onClick={onSync}
        />
      )}
      {canDownload && (
        <Button
          variant="secondary"
          size="sm"
          isLoading={pending}
          isDisabled={selectedCount === 0}
          label={t('group.credentials.batch.download')}
          onClick={onDownload}
        />
      )}
      <Button
        variant="secondary"
        size="sm"
        isLoading={pending}
        isDisabled={selectedCount === 0}
        icon={<Trash2 size={15} aria-hidden="true" />}
        label={t('group.credentials.batch.delete')}
        onClick={onRemove}
      />
    </div>
  )
}
