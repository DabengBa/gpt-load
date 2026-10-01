import {
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
  SegmentedControl,
  SegmentedControlItem,
} from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'

import type { ModelCandidate } from '@shared/control/resources/providers'
import type {
  ModelDraftItem,
  ModelNameConflict,
  ModelSyncMode,
} from '@shared/domain/groups/models/model-diff'
import { useT } from '../../../app/i18n'
import type { MessageId } from '@shared/i18n/message-ids'
import { InlineNotice } from '../../../components/InlineNotice'

const previewLimit = 40

/**
 * Classic features/groups/models/GroupModelSyncDialog.vue — preview + mode
 * picker for reconciling the saved model list with discovered candidates.
 */
export function GroupModelSyncDialog({
  open,
  mode,
  additions,
  removals,
  conflicts,
  pending,
  error,
  onOpenChange,
  onModeChange,
  onConfirm,
}: {
  open: boolean
  mode: ModelSyncMode
  additions: readonly ModelCandidate[]
  removals: readonly ModelDraftItem[]
  conflicts: readonly ModelNameConflict[]
  pending: boolean
  error: string
  onOpenChange(open: boolean): void
  onModeChange(mode: ModelSyncMode): void
  onConfirm(): void
}) {
  const t = useT()

  const showAdditions = mode === 'add' || mode === 'full'
  const showRemovals = mode === 'cleanup' || mode === 'full'
  const activeAdditions = showAdditions ? additions : []
  const activeRemovals = showRemovals ? removals : []
  const additionPreview = activeAdditions.slice(0, previewLimit)
  const removalPreview = activeRemovals.slice(0, previewLimit)
  const additionRemainder = activeAdditions.length - additionPreview.length
  const removalRemainder = activeRemovals.length - removalPreview.length
  const changeCount = activeAdditions.length + activeRemovals.length
  const conflictNames = conflicts.map(({ client_model }) => client_model).join(', ')

  function setMode(value: string): void {
    if (value === 'cleanup' || value === 'add' || value === 'full') onModeChange(value)
  }

  return (
    <Dialog isOpen={open} onOpenChange={onOpenChange} width={520}>
      <Layout
        header={
          <DialogHeader
            title={t('group.modelEditor.sync.title')}
            subtitle={t('group.modelEditor.sync.description')}
            onOpenChange={onOpenChange}
            hasDivider
          />
        }
        content={
          <LayoutContent>
            <div {...stylex.props(styles.root)}>
              <SegmentedControl
                xstyle={styles.modes}
                value={mode}
                label={t('group.modelEditor.sync.modeLabel')}
                size="sm"
                isDisabled={pending}
                onChange={setMode}
              >
                <SegmentedControlItem
                  value="full"
                  label={t('group.modelEditor.sync.mode.full' as MessageId)}
                />
                <SegmentedControlItem
                  value="cleanup"
                  label={t('group.modelEditor.sync.mode.cleanup' as MessageId)}
                />
                <SegmentedControlItem
                  value="add"
                  label={t('group.modelEditor.sync.mode.add' as MessageId)}
                />
              </SegmentedControl>

              <div
                id="group-model-sync-changes"
                {...stylex.props(styles.changes)}
                role="tabpanel"
                aria-label={t('group.modelEditor.sync.modeLabel')}
              >
                {showAdditions && (
                  <section {...stylex.props(styles.section, styles.sectionAddition)}>
                    <strong {...stylex.props(styles.sectionTitleAddition)}>
                      {t('group.modelEditor.sync.additions', {
                        count: activeAdditions.length,
                      })}
                    </strong>
                    {additionPreview.length ? (
                      <ul {...stylex.props(styles.list)}>
                        {additionPreview.map((candidate) => (
                          <li key={candidate.id} {...stylex.props(styles.listItem)}>
                            <code {...stylex.props(styles.code)}>{candidate.id}</code>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <span {...stylex.props(styles.empty)}>
                        {t('group.modelEditor.sync.noAdditions')}
                      </span>
                    )}
                    {additionRemainder > 0 && (
                      <span {...stylex.props(styles.remainder)}>
                        {t('group.modelEditor.sync.more', { count: additionRemainder })}
                      </span>
                    )}
                  </section>
                )}

                {showRemovals && (
                  <section {...stylex.props(styles.section, styles.sectionRemoval)}>
                    <strong {...stylex.props(styles.sectionTitleRemoval)}>
                      {t('group.modelEditor.sync.removals', {
                        count: activeRemovals.length,
                      })}
                    </strong>
                    {removalPreview.length ? (
                      <ul {...stylex.props(styles.list)}>
                        {removalPreview.map((model) => (
                          <li key={model.key} {...stylex.props(styles.listItem)}>
                            <code {...stylex.props(styles.code)}>{model.id}</code>
                            {model.alias_enabled && (
                              <>
                                <span aria-hidden="true">→</span>
                                <code {...stylex.props(styles.code)}>{model.alias}</code>
                              </>
                            )}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <span {...stylex.props(styles.empty)}>
                        {t('group.modelEditor.sync.noRemovals')}
                      </span>
                    )}
                    {removalRemainder > 0 && (
                      <span {...stylex.props(styles.remainder)}>
                        {t('group.modelEditor.sync.more', { count: removalRemainder })}
                      </span>
                    )}
                  </section>
                )}
              </div>

              {conflicts.length > 0 && (
                <InlineNotice tone="danger">
                  {t('group.modelEditor.sync.conflict', { names: conflictNames })}
                </InlineNotice>
              )}
              {error !== '' && <InlineNotice tone="danger">{error}</InlineNotice>}
            </div>
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            <Button
              variant="secondary"
              size="sm"
              label={t('common.cancel')}
              isDisabled={pending}
              onClick={() => onOpenChange(false)}
            />
            <Button
              variant={activeRemovals.length ? 'destructive' : 'primary'}
              size="sm"
              isLoading={pending}
              isDisabled={changeCount === 0 || conflicts.length > 0}
              label={t('group.modelEditor.sync.confirm')}
              onClick={onConfirm}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}

const styles = stylex.create({
  root: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  modes: {
    width: '100%',
  },
  changes: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  section: {
    display: 'grid',
    gap: 'var(--space-2)',
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    paddingTop: '10px',
    paddingBottom: '10px',
    paddingInline: '11px',
  },
  sectionAddition: {
    borderColor: 'color-mix(in srgb, var(--color-success) 32%, var(--color-border-subtle))',
    borderLeftWidth: '3px',
    backgroundColor: 'color-mix(in srgb, var(--color-success-bg) 72%, var(--color-surface))',
  },
  sectionRemoval: {
    borderColor: 'color-mix(in srgb, var(--color-danger) 32%, var(--color-border-subtle))',
    borderLeftWidth: '3px',
    backgroundColor: 'color-mix(in srgb, var(--color-danger-bg) 72%, var(--color-surface))',
  },
  sectionTitleAddition: {
    fontSize: 'var(--text-sm)',
    color: 'var(--color-success)',
  },
  sectionTitleRemoval: {
    fontSize: 'var(--text-sm)',
    color: 'var(--color-danger)',
  },
  list: {
    display: 'grid',
    maxHeight: '150px',
    gap: '5px',
    margin: 0,
    overflowY: 'auto',
    padding: 0,
    listStyle: 'none',
  },
  listItem: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: '6px',
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
  },
  code: {
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  empty: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  remainder: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
})
