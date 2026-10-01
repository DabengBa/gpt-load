import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'

import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'
import { InlineNotice } from '../../components/InlineNotice'

const styles = stylex.create({
  notice: {
    display: 'flex',
    alignItems: { default: 'center', '@media (max-width: 640px)': 'stretch' },
    flexDirection: { default: 'row', '@media (max-width: 640px)': 'column' },
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-warning-bg)',
    paddingBlock: 'var(--space-3)',
    paddingInline: 'var(--space-4)',
  },
  identity: {
    overflowWrap: 'anywhere',
  },
  actions: {
    display: 'flex',
    flexShrink: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
})

export interface ImportOperationNoticeProps {
  messageKey: MessageId | ''
  resourceIdentity: string
  canRetry: boolean
  canAbandon: boolean
  pending: boolean
  onRetry(): void
  onAbandon(): void
}

export function ImportOperationNotice({
  messageKey,
  resourceIdentity,
  canRetry,
  canAbandon,
  pending,
  onRetry,
  onAbandon,
}: ImportOperationNoticeProps) {
  const t = useT()
  if (!messageKey) return null
  return (
    <section {...stylex.props(styles.notice)} aria-live="polite">
      <InlineNotice tone="warning">{t(messageKey)}</InlineNotice>
      {resourceIdentity && <code {...stylex.props(styles.identity)}>{resourceIdentity}</code>}
      <div {...stylex.props(styles.actions)}>
        {!resourceIdentity && (
          <Button
            variant="secondary"
            label={t('import.operation.checkResult')}
            isDisabled={!canRetry}
            isLoading={pending}
            onClick={onRetry}
          />
        )}
        <Button
          variant="ghost"
          label={t('import.operation.abandon')}
          isDisabled={!canAbandon || pending}
          onClick={onAbandon}
        />
      </div>
    </section>
  )
}
