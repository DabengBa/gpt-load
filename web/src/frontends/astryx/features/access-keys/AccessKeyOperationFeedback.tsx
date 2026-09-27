import * as stylex from '@stylexjs/stylex'
import { Banner } from '@astryxdesign/core'

import { useT } from '../../app/i18n'
import type { MessageId } from '@shared/i18n/message-ids'

const styles = stylex.create({
  stack: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
})

export function AccessKeyOperationFeedback({
  failed,
  editNotApplied,
  mutationFeedbackKey,
}: {
  failed: boolean
  editNotApplied: boolean
  mutationFeedbackKey: string
}) {
  const t = useT()
  if (!failed && mutationFeedbackKey === '') return null
  return (
    <div {...stylex.props(styles.stack)}>
      {failed && (
        <Banner
          status="error"
          title={t(
            editNotApplied
              ? 'accessKeys.drawer.editNotApplied'
              : 'accessKeys.drawer.saveFailed',
          )}
        />
      )}
      {mutationFeedbackKey !== '' && (
        <Banner status="warning" title={t(mutationFeedbackKey as MessageId)} />
      )}
    </div>
  )
}
