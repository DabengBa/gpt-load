import * as stylex from '@stylexjs/stylex'
import { Badge, Button, IconButton } from '@astryxdesign/core'
import { KeyRound, RefreshCw } from 'lucide-react'
import { useIntl } from 'react-intl'

import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'

/**
 * Page heading plus the two collection-level actions. Import is only offered
 * once the collection has rows — an empty install gets the empty-state CTA
 * instead, so the same button never appears twice on screen.
 */
export function GroupCollectionHeader({
  total,
  refreshing,
  refreshingLabel,
  onRefresh,
}: {
  total: number | undefined
  refreshing: boolean
  refreshingLabel: string
  onRefresh(): void
}) {
  const t = useT()
  const intl = useIntl()

  return (
    <>
      <header {...stylex.props(styles.header)}>
        <div {...stylex.props(styles.heading)}>
          <h1 id="groups-title" {...stylex.props(styles.title)}>
            {t('groups.title')}
          </h1>
          {total !== undefined && <Badge variant="neutral" label={intl.formatNumber(total)} />}
        </div>
        <div {...stylex.props(styles.headerActions)}>
          <IconButton
            variant="ghost"
            size="sm"
            label={t('groups.collection.retry')}
            icon={<RefreshCw size={15} />}
            isLoading={refreshing}
            onClick={onRefresh}
          />
          {total !== undefined && total > 0 && (
            <Button
              variant="primary"
              size="sm"
              label={t('groups.collection.importCredentials')}
              icon={<KeyRound size={15} />}
              href={pagePath('import')}
            />
          )}
        </div>
      </header>
      <span aria-live="polite" {...stylex.props(styles.refreshing)}>
        {refreshingLabel}
      </span>
    </>
  )
}

const styles = stylex.create({
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    paddingBottom: 'var(--space-2)',
  },
  heading: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-heading-2-size, 20px)',
    fontWeight: 650,
  },
  refreshing: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
})
