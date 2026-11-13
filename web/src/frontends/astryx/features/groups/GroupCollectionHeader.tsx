import * as stylex from '@stylexjs/stylex'
import { IconButton } from '@astryxdesign/core'
import { RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'

import { useT } from '../../app/i18n'

/** Page heading and collection refresh action. */
export function GroupCollectionHeader({
  refreshing,
  refreshingLabel,
  onRefresh,
  children,
}: {
  refreshing: boolean
  refreshingLabel: string
  onRefresh(): void
  children?: ReactNode
}) {
  const t = useT()

  return (
    <>
      <header {...stylex.props(styles.header)}>
        <div {...stylex.props(styles.heading)}>
          <h1 id="groups-title" data-shell-page-title {...stylex.props(styles.title)}>
            {t('groups.title')}
          </h1>
        </div>
        {children && <div {...stylex.props(styles.summary)}>{children}</div>}
        <div {...stylex.props(styles.headerActions)}>
          <IconButton
            variant="ghost"
            size="sm"
            label={t('groups.collection.retry')}
            icon={<RefreshCw size={15} />}
            isLoading={refreshing}
            onClick={onRefresh}
          />
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
  },
  heading: {
    width: { default: 0, '@media (max-width: 860px)': '100%' },
  },
  summary: {
    flex: '1 1 0',
    minWidth: 0,
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
