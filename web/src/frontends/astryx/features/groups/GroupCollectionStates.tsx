import * as stylex from '@stylexjs/stylex'
import { Button, EmptyState, Skeleton } from '@astryxdesign/core'
import { KeyRound, Layers3, Search, TriangleAlert } from 'lucide-react'

import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'

/** Everything the collection renders when it has no table to show. */
export function GroupCollectionSkeleton({ rows, height }: { rows: number; height: number }) {
  const t = useT()
  return (
    <div
      {...stylex.props(styles.skeletonGrid)}
      role="status"
      aria-label={t('groups.collection.loading')}
    >
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} height={height} radius={2} />
      ))}
    </div>
  )
}

export function GroupCollectionError({ onRetry }: { onRetry(): void }) {
  const t = useT()
  return (
    <div {...stylex.props(styles.errorBox)} role="alert">
      <EmptyState
        title={t('groups.collection.errorTitle')}
        description={t('groups.collection.errorDescription')}
        icon={<TriangleAlert size={20} />}
        actions={
          <Button
            variant="secondary"
            size="sm"
            label={t('groups.collection.retry')}
            onClick={onRetry}
          />
        }
      />
    </div>
  )
}

/**
 * A failed background refetch keeps the last good rows on screen; the banner
 * is the only hint that they may be stale.
 */
export function GroupCollectionStaleBanner({ onRetry }: { onRetry(): void }) {
  const t = useT()
  return (
    <div {...stylex.props(styles.staleBanner)} role="status">
      <TriangleAlert size={13} aria-hidden />
      <span>{t('groups.collection.stale')}</span>
      <Button
        variant="secondary"
        size="sm"
        label={t('groups.collection.retry')}
        onClick={onRetry}
      />
    </div>
  )
}

/**
 * Two distinct dead ends: nothing exists yet (import is the way out) versus a
 * filter that matched nothing (reset is the way out).
 */
export function GroupCollectionEmpty({
  variant,
  onReset,
}: {
  variant: 'empty' | 'noResults'
  onReset(): void
}) {
  const t = useT()
  if (variant === 'noResults') {
    return (
      <EmptyState
        title={t('groups.collection.noResultsTitle')}
        description={t('groups.collection.noResultsDescription')}
        icon={<Search size={20} />}
        actions={
          <Button
            variant="secondary"
            size="sm"
            label={t('groups.collection.filters.reset')}
            onClick={onReset}
          />
        }
      />
    )
  }
  return (
    <EmptyState
      title={t('groups.collection.emptyTitle')}
      description={t('groups.collection.emptyDescription')}
      icon={<Layers3 size={20} />}
      actions={
        <Button
          variant="secondary"
          size="sm"
          label={t('groups.collection.importCredentials')}
          icon={<KeyRound size={15} />}
          href={pagePath('import')}
        />
      }
    />
  )
}

const styles = stylex.create({
  skeletonGrid: {
    display: 'grid',
    gap: 'var(--space-2)',
    paddingTop: 'var(--space-2)',
  },
  errorBox: {
    paddingTop: 14,
  },
  staleBanner: {
    marginTop: 'var(--space-2)',
    borderRadius: 'var(--radius-control, 6px)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    backgroundColor:
      'var(--color-warning-bg, color-mix(in srgb, var(--color-warning) 12%, transparent))',
    color: 'var(--color-text)',
    padding: '9px 12px',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    fontSize: 'var(--text-meta)',
  },
})
