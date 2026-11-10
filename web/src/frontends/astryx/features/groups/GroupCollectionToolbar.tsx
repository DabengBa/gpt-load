import * as stylex from '@stylexjs/stylex'
import { Button, Selector, TextInput } from '@astryxdesign/core'
import { Search } from 'lucide-react'
import { useIntl } from 'react-intl'

import type { GroupCollectionSort } from '@shared/control/types'

import { useT } from '../../app/i18n'

export interface GroupSortOption {
  value: GroupCollectionSort
  label: string
}

/**
 * Filters live in the URL, not in local state: the input is a draft that the
 * routing hook debounces into `?q=`, so every keystroke stays shareable once
 * it lands.
 */
export function GroupCollectionToolbar({
  searchDraft,
  sort,
  sortOptions,
  itemsShown,
  totalItems,
  hasChangedConditions,
  onSearchChange,
  onSortChange,
  onReset,
}: {
  searchDraft: string
  sort: GroupCollectionSort
  sortOptions: GroupSortOption[]
  itemsShown: readonly unknown[]
  totalItems: number
  hasChangedConditions: boolean
  onSearchChange(value: string): void
  onSortChange(value: string): void
  onReset(): void
}) {
  const t = useT()
  const intl = useIntl()

  return (
    <div
      {...stylex.props(styles.toolbar)}
      role="group"
      aria-label={t('groups.collection.filters.region')}
    >
      <div {...stylex.props(styles.toolbarSearch)}>
        <TextInput
          size="sm"
          label={t('groups.collection.filters.searchLabel')}
          placeholder={t('groups.collection.filters.searchPlaceholder')}
          startIcon={<Search size={14} />}
          value={searchDraft}
          hasClear
          onChange={onSearchChange}
        />
      </div>
      <div {...stylex.props(styles.toolbarField)}>
        <Selector
          size="sm"
          variant="input"
          label={t('groups.collection.filters.sortLabel')}
          options={sortOptions}
          value={sort}
          onChange={onSortChange}
        />
      </div>
      <div {...stylex.props(styles.toolbarResult)}>
        <span aria-live="polite">
          {t('groups.collection.result', {
            shown: intl.formatNumber(itemsShown.length),
            total: intl.formatNumber(totalItems),
          })}
        </span>
        {hasChangedConditions && (
          <Button
            variant="ghost"
            size="sm"
            label={t('groups.collection.filters.reset')}
            onClick={onReset}
          />
        )}
      </div>
    </div>
  )
}

const styles = stylex.create({
  toolbar: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(220px, 1fr) repeat(2, minmax(180px, 0.5fr))',
      '@media (max-width: 880px)': 'repeat(2, minmax(0, 1fr))',
      '@media (max-width: 480px)': 'minmax(0, 1fr)',
    },
    alignItems: 'end',
    gap: 'var(--space-2)',
    paddingTop: 'var(--space-2)',
  },
  toolbarField: {
    minWidth: 0,
  },
  toolbarSearch: {
    minWidth: 0,
    gridColumn: {
      default: 'auto',
      '@media (max-width: 880px)': '1 / -1',
    },
  },
  toolbarResult: {
    gridColumn: '1 / -1',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    minHeight: 30,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontVariantNumeric: 'tabular-nums',
  },
})
