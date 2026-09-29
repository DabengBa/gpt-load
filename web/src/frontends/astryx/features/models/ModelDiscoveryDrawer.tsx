import * as stylex from '@stylexjs/stylex'
import {
  Button,
  CheckboxInput,
  SegmentedControl,
  SegmentedControlItem,
  Skeleton,
  TextInput,
  Tooltip,
} from '@astryxdesign/core'
import { CircleDollarSign, Radar, RefreshCw, Search } from 'lucide-react'
import { useState, type ReactNode } from 'react'

import type { ModelCandidate } from '@shared/control/resources/providers'
import type { ModelDiscoveryDrawerLabels } from '@shared/domain/models/model-draft'
import { useStableLoading } from '../../app/collection-loading'
import { DetailPanel } from '../../components/DetailPanel'
import { InlineNotice } from '../../components/InlineNotice'

export type DiscoveryFilter = 'unadded' | 'all'

/**
 * Classic features/models/ModelDiscoveryDrawer.vue — the candidate picker that
 * sits in a detail panel over the group models editor. Route-driven search and
 * filter stay internal drafts that publish outward on change.
 */
export function ModelDiscoveryDrawer({
  open,
  candidates,
  currentIds: currentIdList,
  loading,
  error,
  labels,
  dismissible = true,
  blocked = false,
  search,
  filter,
  footerActions,
  onOpenChange,
  onSearchChange,
  onFilterChange,
  onRetry,
  onConfirm,
}: {
  open: boolean
  candidates: readonly ModelCandidate[]
  currentIds: readonly string[]
  loading: boolean
  error: string
  labels: ModelDiscoveryDrawerLabels
  dismissible?: boolean
  blocked?: boolean
  search?: string
  filter?: DiscoveryFilter
  footerActions?: ReactNode
  onOpenChange(open: boolean): void
  onSearchChange(search: string): void
  onFilterChange(filter: DiscoveryFilter): void
  onRetry(): void
  onConfirm(candidates: ModelCandidate[]): void
}) {
  const loadingVisible = useStableLoading(loading)

  const [internalSearch, setInternalSearch] = useState(search ?? '')
  const [internalFilter, setInternalFilter] = useState<DiscoveryFilter>(filter ?? 'unadded')
  const [selectedIds, setSelectedIds] = useState<readonly string[]>([])

  // Classic watch(props.search/props.filter): external route values mirror
  // into the internal drafts (render adjustment — never paint stale values).
  const [lastSearch, setLastSearch] = useState(search)
  if (lastSearch !== search) {
    setLastSearch(search)
    setInternalSearch(search ?? '')
  }
  const [lastFilter, setLastFilter] = useState(filter)
  if (lastFilter !== filter) {
    setLastFilter(filter)
    setInternalFilter(filter ?? 'unadded')
  }

  // Classic watch(props.open): opening reseeds search/filter and drops the
  // previous selection.
  const [wasOpen, setWasOpen] = useState(open)
  if (wasOpen !== open) {
    setWasOpen(open)
    if (open) {
      setInternalSearch(search ?? '')
      setInternalFilter(filter ?? 'unadded')
      setSelectedIds([])
    }
  }

  const currentIds = new Set(currentIdList.map((id) => id.trim()).filter(Boolean))
  // Classic watch(props.candidates/props.currentIds): prune selections that are
  // no longer pickable. Render-time adjustments against the previous prop
  // references — never paint stale selection state.
  const [lastCandidates, setLastCandidates] = useState(candidates)
  const [lastCurrentIdList, setLastCurrentIdList] = useState(currentIdList)
  if (lastCandidates !== candidates || lastCurrentIdList !== currentIdList) {
    setLastCandidates(candidates)
    setLastCurrentIdList(currentIdList)
    const valid = new Set(candidates.map(({ id }) => id))
    setSelectedIds((current) =>
      current.filter((id) => valid.has(id) && !currentIds.has(id)),
    )
  }

  const selected = new Set(selectedIds)
  const query = internalSearch.trim().toLocaleLowerCase()
  const visibleCandidates = candidates.filter(
    (candidate) =>
      (internalFilter === 'all' || !currentIds.has(candidate.id)) &&
      (!query ||
        `${candidate.name} ${candidate.id} ${candidate.sources.join(' ')}`
          .toLocaleLowerCase()
          .includes(query)),
  )
  const selectableVisible = visibleCandidates.filter(
    (candidate) => !currentIds.has(candidate.id),
  )
  const allVisibleSelected =
    selectableVisible.length > 0 &&
    selectableVisible.every((candidate) => selected.has(candidate.id))

  function setSearchValue(value: string): void {
    setInternalSearch(value)
    onSearchChange(value)
  }

  function setFilterValue(value: string): void {
    if (value !== 'unadded' && value !== 'all') return
    setInternalFilter(value)
    onFilterChange(value)
  }

  function setCandidate(candidate: ModelCandidate, checked: boolean): void {
    const next = new Set(selectedIds)
    if (checked) next.add(candidate.id)
    else next.delete(candidate.id)
    setSelectedIds([...next])
  }

  function toggleVisibleCandidates(): void {
    const next = new Set(selectedIds)
    if (allVisibleSelected) {
      for (const candidate of selectableVisible) next.delete(candidate.id)
    } else {
      for (const candidate of selectableVisible) next.add(candidate.id)
    }
    setSelectedIds([...next])
  }

  function confirm(): void {
    if (blocked || selectedIds.length === 0 || loading) return
    const picked = new Set(selectedIds)
    onConfirm(candidates.filter(({ id }) => picked.has(id)))
  }

  return (
    <DetailPanel
      isOpen={open}
      title={labels.title}
      subtitle={labels.description}
      dismissible={dismissible && !loading}
      onOpenChange={onOpenChange}
      footer={
        <div {...stylex.props(styles.footer)}>
          <div {...stylex.props(styles.selection)}>
            <Button
              variant="secondary"
              size="sm"
              isDisabled={blocked || loading || selectableVisible.length === 0}
              onClick={toggleVisibleCandidates}
              label={allVisibleSelected ? labels.deselectAll : labels.selectAll}
            />
            <span aria-live="polite">{labels.selected(selectedIds.length)}</span>
          </div>
          <div {...stylex.props(styles.actions)}>
            {footerActions}
            <Button
              variant="secondary"
              size="sm"
              isDisabled={blocked || loading}
              onClick={() => onOpenChange(false)}
              label={labels.cancel}
            />
            <Button
              size="sm"
              isDisabled={blocked || loading || selectedIds.length === 0}
              onClick={confirm}
              label={labels.confirm}
            />
          </div>
        </div>
      }
    >
      <div {...stylex.props(styles.filters)}>
        <TextInput
          xstyle={styles.search}
          label={labels.search}
          isLabelHidden
          placeholder={labels.search}
          startIcon={<Search size={14} aria-hidden />}
          value={internalSearch}
          hasClear
          size="sm"
          isDisabled={blocked}
          onChange={setSearchValue}
        />
        <SegmentedControl
          value={internalFilter}
          label={labels.filterLabel}
          size="sm"
          isDisabled={blocked}
          onChange={setFilterValue}
        >
          <SegmentedControlItem value="all" label={labels.filterAll} />
          <SegmentedControlItem value="unadded" label={labels.filterUnadded} />
        </SegmentedControl>
      </div>
      <div aria-busy={loading || undefined}>
        {loading || loadingVisible ? (
          <div role="status" aria-label={labels.loading} {...stylex.props(styles.skeleton)}>
            <Skeleton height={58} radius={2} />
            <Skeleton height={58} radius={2} />
            <Skeleton height={58} radius={2} />
            <Skeleton height={58} radius={2} />
            <Skeleton height={58} radius={2} />
          </div>
        ) : error ? (
          <div {...stylex.props(styles.state)}>
            <InlineNotice
              tone="danger"
              action={
                <Button
                  variant="ghost"
                  size="sm"
                  isDisabled={blocked}
                  onClick={onRetry}
                  icon={<RefreshCw size={15} aria-hidden />}
                  label={labels.retry}
                />
              }
            >
              {error}
            </InlineNotice>
          </div>
        ) : (
          <fieldset {...stylex.props(styles.candidateList)}>
            <legend {...stylex.props(styles.srOnly)}>{labels.filterLabel}</legend>
            {visibleCandidates.map((candidate) => {
              const added = currentIds.has(candidate.id)
              return (
                <label
                  key={candidate.id}
                  {...stylex.props(styles.candidate, added && styles.candidateAdded)}
                >
                  <CheckboxInput
                    value={added || selected.has(candidate.id)}
                    isDisabled={blocked || added}
                    label={
                      added ? `${labels.alreadyAdded} · ${candidate.name}` : candidate.name
                    }
                    isLabelHidden
                    disabledMessage={added ? labels.alreadyAdded : undefined}
                    onChange={(checked) => setCandidate(candidate, checked)}
                  />
                  <span {...stylex.props(styles.identity)}>
                    <strong {...stylex.props(styles.identityName)} title={candidate.name}>
                      {candidate.name}
                    </strong>
                    <code {...stylex.props(styles.identityId)} title={candidate.id}>
                      {candidate.id}
                    </code>
                  </span>
                  <span {...stylex.props(styles.evidence)}>
                    {candidate.sources.includes('live') && (
                      <Tooltip content={labels.sources.live}>
                        <span
                          {...stylex.props(styles.statusIcon, styles.statusIconLive)}
                          role="img"
                          aria-label={labels.sources.live}
                        >
                          <Radar size={17} strokeWidth={1.9} aria-hidden />
                        </span>
                      </Tooltip>
                    )}
                    {candidate.pricing_source && (
                      <Tooltip content={labels.pricingDiscovered(candidate.pricing_source)}>
                        <span
                          {...stylex.props(styles.statusIcon, styles.statusIconPricing)}
                          role="img"
                          aria-label={labels.pricingDiscovered(candidate.pricing_source)}
                        >
                          <CircleDollarSign size={17} strokeWidth={1.9} aria-hidden />
                        </span>
                      </Tooltip>
                    )}
                  </span>
                </label>
              )
            })}
            {visibleCandidates.length === 0 && (
              <div {...stylex.props(styles.emptyFeedback)}>
                <InlineNotice tone="warning">
                  {candidates.length ? labels.noMatches : labels.empty}
                </InlineNotice>
              </div>
            )}
          </fieldset>
        )}
      </div>
    </DetailPanel>
  )
}

const styles = stylex.create({
  filters: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    marginBottom: 'var(--space-3)',
  },
  search: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: '0%',
    minWidth: 0,
  },
  skeleton: {
    display: 'grid',
    gap: '2px',
    minHeight: '328px',
  },
  state: {
    display: 'grid',
    minHeight: '280px',
    alignItems: 'start',
    paddingTop: 'var(--space-3)',
  },
  candidateList: {
    display: 'grid',
    margin: 0,
    borderWidth: 0,
    padding: 0,
  },
  candidate: {
    display: 'grid',
    minHeight: 0,
    gridTemplateColumns: {
      default: 'auto minmax(0, 1fr) minmax(52px, auto)',
      '@media (max-width: 520px)': 'auto minmax(0, 1fr)',
    },
    alignItems: 'center',
    gap: '10px',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBlock: 'var(--space-3)',
    paddingInline: '1px',
    cursor: 'pointer',
  },
  candidateAdded: {
    color: 'var(--color-text-muted)',
    cursor: 'not-allowed',
  },
  identity: {
    display: 'grid',
    minWidth: 0,
    gap: '2px',
  },
  identityName: {
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: 'var(--text-sm)',
  },
  identityId: {
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  evidence: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: { default: 'flex-end', '@media (max-width: 520px)': 'flex-start' },
    gridColumn: { '@media (max-width: 520px)': '2' },
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
  },
  statusIcon: {
    display: 'grid',
    width: '24px',
    height: '24px',
    placeItems: 'center',
    borderRadius: 'var(--radius-control)',
  },
  statusIconLive: {
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
  },
  statusIconPricing: {
    backgroundColor: 'var(--color-success-bg)',
    color: 'var(--color-success)',
  },
  emptyFeedback: {
    marginTop: 'var(--space-3)',
  },
  footer: {
    display: 'flex',
    width: '100%',
    alignItems: { default: 'center', '@media (max-width: 520px)': 'stretch' },
    flexDirection: { default: 'row', '@media (max-width: 520px)': 'column' },
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  selection: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: { '@media (max-width: 520px)': 'space-between' },
    gap: 'var(--space-2)',
  },
  actions: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  srOnly: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
})
