import * as stylex from '@stylexjs/stylex'
import { Badge, IconButton, Tooltip } from '@astryxdesign/core'
import {
  ArrowRight,
  CircleAlert,
  CircleCheck,
  CircleHelp,
  CircleOff,
  ScrollText,
  type LucideIcon,
} from 'lucide-react'
import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { useIntl } from 'react-intl'

import type { HealthCredentialCountsDto, HealthGroupDto } from '@shared/control/resources/health'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { CredentialHealthBar } from '../../components/CredentialHealthBar'
import { MonitorSectionHeading } from './MonitorSectionHeading'

const TABLET = '@media (max-width: 860px)'
const NARROW = '@media (max-width: 520px)'

const styles = stylex.create({
  section: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-4)',
  },
  // LedgerRecordList port: the container is the grid; the page-owned
  // --ledger-record-list-grid var (set inline below) supplies desktop columns.
  list: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'var(--ledger-record-list-grid, minmax(0, 1fr))',
      [TABLET]: 'minmax(0, 1fr)',
    },
    columnGap: {
      default: 'var(--ledger-record-list-column-gap, 16px)',
      [TABLET]: 10,
    },
    rowGap: { [TABLET]: 10 },
    overflowX: { default: 'auto', [TABLET]: 'visible' },
    overflowY: { default: 'hidden', [TABLET]: 'visible' },
    borderBottomWidth: { default: 1, [TABLET]: 0 },
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    paddingTop: { [TABLET]: 10 },
  },
  headRow: {
    display: { default: 'grid', [TABLET]: 'none' },
    gridColumn: '1 / -1',
    gridTemplateColumns: 'subgrid',
    alignItems: 'center',
    minHeight: 38,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    fontWeight: 500,
    letterSpacing: '0.04em',
  },
  headCell: {
    justifySelf: 'stretch',
    textAlign: 'left',
  },
  record: {
    position: 'relative',
    display: 'grid',
    gridColumn: {
      default: '1 / -1',
      [TABLET]: 1,
    },
    gridTemplateColumns: {
      default: 'subgrid',
      // Classic page override: identity / keys span the full card width while
      // status, exceptions and actions share the second row.
      [TABLET]: 'minmax(0, 1fr) auto',
      [NARROW]: 'minmax(0, 1fr)',
    },
    alignItems: { default: 'center', [TABLET]: 'start' },
    minHeight: { default: 78, [TABLET]: 0 },
    rowGap: { [TABLET]: 14 },
    columnGap: { [TABLET]: 16 },
    borderStyle: 'solid',
    borderTopWidth: 1,
    borderRightWidth: { default: 0, [TABLET]: 1 },
    borderBottomWidth: { default: 0, [TABLET]: 1 },
    borderLeftWidth: { default: 0, [TABLET]: 1 },
    borderColor: 'var(--color-border-subtle)',
    borderRadius: { [TABLET]: 'var(--radius-control)' },
    backgroundColor: {
      default: 'transparent',
      ':hover': 'var(--color-surface-sunken)',
      // Card rows already carry the surface color; classic suppresses the
      // hover highlight there.
      [TABLET]: {
        default: 'var(--color-surface)',
        ':hover': 'var(--color-surface)',
      },
    },
    paddingBlock: { default: 12, [TABLET]: 16, [NARROW]: 14 },
    paddingInline: { default: 0, [TABLET]: 16, [NARROW]: 13 },
    transitionProperty: 'background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  // First record abuts the header divider — the classic :first-of-type rule
  // gives it the stronger border color; under the card layout that override is
  // a no-op (full border is already subtle). :first-of-type is not in the
  // stylex allowlist, so the row applies this on index === 0 instead.
  recordFirst: {
    borderTopColor: {
      default: 'var(--color-border-control)',
      [TABLET]: 'var(--color-border-subtle)',
    },
  },
  cell: {
    minWidth: 0,
    justifySelf: 'stretch',
    textAlign: 'left',
  },
  recordIdentity: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'flex-start',
    flexDirection: 'column',
    flexWrap: 'wrap',
    gap: 'var(--space-1)',
    gridColumn: { [TABLET]: '1 / -1', [NARROW]: 1 },
  },
  recordName: {
    maxWidth: '100%',
    color: { default: 'var(--color-text)', ':hover': 'var(--color-action)' },
    fontWeight: 620,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  recordStatus: {
    gridColumn: { [NARROW]: 1 },
  },
  recordKeys: {
    gridColumn: { [TABLET]: '1 / -1', [NARROW]: 1 },
  },
  recordExceptions: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    gridColumn: { [NARROW]: 1 },
  },
  recordNone: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  recordActions: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    justifyContent: { default: 'flex-end', [NARROW]: 'flex-start' },
    alignSelf: { [TABLET]: 'center' },
    gridColumn: { [NARROW]: 1 },
  },
  footer: {
    display: 'flex',
    minHeight: 42,
    alignItems: 'center',
    justifyContent: 'flex-end',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
  },
  // Classic AppButton variant="link" size="inline" — a baseline-aligned text
  // button that underlines on hover.
  toggle: {
    display: 'inline',
    minHeight: 0,
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: { default: 'var(--color-action)', ':hover': 'var(--color-action-hover)' },
    padding: 0,
    // The astryx reset already applies `font: inherit` to buttons.
    verticalAlign: 'baseline',
    cursor: 'pointer',
    textDecorationLine: { default: 'none', ':hover': 'underline' },
  },
  empty: {
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    padding: 'var(--space-6)',
    textAlign: 'center',
  },
})

type StatusTone = 'neutral' | 'success' | 'warning' | 'danger'

interface GroupStatus {
  key: 'disabled' | 'empty' | 'unavailable' | 'limited' | 'available'
  label: string
  tone: StatusTone
}

const badgeVariants = {
  neutral: 'neutral',
  success: 'success',
  warning: 'warning',
  danger: 'error',
} as const

// Classic StatusBadge always renders a tone icon (12px at compact size).
const badgeIcons: Record<StatusTone, LucideIcon> = {
  neutral: CircleHelp,
  success: CircleCheck,
  warning: CircleAlert,
  danger: CircleOff,
}

function groupDetailHref(id: number): string {
  return `${pagePath('groups')}/${id}`
}

function groupLogsHref(id: number): string {
  return `${pagePath('logs')}?group_id=${id}`
}

/**
 * Per-group credential health list — classic GroupHealthCollection.vue,
 * including its LedgerRecordList grid container (role=table) with the
 * scroll-hint overflow affordance.
 */
export function GroupHealthCollection({
  groups,
  expanded,
  onToggle,
}: {
  groups: HealthGroupDto[]
  expanded: boolean
  onToggle: () => void
}) {
  const intl = useIntl()
  const t = useT()
  const n = (value: number): string => intl.formatNumber(value)
  const defaultLimit = 5

  function status(group: HealthGroupDto): GroupStatus {
    if (!group.enabled) {
      return { key: 'disabled', label: t('monitor.health.groups.disabled'), tone: 'neutral' }
    }
    if (group.counts.credentials === 0) {
      return {
        key: 'empty',
        label: t('monitor.health.groups.emptyCredentials'),
        tone: 'danger',
      }
    }
    if (group.counts.available === 0) {
      return {
        key: 'unavailable',
        label: t('monitor.health.groups.unavailable'),
        tone: 'danger',
      }
    }
    if (group.counts.cooldown > 0 || group.counts.blacklisted > 0) {
      return { key: 'limited', label: t('monitor.health.groups.limited'), tone: 'warning' }
    }
    return { key: 'available', label: t('monitor.health.groups.available'), tone: 'success' }
  }

  function statusRank(group: HealthGroupDto): number {
    const key = status(group).key
    if (key === 'unavailable' || key === 'empty') return 0
    if (key === 'limited') return 1
    if (key === 'available') return 2
    return 3
  }

  const sortedGroups = [...groups].sort(
    (left, right) =>
      statusRank(left) - statusRank(right) ||
      left.name.localeCompare(right.name) ||
      left.id - right.id,
  )
  const visibleGroups = expanded ? sortedGroups : sortedGroups.slice(0, defaultLimit)
  const canToggle = sortedGroups.length > defaultLimit

  function credentialHealthLabel(counts: HealthCredentialCountsDto): string {
    return t('monitor.health.groups.credentialHealthLabel', {
      total: n(counts.credentials),
      available: n(counts.available),
      cooldown: n(counts.cooldown),
      blacklisted: n(counts.blacklisted),
    })
  }

  // LedgerRecordList overflow wiring: the classic measured on mount, on every
  // update and on window resize. ResizeObserver already fires once on observe()
  // (the mount measure) and re-running the effect when the visible row count
  // changes reproduces the update measure.
  const listRef = useRef<HTMLDivElement | null>(null)
  const [overflowing, setOverflowing] = useState(false)
  useEffect(() => {
    const element = listRef.current
    if (element === null) return
    const update = (): void => {
      setOverflowing(element.scrollWidth > element.clientWidth + 1)
    }
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(update) : undefined
    observer?.observe(element)
    // Without ResizeObserver there is no initial delivery — defer the mount
    // measure a frame so it stays out of the synchronous effect body.
    const fallbackFrame = observer === undefined ? requestAnimationFrame(update) : undefined
    window.addEventListener('resize', update)
    return () => {
      observer?.disconnect()
      if (fallbackFrame !== undefined) cancelAnimationFrame(fallbackFrame)
      window.removeEventListener('resize', update)
    }
  }, [visibleGroups.length])

  const label = t('monitor.health.groups.tableLabel')
  const scrollHint = t('monitor.scrollHint')
  const accessibleLabel = overflowing && scrollHint !== '' ? `${label} · ${scrollHint}` : label

  return (
    <section {...stylex.props(styles.section)} aria-labelledby="group-health-title">
      <MonitorSectionHeading
        id="group-health-title"
        title={t('monitor.health.groups.title')}
        description={t('monitor.health.groups.description')}
      />

      {sortedGroups.length === 0 ? (
        <div {...stylex.props(styles.empty)}>{t('monitor.health.groups.empty')}</div>
      ) : (
        <>
          <div
            ref={listRef}
            {...stylex.props(styles.list)}
            style={
              {
                '--ledger-record-list-grid':
                  'minmax(170px, 1.25fr) 126px minmax(190px, 1.25fr) minmax(150px, 0.9fr) 84px',
              } as CSSProperties
            }
            role="table"
            data-testid="ledger-record-list"
            aria-label={accessibleLabel}
            aria-rowcount={visibleGroups.length + 1}
            tabIndex={overflowing ? 0 : undefined}
          >
            <div
              {...stylex.props(styles.headRow)}
              role="row"
              data-testid="ledger-record-list__header"
              aria-rowindex={1}
            >
              <span role="columnheader" {...stylex.props(styles.headCell)}>
                {/* Catalog value is 'Group: {name}'; vue-i18n renders a missing
                    param as '' ("Group: ") — passing an explicit empty name
                    keeps that output and avoids react-intl's FORMAT_ERROR. */}
                {t('monitor.health.groups.columns.group', { name: '' })}
              </span>
              <span role="columnheader" {...stylex.props(styles.headCell)}>
                {t('monitor.health.groups.columns.status')}
              </span>
              <span role="columnheader" {...stylex.props(styles.headCell)}>
                {t('monitor.health.groups.columns.credentialHealth')}
              </span>
              <span role="columnheader" {...stylex.props(styles.headCell)}>
                {t('monitor.health.groups.columns.exceptions')}
              </span>
              <span role="columnheader" {...stylex.props(styles.headCell)}>
                {t('monitor.health.groups.columns.actions')}
              </span>
            </div>

            {visibleGroups.map((group, index) => {
              const groupStatus = status(group)
              const StatusIcon = badgeIcons[groupStatus.tone]
              return (
                <article
                  key={group.id}
                  {...stylex.props(styles.record, index === 0 && styles.recordFirst)}
                  role="row"
                  aria-rowindex={index + 2}
                >
                  <div role="cell" {...stylex.props(styles.cell, styles.recordIdentity)}>
                    <Tooltip content={group.name}>
                      <RouteLink
                        to={groupDetailHref(group.id)}
                        {...stylex.props(styles.recordName)}
                      >
                        {group.name}
                      </RouteLink>
                    </Tooltip>
                  </div>

                  <div role="cell" {...stylex.props(styles.cell, styles.recordStatus)}>
                    <Badge
                      variant={badgeVariants[groupStatus.tone]}
                      icon={<StatusIcon size={12} aria-hidden="true" />}
                      label={groupStatus.label}
                    />
                  </div>

                  <div role="cell" {...stylex.props(styles.cell, styles.recordKeys)}>
                    <CredentialHealthBar
                      counts={group.counts}
                      label={credentialHealthLabel(group.counts)}
                    />
                  </div>

                  <div role="cell" {...stylex.props(styles.cell, styles.recordExceptions)}>
                    {group.counts.cooldown > 0 && (
                      <Badge
                        variant="warning"
                        icon={<CircleAlert size={12} aria-hidden="true" />}
                        label={t('monitor.health.groups.cooldownCount', {
                          count: n(group.counts.cooldown),
                        })}
                      />
                    )}
                    {group.counts.blacklisted > 0 && (
                      <Badge
                        variant="error"
                        icon={<CircleOff size={12} aria-hidden="true" />}
                        label={t('monitor.health.groups.blacklistedCount', {
                          count: n(group.counts.blacklisted),
                        })}
                      />
                    )}
                    {group.counts.cooldown === 0 && group.counts.blacklisted === 0 && (
                      <span {...stylex.props(styles.recordNone)}>
                        {t('monitor.health.groups.none')}
                      </span>
                    )}
                  </div>

                  <div role="cell" {...stylex.props(styles.cell, styles.recordActions)}>
                    <IconButton
                      variant="ghost"
                      size="sm"
                      label={t('monitor.health.groups.viewLogsFor', {
                        name: group.name,
                      })}
                      icon={<ScrollText size={15} aria-hidden="true" />}
                      href={groupLogsHref(group.id)}
                    />
                    <IconButton
                      variant="ghost"
                      size="sm"
                      label={t('monitor.health.groups.viewGroup', {
                        name: group.name,
                      })}
                      icon={<ArrowRight size={15} aria-hidden="true" />}
                      href={groupDetailHref(group.id)}
                    />
                  </div>
                </article>
              )
            })}
          </div>

          {canToggle && (
            <footer {...stylex.props(styles.footer)}>
              <button type="button" {...stylex.props(styles.toggle)} onClick={onToggle}>
                {expanded
                  ? t('monitor.health.groups.collapse')
                  : t('monitor.health.groups.showAll', {
                      count: n(sortedGroups.length),
                    })}
              </button>
            </footer>
          )}
        </>
      )}
    </section>
  )
}
