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
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useIntl } from 'react-intl'

import type { HealthProblemCredentialDto } from '@shared/control/resources/health'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { MonitorSectionHeading } from './MonitorSectionHeading'

const TABLET = '@media (max-width: 860px)'
const NARROW = '@media (max-width: 520px)'

const styles = stylex.create({
  section: {
    display: 'grid',
    minWidth: 0,
    gridTemplateRows: {
      default: 'auto var(--health-focus-content-height, 266px)',
      [TABLET]: 'auto auto',
    },
    gap: 'var(--space-4)',
  },
  // LedgerRecordList port + classic `.problem-health-grid` page class: the
  // desktop columns are owned by this page, and the list scrolls vertically
  // inside the shared --health-focus-content-height row.
  list: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'minmax(220px, 1.5fr) 90px minmax(170px, 1fr) minmax(165px, 1.05fr) 64px',
      [TABLET]: 'minmax(0, 1fr)',
    },
    columnGap: 10,
    rowGap: { [TABLET]: 10 },
    height: { default: '100%', [TABLET]: 'auto' },
    alignContent: 'start',
    overflowX: { default: 'auto', [TABLET]: 'visible' },
    overflowY: { default: 'auto', [TABLET]: 'visible' },
    scrollbarGutter: { default: 'stable', [TABLET]: 'auto' },
    borderBottomWidth: { default: 1, [TABLET]: 0 },
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    paddingTop: { [TABLET]: 10 },
  },
  headRow: {
    display: { default: 'grid', [TABLET]: 'none' },
    // `.problem-health-grid :deep(.ledger-record-list__header)` — the sticky
    // header only exists while the list scrolls vertically.
    position: { default: 'sticky', [TABLET]: 'static' },
    zIndex: 1,
    top: 0,
    backgroundColor: 'var(--color-surface)',
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
  actionsHeading: {
    textAlign: 'right',
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
      [TABLET]: 'minmax(0, 1fr) auto',
      [NARROW]: 'minmax(0, 1fr)',
    },
    alignItems: { default: 'center', [TABLET]: 'start' },
    // Classic --ledger-record-list-record-min-height: 76px.
    minHeight: { default: 76, [TABLET]: 0 },
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
      [TABLET]: {
        default: 'var(--color-surface)',
        ':hover': 'var(--color-surface)',
      },
    },
    // Classic --ledger-record-list-record-padding: 10px 0.
    paddingBlock: { default: 10, [TABLET]: 16, [NARROW]: 14 },
    paddingInline: { default: 0, [TABLET]: 16, [NARROW]: 13 },
    transitionProperty: 'background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  // `:first-of-type` is not in the stylex allowlist; applied on index === 0.
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
  // Classic shares `.problem-health-record__identity/__window/__recovery`
  // between cells: a flex column that spans the card width on tablet.
  stackCell: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'flex-start',
    flexDirection: 'column',
    gap: 'var(--space-1)',
    gridColumn: { [TABLET]: '1 / -1', [NARROW]: 1 },
  },
  identityLink: {
    maxWidth: '100%',
    color: { default: 'var(--color-text)', ':hover': 'var(--color-action)' },
    fontFamily: 'var(--font-mono)',
    fontWeight: 620,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  // Classic `.problem-health-record small` — meta/consecutive/hint lines.
  recordMeta: {
    width: '100%',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  windowSummary: {
    display: 'inline-flex',
    alignItems: 'baseline',
    gap: 4,
    maxWidth: '100%',
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  windowValueSuccess: {
    color: 'var(--color-success)',
  },
  windowValueDanger: {
    color: 'var(--color-danger)',
  },
  recoveryTime: {
    width: 'fit-content',
    borderRadius: 3,
    maxWidth: '100%',
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  actionsCell: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    justifyContent: { default: 'flex-end', [NARROW]: 'flex-start' },
    gap: 2,
    gridColumn: { [NARROW]: 1 },
  },
  clearPanel: {
    display: 'flex',
    minWidth: 0,
    height: { default: '100%', [TABLET]: 'auto' },
    minHeight: { [TABLET]: 150 },
    alignItems: { default: 'center', [NARROW]: 'flex-start' },
    flexDirection: { [NARROW]: 'column' },
    justifyContent: { default: 'space-between', [NARROW]: 'center' },
    gap: 'var(--space-5)',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-control)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    backgroundColor: 'color-mix(in srgb, var(--color-success-bg) 42%, var(--color-surface))',
    paddingBlock: 'var(--space-5)',
    paddingInline: 'var(--space-4)',
    textAlign: { [NARROW]: 'left' },
  },
  clearPanelInactive: {
    backgroundColor: 'color-mix(in srgb, var(--color-surface-sunken) 72%, var(--color-surface))',
  },
  clearMain: {
    display: 'flex',
    minWidth: 0,
    alignItems: { default: 'center', [NARROW]: 'flex-start' },
    flexDirection: { [NARROW]: 'column' },
    gap: 'var(--space-3)',
  },
  clearCopy: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
  },
  clearHint: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  clearMeta: {
    flex: '0 0 auto',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    whiteSpace: { default: 'nowrap', [NARROW]: 'normal' },
  },
})

export interface HealthProblemItem {
  credential: HealthProblemCredentialDto
  kind: 'cooldown' | 'blacklisted'
  tone: 'warning' | 'danger'
}

export interface RecoveryDisplay {
  relative: string
  exact: string
  labelKey: 'cooldownRecovery' | 'scheduledReleaseRecovery'
  hintKey: 'cooldownHint' | 'scheduledReleaseHint'
}

type StatusTone = 'neutral' | 'success' | 'warning' | 'danger'

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

function groupCredentialsHref(groupId: number): string {
  return `${pagePath('groups')}/${groupId}?tab=credentials`
}

function credentialLogsHref(groupId: number, credentialId: number): string {
  return `${pagePath('logs')}?group_id=${groupId}&credential_id=${credentialId}`
}

/**
 * Classic OverflowTooltip: the tooltip only exists while the trigger actually
 * clips its content; a clipping trigger becomes tabbable so keyboard users can
 * reach the full text. `to` renders a RouteLink, matching the classic
 * `:as="RouterLink"` usage.
 */
function OverflowTip({
  as: Tag = 'span',
  content,
  to,
  ariaLabel,
  xstyle,
  children,
}: {
  as?: 'span' | 'small'
  content: string
  to?: string
  ariaLabel?: string
  xstyle?: stylex.StyleXStyles
  children: ReactNode
}) {
  const [element, setElement] = useState<HTMLElement | null>(null)
  const [overflowing, setOverflowing] = useState(false)

  useEffect(() => {
    if (element === null) return
    const update = (): void => {
      // The classic falls back to the trigger's own text when `content` is
      // empty; every caller here passes content, but keep the guard identical.
      const effective = content !== '' ? content : (element.textContent?.trim() ?? '')
      setOverflowing(
        effective.length > 0 &&
          (element.scrollWidth > element.clientWidth + 1 ||
            element.scrollHeight > element.clientHeight + 1),
      )
    }
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(update) : undefined
    observer?.observe(element)
    const fallbackFrame = observer === undefined ? requestAnimationFrame(update) : undefined
    return () => {
      observer?.disconnect()
      if (fallbackFrame !== undefined) cancelAnimationFrame(fallbackFrame)
    }
  }, [element, content])

  return (
    <Tooltip content={content} isEnabled={overflowing}>
      {to !== undefined ? (
        <RouteLink
          to={to}
          ref={setElement}
          tabIndex={overflowing ? 0 : undefined}
          aria-label={ariaLabel}
          {...stylex.props(xstyle)}
        >
          {children}
        </RouteLink>
      ) : (
        <Tag
          ref={setElement}
          tabIndex={overflowing ? 0 : undefined}
          aria-label={ariaLabel}
          {...stylex.props(xstyle)}
        >
          {children}
        </Tag>
      )}
    </Tooltip>
  )
}

/**
 * Credentials that need attention — classic HealthProblemCollection.vue,
 * including its LedgerRecordList grid (role=table) with the scroll-hint
 * overflow affordance and the clear/inactive empty panel.
 */
export function HealthProblemCollection({
  items,
  recoveryByCredential,
  statsWindowSeconds,
  availableCount,
}: {
  items: HealthProblemItem[]
  recoveryByCredential: Record<number, RecoveryDisplay | undefined>
  statsWindowSeconds: number
  availableCount: number
}) {
  const intl = useIntl()
  const t = useT()
  const n = (value: number): string => intl.formatNumber(value)
  const statsWindowMinutes = Math.max(1, Math.round(statsWindowSeconds / 60))

  function statusLabel(item: HealthProblemItem): string {
    return t(`monitor.health.problems.${item.kind}`)
  }

  function credentialMeta(credential: HealthProblemCredentialDto): string {
    const result = [t('monitor.health.problems.credentialMeta', { group: credential.group_name })]
    if (credential.last_status_code !== null) {
      result.push(String(credential.last_status_code))
    }
    result.push(t(`monitor.health.failureCategories.${credential.last_failure_category}`))
    return result.join(' · ')
  }

  // LedgerRecordList overflow wiring: the classic measured on mount, on every
  // update and on window resize. ResizeObserver already fires once on observe()
  // (the mount measure) and re-running the effect when the row count changes
  // reproduces the update measure.
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
  }, [items.length])

  const label = t('monitor.health.problems.tableLabel')
  const scrollHint = t('monitor.scrollHint')
  const accessibleLabel = overflowing && scrollHint !== '' ? `${label} · ${scrollHint}` : label
  const clear = availableCount > 0

  return (
    <section {...stylex.props(styles.section)} aria-labelledby="problem-health-title">
      <MonitorSectionHeading
        id="problem-health-title"
        title={t('monitor.health.problems.title')}
        description={t('monitor.health.problems.description', {
          minutes: statsWindowMinutes,
        })}
        meta={
          items.length > 0 ? t('monitor.health.problems.count', { count: items.length }) : undefined
        }
      />

      {items.length === 0 ? (
        <div
          {...stylex.props(styles.clearPanel, !clear && styles.clearPanelInactive)}
          role="status"
          aria-label={t('monitor.health.problems.empty')}
        >
          <div {...stylex.props(styles.clearMain)}>
            <Badge
              variant={badgeVariants[clear ? 'success' : 'neutral']}
              icon={
                clear ? (
                  <CircleCheck size={12} aria-hidden="true" />
                ) : (
                  <CircleHelp size={12} aria-hidden="true" />
                )
              }
              label={
                clear
                  ? t('monitor.health.problems.empty')
                  : t('monitor.health.problems.inactiveTitle')
              }
            />
            <div {...stylex.props(styles.clearCopy)}>
              <span>
                {clear
                  ? t('monitor.health.problems.emptyDescription', {
                      count: availableCount,
                    })
                  : t('monitor.health.problems.inactiveDescription')}
              </span>
              <small {...stylex.props(styles.clearHint)}>
                {t('monitor.health.problems.emptyHint')}
              </small>
            </div>
          </div>
          <span {...stylex.props(styles.clearMeta)}>
            {t('monitor.health.problems.windowLabel', {
              minutes: statsWindowMinutes,
            })}
          </span>
        </div>
      ) : (
        <div
          ref={listRef}
          {...stylex.props(styles.list)}
          role="table"
          data-testid="ledger-record-list"
          aria-label={accessibleLabel}
          aria-rowcount={items.length + 1}
          tabIndex={overflowing ? 0 : undefined}
        >
          <div
            {...stylex.props(styles.headRow)}
            role="row"
            data-testid="ledger-record-list__header"
            aria-rowindex={1}
          >
            <span role="columnheader" {...stylex.props(styles.headCell)}>
              {t('monitor.health.problems.columns.identity')}
            </span>
            <span role="columnheader" {...stylex.props(styles.headCell)}>
              {t('monitor.health.problems.columns.status')}
            </span>
            <span role="columnheader" {...stylex.props(styles.headCell)}>
              {t('monitor.health.problems.columns.window')}
            </span>
            <span role="columnheader" {...stylex.props(styles.headCell)}>
              {t('monitor.health.problems.columns.recovery')}
            </span>
            <span role="columnheader" {...stylex.props(styles.headCell, styles.actionsHeading)}>
              {t('monitor.health.problems.columns.actions')}
            </span>
          </div>

          {items.map((item, index) => {
            const credential = item.credential
            const recovery = recoveryByCredential[credential.credential_id]
            const windowLabel = t('monitor.health.problems.window', {
              minutes: statsWindowMinutes,
              problems: credential.recent_problem_count,
              successes: credential.recent_success_count,
            })
            const consecutiveLabel = t('monitor.health.problems.consecutive', {
              count: credential.consecutive_problem_count,
            })
            const meta = credentialMeta(credential)
            const StatusIcon = badgeIcons[item.tone]
            return (
              <article
                key={credential.credential_id}
                {...stylex.props(styles.record, index === 0 && styles.recordFirst)}
                role="row"
                aria-rowindex={index + 2}
              >
                <div role="cell" {...stylex.props(styles.cell, styles.stackCell)}>
                  <OverflowTip
                    content={credential.identity}
                    to={groupCredentialsHref(credential.group_id)}
                    xstyle={styles.identityLink}
                  >
                    {credential.identity}
                  </OverflowTip>
                  <OverflowTip as="small" content={meta} xstyle={styles.recordMeta}>
                    {meta}
                  </OverflowTip>
                </div>

                <div role="cell" {...stylex.props(styles.cell)}>
                  <Badge
                    variant={badgeVariants[item.tone]}
                    icon={<StatusIcon size={12} aria-hidden="true" />}
                    label={statusLabel(item)}
                  />
                </div>

                <div role="cell" {...stylex.props(styles.cell, styles.stackCell)}>
                  <OverflowTip
                    content={windowLabel}
                    ariaLabel={windowLabel}
                    xstyle={styles.windowSummary}
                  >
                    <span>
                      {t('monitor.health.problems.windowCompactPrefix', {
                        minutes: statsWindowMinutes,
                      })}
                    </span>
                    <span
                      {...stylex.props(
                        credential.recent_problem_count === 0
                          ? styles.windowValueSuccess
                          : styles.windowValueDanger,
                      )}
                    >
                      {n(credential.recent_problem_count)}
                    </span>
                    <span aria-hidden="true">/</span>
                    <span
                      {...stylex.props(
                        credential.recent_success_count === 0
                          ? styles.windowValueDanger
                          : styles.windowValueSuccess,
                      )}
                    >
                      {n(credential.recent_success_count)}
                    </span>
                  </OverflowTip>
                  <OverflowTip as="small" content={consecutiveLabel} xstyle={styles.recordMeta}>
                    {consecutiveLabel}
                  </OverflowTip>
                </div>

                <div role="cell" {...stylex.props(styles.cell, styles.stackCell)}>
                  {recovery !== undefined && (
                    <>
                      <Tooltip content={recovery.exact} placement="below">
                        <span {...stylex.props(styles.recoveryTime)} tabIndex={0}>
                          {t(`monitor.health.problems.${recovery.labelKey}`, {
                            time: recovery.relative,
                          })}
                        </span>
                      </Tooltip>
                      <OverflowTip
                        as="small"
                        content={t(`monitor.health.problems.${recovery.hintKey}`)}
                        xstyle={styles.recordMeta}
                      >
                        {t(`monitor.health.problems.${recovery.hintKey}`)}
                      </OverflowTip>
                    </>
                  )}
                </div>

                <div role="cell" {...stylex.props(styles.cell, styles.actionsCell)}>
                  <IconButton
                    variant="ghost"
                    size="sm"
                    label={t('monitor.health.problems.viewLogs', {
                      credential: credential.identity,
                    })}
                    icon={<ScrollText size={15} aria-hidden="true" />}
                    href={credentialLogsHref(credential.group_id, credential.credential_id)}
                  />
                  <IconButton
                    variant="ghost"
                    size="sm"
                    label={t('monitor.health.problems.viewGroup', {
                      group: credential.group_name,
                    })}
                    icon={<ArrowRight size={15} aria-hidden="true" />}
                    href={groupCredentialsHref(credential.group_id)}
                  />
                </div>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
