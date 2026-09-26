import * as stylex from '@stylexjs/stylex'
import { Badge, Tooltip } from '@astryxdesign/core'
import { CircleAlert, CircleHelp, CircleOff, PencilLine, RotateCcw } from 'lucide-react'
import { useState, type ReactNode } from 'react'

const narrowRow = '@media (max-width: 800px)'
const noHover = '@media (hover: none)'
const narrowBlock = '@media (max-width: 560px)'

const styles = stylex.create({
  hint: {
    display: 'inline-flex',
    flex: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    width: '18px',
    height: '18px',
    borderWidth: 0,
    borderRadius: 'var(--radius-tag)',
    backgroundColor: { default: 'transparent', ':hover': 'var(--color-surface-sunken)' },
    color: { default: 'var(--color-text-faint)', ':hover': 'var(--color-text)' },
    padding: 0,
    cursor: 'help',
    outlineWidth: { ':focus-visible': '2px' },
    outlineStyle: { ':focus-visible': 'solid' },
    outlineColor: { ':focus-visible': 'var(--color-focus)' },
    outlineOffset: { ':focus-visible': '2px' },
  },
  action: {
    display: 'inline-flex',
    flex: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    width: '27px',
    height: '27px',
    borderWidth: 0,
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'transparent',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.46 },
    transitionProperty: 'background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
    outlineWidth: { ':focus-visible': '2px' },
    outlineStyle: { ':focus-visible': 'solid' },
    outlineColor: { ':focus-visible': 'var(--color-focus)' },
    outlineOffset: { ':focus-visible': '2px' },
  },
  actionToneAction: {
    color: 'var(--color-action)',
    backgroundColor: { ':hover:not(:disabled)': 'var(--color-action-soft)' },
  },
  actionToneWarning: {
    color: 'var(--color-warning)',
    backgroundColor: { ':hover:not(:disabled)': 'var(--color-warning-bg)' },
  },
  actionReveal: {
    opacity: { default: 0, [noHover]: 1, [narrowRow]: 1 },
    transitionProperty: 'opacity, background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  row: {
    display: 'grid',
    gridTemplateColumns: {
      default: '216px minmax(0, 1fr)',
      [narrowRow]: 'minmax(0, 1fr)',
    },
    alignItems: 'center',
    columnGap: 'var(--space-4)',
    rowGap: { [narrowRow]: 'var(--space-2)' },
    borderLeftWidth: '2px',
    borderLeftStyle: 'solid',
    borderLeftColor: 'transparent',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingInline: '12px 10px',
  },
  rowDivided: {
    borderBottomWidth: '1px',
    borderBottomStyle: 'dashed',
    borderBottomColor: 'var(--color-border-subtle)',
  },
  rowEditing: {
    borderLeftColor: 'var(--color-action)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
  },
  rowIdentity: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    minWidth: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontWeight: 600,
    lineHeight: 1.4,
  },
  rowIdentityEditing: {
    color: 'var(--color-text)',
  },
  rowCluster: {
    display: 'flex',
    minHeight: '28px',
    flexWrap: 'wrap',
    alignItems: 'center',
    minWidth: 0,
    gap: 'var(--space-3)',
  },
  rowValue: {
    minWidth: 0,
  },
  rowPlain: {
    color: 'var(--color-text)',
    fontSize: 'var(--text-body)',
    fontVariantNumeric: 'tabular-nums',
  },
  block: {
    display: 'grid',
    gap: 'var(--space-3)',
    borderLeftWidth: '2px',
    borderLeftStyle: 'solid',
    borderLeftColor: 'transparent',
    paddingLeft: '12px',
  },
  blockEditing: {
    borderLeftColor: 'var(--color-action)',
  },
  blockHeading: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) auto',
      [narrowBlock]: 'minmax(0, 1fr)',
    },
    alignItems: 'start',
    gap: 'var(--space-4)',
  },
  blockIdentity: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    minWidth: 0,
  },
  blockTitle: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontWeight: 600,
  },
  blockTitleEditing: {
    color: 'var(--color-text)',
  },
  blockMeta: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    justifyContent: { [narrowBlock]: 'flex-start' },
  },
  blockCount: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  },
})

function HintButton({ label, help }: { label: string; help: string }) {
  return (
    <Tooltip content={help}>
      <button
        type="button"
        {...stylex.props(styles.hint)}
        aria-label={`${label} · ${help}`}
      >
        <CircleHelp size={13} aria-hidden />
      </button>
    </Tooltip>
  )
}

function ActionButton({
  actionLabel,
  label,
  overridden,
  disabled,
  reveal,
  onToggle,
}: {
  actionLabel: string
  label: string
  overridden: boolean
  disabled: boolean
  reveal: boolean
  onToggle(): void
}) {
  return (
    <Tooltip content={actionLabel}>
      <button
        type="button"
        {...stylex.props(
          styles.action,
          overridden ? styles.actionToneWarning : styles.actionToneAction,
          reveal && styles.actionReveal,
        )}
        aria-label={`${actionLabel} · ${label}`}
        aria-pressed={overridden}
        disabled={disabled}
        onClick={onToggle}
      >
        {overridden ? (
          <RotateCcw size={14} aria-hidden />
        ) : (
          <PencilLine size={14} aria-hidden />
        )}
      </button>
    </Tooltip>
  )
}

export interface SettingRowProps {
  label: string
  value: string
  help?: string
  sourceLabel: string
  actionLabel: string
  overridden?: boolean
  pendingRestore?: boolean
  locked?: boolean
  disabled?: boolean
  divided?: boolean
  onToggle(): void
  control?: ReactNode
}

// Classic reveals the ghosted override affordance via `.row:hover .action` /
// `:focus-within` descendant selectors, which StyleX cannot express — the row
// tracks pointer hover and descendant focus in JS instead.
export function SettingRow({
  label,
  value,
  help,
  sourceLabel,
  actionLabel,
  overridden = false,
  pendingRestore = false,
  locked = false,
  disabled = false,
  divided = true,
  onToggle,
  control,
}: SettingRowProps) {
  const [rowActive, setRowActive] = useState(false)
  return (
    <div
      {...stylex.props(styles.row, divided && styles.rowDivided, overridden && styles.rowEditing)}
      onMouseEnter={() => setRowActive(true)}
      onMouseLeave={() => setRowActive(false)}
      onFocus={() => setRowActive(true)}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setRowActive(false)
        }
      }}
    >
      <div {...stylex.props(styles.rowIdentity, overridden && styles.rowIdentityEditing)}>
        <span>{label}</span>
        {help !== undefined && <HintButton label={label} help={help} />}
      </div>
      <div {...stylex.props(styles.rowCluster)}>
        {(pendingRestore || locked) && (
          <Badge
            variant={locked ? 'neutral' : 'warning'}
            label={sourceLabel}
            icon={
              locked ? (
                <CircleOff size={12} aria-hidden />
              ) : (
                <CircleAlert size={12} aria-hidden />
              )
            }
          />
        )}
        <div {...stylex.props(styles.rowValue)}>
          {overridden ? control : <span {...stylex.props(styles.rowPlain)}>{value}</span>}
        </div>
        {!locked && (
          <ActionButton
            actionLabel={actionLabel}
            label={label}
            overridden={overridden}
            disabled={disabled}
            reveal={!overridden && !rowActive}
            onToggle={onToggle}
          />
        )}
      </div>
    </div>
  )
}

export interface SettingBlockProps {
  title: string
  help?: string
  meta?: string
  sourceLabel: string
  actionLabel: string
  overridden?: boolean
  pendingRestore?: boolean
  disabled?: boolean
  onToggle(): void
  children?: ReactNode
}

export function SettingBlock({
  title,
  help,
  meta,
  sourceLabel,
  actionLabel,
  overridden = false,
  pendingRestore = false,
  disabled = false,
  onToggle,
  children,
}: SettingBlockProps) {
  return (
    <article {...stylex.props(styles.block, overridden && styles.blockEditing)}>
      <header {...stylex.props(styles.blockHeading)}>
        <div {...stylex.props(styles.blockIdentity)}>
          <strong {...stylex.props(styles.blockTitle, overridden && styles.blockTitleEditing)}>
            {title}
          </strong>
          {help !== undefined && <HintButton label={title} help={help} />}
        </div>
        <div {...stylex.props(styles.blockMeta)}>
          {meta !== undefined && <span {...stylex.props(styles.blockCount)}>{meta}</span>}
          {pendingRestore && (
            <Badge
              variant="warning"
              label={sourceLabel}
              icon={<CircleAlert size={12} aria-hidden />}
            />
          )}
          <ActionButton
            actionLabel={actionLabel}
            label={title}
            overridden={overridden}
            disabled={disabled}
            reveal={false}
            onToggle={onToggle}
          />
        </div>
      </header>
      {children}
    </article>
  )
}
