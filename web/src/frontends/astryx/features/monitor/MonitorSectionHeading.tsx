import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { CircleHelp } from 'lucide-react'
import type { ReactNode } from 'react'

import { useT } from '../../app/i18n'

const NARROW = '@media (max-width: 520px)'

const styles = stylex.create({
  heading: {
    display: 'flex',
    minWidth: 0,
    minHeight: 38,
    alignItems: {
      default: 'center',
      [NARROW]: 'flex-start',
    },
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    backgroundColor: 'color-mix(in srgb, var(--color-surface-sunken) 52%, var(--color-surface))',
    borderRadius: 'var(--radius-tag)',
    paddingBlock: 6,
    paddingInline: 10,
  },
  title: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2-5)',
  },
  // Classic renders the leading accent dot as a ::before pseudo-element; StyleX
  // has no pseudo-elements, so it is a real aria-hidden span instead.
  dot: {
    flex: '0 0 auto',
    width: 6,
    height: 6,
    borderRadius: '50%',
    backgroundColor: 'var(--color-action)',
  },
  h2: {
    minWidth: 0,
    margin: 0,
    fontSize: 'var(--title-section)',
    fontWeight: 650,
    letterSpacing: '-0.01em',
    lineHeight: 'var(--line-compact)',
  },
  help: {
    display: 'inline-flex',
    flex: '0 0 auto',
    width: { default: 32, [NARROW]: 44 },
    height: { default: 32, [NARROW]: 44 },
    margin: { default: -5, [NARROW]: -11 },
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 0,
    borderRadius: 999,
    backgroundColor: {
      default: 'transparent',
      ':hover': 'var(--color-surface-sunken)',
    },
    color: {
      default: 'var(--color-text-faint)',
      ':hover': 'var(--color-text-muted)',
    },
    cursor: 'help',
    padding: 0,
    outlineWidth: { ':focus-visible': 2 },
    outlineStyle: { ':focus-visible': 'solid' },
    outlineColor: { ':focus-visible': 'var(--color-action)' },
    outlineOffset: { ':focus-visible': 2 },
  },
  meta: {
    flex: '0 0 auto',
    marginLeft: 'auto',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    fontVariantNumeric: 'tabular-nums',
  },
  actions: {
    display: 'flex',
    flex: '0 0 auto',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
})

/**
 * Section heading for monitor panels — classic MonitorSectionHeading.vue.
 * The `actions` prop is the React counterpart of the classic `actions` slot.
 */
export function MonitorSectionHeading({
  title,
  description,
  id,
  meta,
  actions,
}: {
  title: string
  description?: string
  id?: string
  meta?: string
  actions?: ReactNode
}) {
  const t = useT()

  return (
    <header {...stylex.props(styles.heading)}>
      <div {...stylex.props(styles.title)}>
        <span {...stylex.props(styles.dot)} aria-hidden="true" />
        <h2 id={id} {...stylex.props(styles.h2)}>
          {title}
        </h2>
        {description ? (
          // touchTrigger="tap": the icon button's only action is revealing the
          // hint, exactly the case the DS tooltip reserves tap triggering for.
          <Tooltip content={description} placement="below" alignment="start" touchTrigger="tap">
            <button
              type="button"
              {...stylex.props(styles.help)}
              aria-label={`${title}: ${t('monitor.help')}`}
            >
              <CircleHelp size={15} strokeWidth={1.8} aria-hidden="true" />
            </button>
          </Tooltip>
        ) : null}
      </div>
      {meta ? <span {...stylex.props(styles.meta)}>{meta}</span> : null}
      {actions ? <div {...stylex.props(styles.actions)}>{actions}</div> : null}
    </header>
  )
}
